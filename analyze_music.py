import subprocess
import shutil
from pathlib import Path
from mutagen.id3 import ID3, TXXX, ID3NoHeaderError
from mutagen.flac import FLAC
from mutagen.wave import WAVE
from essentia.standard import (
    MonoLoader, RhythmExtractor2013, KeyExtractor, ReplayGain,
    TensorflowPredictEffnetDiscogs, TensorflowPredict2D, RMS
)
import json

# Working directories
SESSION_DIR = Path(__file__).parent.resolve()
MUSIC_DIR   = SESSION_DIR / 'music'
TEMP_DIR    = SESSION_DIR / 'temp'

# Prediction models
MOOD_MODEL_PATH      = 'essentia_models/mood_mirex/mtg_jamendo_moodtheme-discogs-effnet-1.pb'
GENRE_MODEL_PATH     = 'essentia_models/genre/genre_discogs400-discogs-effnet-1.pb'
EMBEDDING_MODEL_PATH = 'essentia_models/discogs/discogs-effnet-bs64-1.pb'
GENRE_LABELS_PATH    = 'essentia_models/genre/genre_discogs400-discogs-effnet-1.json'

# Load genre labels
with open(GENRE_LABELS_PATH, 'r') as f:
    GENRE_LABELS = json.load(f)['classes']

# Allowed mood labels
MOOD_LABELS = [
    "action", "adventure", "advertising", "background", "ballad", "calm",
    "children", "christmas", "commercial", "cool", "corporate", "dark",
    "deep", "documentary", "drama", "dramatic", "dream", "emotional",
    "energetic", "epic", "fast", "film", "fun", "funny", "game", "groovy",
    "happy", "heavy", "holiday", "hopeful", "inspiring", "love",
    "meditative", "melancholic", "melodic", "motivational", "movie",
    "nature", "party", "positive", "powerful", "relaxing", "retro",
    "romantic", "sad", "sexy", "slow", "soft", "soundscape", "space",
    "sport", "summer", "trailer", "travel", "upbeat", "uplifting"
]

def prepare_temp_directory():
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    TEMP_DIR.mkdir(parents=True)
    for src in MUSIC_DIR.glob('*'):
        if src.suffix.lower() in ('.mp3', '.flac', '.wav'):
            shutil.copy2(src, TEMP_DIR / src.name)

def convert_to_wav(filepath: Path, sample_rate: int = 16000) -> Path:
    wav_path = filepath.with_suffix(f'.temp_{sample_rate}.wav')
    subprocess.run([
        'ffmpeg', '-y', '-i', str(filepath),
        '-ac', '1', '-ar', str(sample_rate), str(wav_path)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return wav_path

def extract_features(filepath: Path):
    # convert and load at 16 kHz
    wav16 = convert_to_wav(filepath, sample_rate=16000)
    audio = MonoLoader(filename=str(wav16), sampleRate=16000)()
    # BPM
    bpm, *_ = RhythmExtractor2013(method='multifeature')(audio)
    # Key
    key, scale, _ = KeyExtractor()(audio)
    root_key = f"{key} {scale}"
    # Loudness
    gain = ReplayGain()(audio)
    lufs = gain
    # Embedding
    embedder = TensorflowPredictEffnetDiscogs(
        graphFilename=EMBEDDING_MODEL_PATH,
        output='PartitionedCall:1'
    )
    embedding = embedder(audio)
    # Mood prediction
    mood_pred = TensorflowPredict2D(
        graphFilename=MOOD_MODEL_PATH,
        input='model/Placeholder',
        output='model/Sigmoid'
    )(embedding).flatten()
    moods = sorted(
        [(lbl.capitalize(), sc) for lbl, sc in zip(MOOD_LABELS, mood_pred)],
        key=lambda x: -x[1]
    )[:5]
    moods_str = ", ".join(lbl for lbl, _ in moods)
    # Genre prediction
    genre_pred = TensorflowPredict2D(
        graphFilename=GENRE_MODEL_PATH,
        input='serving_default_model_Placeholder',
        output='PartitionedCall'
    )(embedding).flatten()
    genres_sorted = sorted(zip(GENRE_LABELS, genre_pred), key=lambda x: -x[1])
    top5 = [name for name, _ in genres_sorted[:5]]
    print(f"Top 5 genres for {filepath.name}: {', '.join(top5)}")
    genres_str = ", ".join(top5[:3])
    # cleanup
    wav16.unlink(missing_ok=True)
    return round(bpm), root_key, moods_str, genres_str, round(lufs, 2), round(gain, 2)

def write_metadata(path: Path, bpm, root_key, moods, genres, lufs, gain):
    suffix = path.suffix.lower()
    if suffix == '.mp3':
        try:
            tags = ID3(path)
        except ID3NoHeaderError:
            tags = ID3()
        for desc, val in [
            ('BPM', bpm), ('ROOT_KEY', root_key), ('MOODS', moods),
            ('GENRE', genres), ('LUFS', lufs), ('LUFS_GAIN', gain)
        ]:
            tags.setall(f"TXXX:{desc}", [TXXX(encoding=3, desc=desc, text=str(val))])
        tags.save(path)
    elif suffix == '.flac':
        audio = FLAC(path)
        audio['BPM']        = str(bpm)
        audio['ROOT_KEY']   = root_key
        audio['MOODS']      = moods
        audio['GENRE']      = genres
        audio['LUFS']       = str(lufs)
        audio['LUFS_GAIN']  = str(gain)
        audio.save()
    elif suffix == '.wav':
        audio = WAVE(path)
        info = audio.tags or {}
        info.update({
            'BPM': str(bpm), 'ROOT_KEY': root_key, 'MOODS': moods,
            'GENRE': genres, 'LUFS': str(lufs), 'LUFS_GAIN': str(gain)
        })
        audio.tags = info
        audio.save()

def process_all():
    prepare_temp_directory()
    for file in TEMP_DIR.glob('*'):
        if file.suffix.lower() not in ('.mp3', '.flac', '.wav'):
            continue
        features = extract_features(file)
        write_metadata(MUSIC_DIR / file.name, *features)

if __name__ == '__main__':
    process_all()
