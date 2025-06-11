import os
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
GENRE_MODEL_PATH     = 'essentia_models/genre/genre_rosamerica-discogs-effnet-1.pb'
EMBEDDING_MODEL_PATH = 'essentia_models/discogs/discogs-effnet-bs64-1.pb'
GENRE_LABELS_PATH    = 'essentia_models/genre/genre_rosamerica-discogs-effnet-1.json'

# Load genre labels
with open(GENRE_LABELS_PATH) as f:
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

# Prepare a clean temp directory
if TEMP_DIR.exists():
    shutil.rmtree(TEMP_DIR)
TEMP_DIR.mkdir(parents=True)

# Copy source files to temp
for src in MUSIC_DIR.glob('*'):
    if src.suffix.lower() in ['.mp3', '.flac', '.wav']:
        shutil.copy2(src, TEMP_DIR / src.name)


def convert_to_wav(filepath: Path, sample_rate: int = 16000) -> Path:
    """
    Convert audio file to mono WAV at given sample_rate for analysis.
    Returns the path to the temporary WAV file.
    """
    wav_path = filepath.with_suffix(f'.temp_{sample_rate}.wav')
    subprocess.run([
        'ffmpeg', '-y', '-i', str(filepath),
        '-ac', '1', '-ar', str(sample_rate), str(wav_path)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return wav_path


def extract_features(filepath: Path):
    """
    Extract features: BPM, root key, loudness, embedding, moods, genres.
    Uses 16 kHz WAV for all models.
    Prints top 5 genres to stdout.
    Returns a tuple:
      (bpm, root_key, moods_str, top_3_genres_str, lufs, gain)
    """
    # Convert to one 16 kHz WAV file
    wav16 = convert_to_wav(filepath, sample_rate=16000)

    # Load audio for all analysis (16 kHz)
    audio = MonoLoader(filename=str(wav16), sampleRate=16000)()

    # Rhythm / BPM
    bpm, *_ = RhythmExtractor2013(method='multifeature')(audio)

    # Key / scale
    key, scale, _ = KeyExtractor()(audio)
    root_key = f"{key} {scale}"

    # Loudness measures (LUFS and RMS)
    gain = ReplayGain()(audio)
    lufs = gain
    rms = RMS()(audio)

    # Embedding for both mood & genre predictions
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
    moods = [
        (lbl.capitalize(), sc)
        for lbl, sc in zip(MOOD_LABELS, mood_pred)
    ]
    top_moods = sorted(moods, key=lambda x: -x[1])[:5]
    moods_str = ", ".join(lbl for lbl, _ in top_moods)

    # Genre prediction
    genre_pred = TensorflowPredict2D(
        graphFilename=GENRE_MODEL_PATH,
        input='model/Placeholder',
        output='model/Softmax'
    )(embedding).flatten()
    genres_sorted = sorted(
        zip(GENRE_LABELS, genre_pred),
        key=lambda x: -x[1]
    )

    # Prepare top genres list
    top5_genres = [name for name, _ in genres_sorted[:5]]
    print(f"Top 5 genres for {filepath.name}: {', '.join(top5_genres)}")

    # Use only top 3 genres for metadata
    top_3_genres_str = ", ".join(top5_genres[:3])

    # Clean up intermediate WAV file
    if wav16.exists():
        wav16.unlink()

    return (
        round(bpm),
        root_key,
        moods_str,
        top_3_genres_str,
        round(lufs, 2),
        round(gain, 2)
    )


def write_metadata(original_path: Path, bpm, root_key, moods, genres, lufs, gain):
    """
    Write metadata tags back to the original file.
    Supports MP3 (ID3-TXXX), FLAC (Vorbis comments), WAV (RIFF INFO).
    """
    suffix = original_path.suffix.lower()
    if suffix == '.mp3':
        try:
            tags = ID3(original_path)
        except ID3NoHeaderError:
            tags = ID3()
        fields = [
            ('BPM', bpm),
            ('ROOT_KEY', root_key),
            ('MOODS', moods),
            ('GENRE', genres),
            ('LUFS', lufs),
            ('LUFS_GAIN', gain)
        ]
        for desc, val in fields:
            tags.setall(
                f"TXXX:{desc}",
                [TXXX(encoding=3, desc=desc, text=str(val))]
            )
        tags.save(original_path)

    elif suffix == '.flac':
        audio = FLAC(original_path)
        audio['BPM']         = str(bpm)
        audio['ROOT_KEY']    = root_key
        audio['MOODS']       = moods
        audio['GENRE']       = genres
        audio['LUFS']        = str(lufs)
        audio['LUFS_GAIN']   = str(gain)
        audio.save()

    elif suffix == '.wav':
        audio = WAVE(original_path)
        info = audio.tags or {}
        info['BPM']         = str(bpm)
        info['ROOT_KEY']    = root_key
        info['MOODS']       = moods
        info['GENRE']       = genres
        info['LUFS']        = str(lufs)
        info['LUFS_GAIN']   = str(gain)
        audio.tags = info
        audio.save()


def process_temp():
    """
    Process all files in temp, extract features, and write metadata.
    """
    for temp_file in TEMP_DIR.glob('*'):
        if temp_file.suffix.lower() not in ['.mp3', '.flac', '.wav']:
            continue
        features = extract_features(temp_file)
        original = MUSIC_DIR / temp_file.name
        write_metadata(original, *features)


if __name__ == '__main__':
    process_temp()
