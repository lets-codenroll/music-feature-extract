from mutagen.id3 import ID3, TXXX, ID3NoHeaderError
from pathlib import Path
import sys

def read_metadata(mp3_path):
    try:
        tags = ID3(mp3_path)
        bpm = float(tags.getall("TXXX:BPM")[0].text[0])
        lufs = float(tags.getall("TXXX:LUFS")[0].text[0])
        rms = float(tags.getall("TXXX:LUFS_GAIN")[0].text[0])  # נשתמש ב-GAIN כתחליף ל-RMS
        return bpm, lufs, rms, tags
    except Exception:
        print(f"⚠️  Skipping {mp3_path.name}: Missing or invalid metadata.")
        return None

def calculate_energy_level(bpm, lufs, rms):
    if bpm >= 140 or rms >= 0.08 or lufs >= -10:
        return 5
    elif bpm >= 120 or rms >= 0.06 or lufs >= -12:
        return 4
    elif bpm >= 100 or rms >= 0.04 or lufs >= -14:
        return 3
    elif bpm >= 80 or rms >= 0.02 or lufs >= -16:
        return 2
    else:
        return 1

def write_energy_level(mp3_path, tags, energy):
    tags.setall("TXXX:ENERGY_LEVEL", [TXXX(encoding=3, desc="ENERGY_LEVEL", text=str(energy))])
    tags.save(mp3_path)

def analyze_file(mp3_path):
    result = read_metadata(mp3_path)
    if not result:
        return
    bpm, lufs, rms, tags = result
    energy = calculate_energy_level(bpm, lufs, rms)
    write_energy_level(mp3_path, tags, energy)

    print(f"\n🎵 {mp3_path.name}")
    print(f"  BPM: {round(bpm)}")
    print(f"  LUFS: {round(lufs, 2)}")
    print(f"  RMS (gain): {round(rms, 4)}")
    print(f"  ENERGY_LEVEL: {energy} ✅ written")

def analyze_folder(folder_path):
    folder = Path(folder_path)
    for file in folder.glob("*.mp3"):
        analyze_file(file)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python energy_level.py <folder-path>")
        sys.exit(1)

    analyze_folder(sys.argv[1])
