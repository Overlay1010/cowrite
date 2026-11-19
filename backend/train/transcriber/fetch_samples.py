"""학습용 녹음 샘플: FluidR3 GM (Frank Wen, CC BY 3.0 — midi-js-soundfonts mp3)에서 음 높이가 있는 악기만,
3반음마다 받아 22050Hz 모노 wav로. python -I fetch_samples.py <저장 폴더>"""
import io
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

BASE = "https://gleitz.github.io/midi-js-soundfonts/FluidR3_GM/"
FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
NAMES = {
    0: "acoustic_grand_piano", 2: "electric_grand_piano", 4: "electric_piano_1", 5: "electric_piano_2", 6: "harpsichord",
    8: "celesta", 9: "glockenspiel", 10: "music_box", 11: "vibraphone", 12: "marimba", 13: "xylophone", 14: "tubular_bells",
    16: "drawbar_organ", 17: "percussive_organ", 19: "church_organ", 21: "accordion",
    24: "acoustic_guitar_nylon", 25: "acoustic_guitar_steel", 26: "electric_guitar_jazz", 27: "electric_guitar_clean",
    32: "acoustic_bass", 33: "electric_bass_finger", 38: "synth_bass_1", 39: "synth_bass_2",
    40: "violin", 42: "cello", 45: "pizzicato_strings", 46: "orchestral_harp", 48: "string_ensemble_1", 50: "synth_strings_1",
    52: "choir_aahs", 53: "voice_oohs", 54: "synth_choir", 56: "trumpet", 61: "brass_section", 62: "synth_brass_1",
    65: "alto_sax", 71: "clarinet", 73: "flute", 75: "pan_flute", 79: "ocarina",
    80: "lead_1_square", 81: "lead_2_sawtooth", 82: "lead_3_calliope", 84: "lead_5_charang", 85: "lead_6_voice", 87: "lead_8_bass__lead",
    88: "pad_1_new_age", 89: "pad_2_warm", 90: "pad_3_polysynth", 91: "pad_4_choir", 92: "pad_5_bowed", 93: "pad_6_metallic", 94: "pad_7_halo", 95: "pad_8_sweep",
    96: "fx_1_rain", 98: "fx_3_crystal", 99: "fx_4_atmosphere", 100: "fx_5_brightness", 102: "fx_7_echoes",
    104: "sitar", 107: "koto", 108: "kalimba", 112: "tinkle_bell", 114: "steel_drums",
}
BASS = {32, 33, 38, 39, 87}


def main(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    total = 0
    for prog, name in NAMES.items():
        d = out / f"{prog:03d}_{name}"
        d.mkdir(exist_ok=True)
        lo, hi = (28, 64) if prog in BASS else (36, 96)
        for p in range(lo, hi + 1, 3):
            f = d / f"{p}.wav"
            if f.exists():
                continue
            url = f"{BASE}{name}-mp3/{FLAT[p % 12]}{p // 12 - 1}.mp3"
            try:
                data = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "sample-fetch"}), timeout=20).read()
                x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
            except Exception as e:  # 없는 음은 건너뜀
                print("skip", url, type(e).__name__, file=sys.stderr)
                continue
            x = x.mean(1)
            x = resample_poly(x, 22050 // 50, sr // 50).astype(np.float32) if sr != 22050 else x
            sf.write(f, x, 22050)
            total += 1
            time.sleep(0.05)
        print(prog, name, len(list(d.glob("*.wav"))), flush=True)
    print("받은 파일", total)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
