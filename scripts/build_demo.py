#!/usr/bin/env python3
"""Import WAV files, regenerate comparable spectrograms, and build a static demo.

Full/partial tree: python scripts/build_demo.py --audio-root /path/to/results
One method:       python scripts/build_demo.py --audio-root /path/to/wavs --method proposed
Rebuild only:     python scripts/build_demo.py
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft, resample_poly

ROOT = Path(__file__).resolve().parents[1]
METHODS = [
    ("clean", "Clean", "Reference", ("clean", "reference")),
    ("noisy", "Noisy", "Input", ("noisy", "input")),
    ("apollo", "Apollo", "Baseline", ("apollo",)),
    ("mmaudio-flow", "MMAudio-Flow", "Baseline", ("mmaudio-flow", "mmaudio_flow", "mmaudioflow", "mmaudio")),
    ("sonicmaster", "SonicMaster", "Baseline", ("sonicmaster", "sonic_master", "sonic-master")),
    ("proposed", "Proposed", "Our method", ("enh", "proposed", "ours")),
]
N_FFT, HOP, DB_MIN, DB_MAX = 512, 128, -80, 0


def read_audio(path: Path):
    rate, pcm = wavfile.read(path)
    if pcm.ndim == 1:
        pcm = pcm[:, None]
    if pcm.size == 0:
        raise ValueError(f"Empty audio: {path}")
    if pcm.dtype == np.uint8:
        audio = (pcm.astype(np.float64) - 128) / 128
    elif np.issubdtype(pcm.dtype, np.signedinteger):
        audio = pcm.astype(np.float64) / (2 ** (8 * pcm.dtype.itemsize - 1))
    elif np.issubdtype(pcm.dtype, np.floating):
        audio = pcm.astype(np.float64)
    else:
        raise ValueError(f"Unsupported WAV encoding: {path}")
    if not np.isfinite(audio).all():
        raise ValueError(f"Non-finite audio samples: {path}")
    return int(rate), audio


def sha256(path: Path):
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def natural_key(value):
    return (0, int(value)) if value.isdigit() else (1, value)


def wav_files(directory: Path):
    return sorted((p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".wav"), key=lambda p: natural_key(p.stem))


def plan_import(source: Path, method: str | None):
    if not source.is_dir():
        raise ValueError(f"Audio folder does not exist: {source}")
    imports = []
    if method:
        files = wav_files(source)
        if not files:
            raise ValueError(f"No WAV files in {source}")
        imports.extend((method, path) for path in files)
    else:
        directories = {p.name.lower(): p for p in source.iterdir() if p.is_dir()}
        for key, _, _, aliases in METHODS:
            matches = [directories[a] for a in aliases if a in directories]
            if len(matches) > 1:
                raise ValueError(f"Multiple folders for {key}: {matches}. Keep one, or use --method.")
            if matches:
                imports.extend((key, path) for path in wav_files(matches[0]))
        if not imports:
            raise ValueError("No recognized method folders found. For a flat WAV folder, add --method proposed (or another method).")
    seen = set()
    for key, path in imports:
        target = (key, path.stem)
        if target in seen:
            raise ValueError(f"Duplicate sample ID for {key}: {path.stem}")
        seen.add(target)
        read_audio(path)  # Validate the complete import before replacing any file.
    return imports


def import_audio(source: Path | None, method: str | None):
    imports = plan_import(source, method) if source else []
    reference_ids = {p.stem for p in (ROOT / "audio/clean").glob("*.wav")}
    reference_ids.update(p.stem for key, p in imports if key == "clean")
    if not reference_ids:
        raise ValueError("No clean references found. First import a tree with a clean/ folder.")
    for key, path in imports:
        if path.stem not in reference_ids:
            raise ValueError(f"No matching clean reference for {key}/{path.name}; use the same sample filename.")
    for key, path in imports:
        destination = ROOT / "audio" / key / f"{path.stem}.wav"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.resolve() != path.resolve():
            shutil.copy2(path, destination)
        if sha256(destination) != sha256(path):
            raise RuntimeError(f"Copy verification failed: {path}")
    return imports


def spectrum(audio, rate):
    # SciPy's spectrum scaling divides STFT amplitudes by sum(window).
    # Fixed reference amplitude 1.0; no per-file normalization or gain change.
    padded = np.pad(audio, ((0, max(0, N_FFT - len(audio))), (0, 0)))
    freq, time, z = stft(padded.T, fs=rate, window="hann", nperseg=N_FFT,
                         noverlap=N_FFT - HOP, nfft=N_FFT,
                         boundary="zeros", padded=True, axis=-1)
    power = np.mean(np.abs(z) ** 2, axis=0)
    db = 10 * np.log10(np.maximum(power, 1e-12))
    return freq, time, db


def analysis_audio(audio, rate, analysis_rate):
    if rate == analysis_rate:
        return audio
    divisor = math.gcd(rate, analysis_rate)
    return resample_poly(audio, analysis_rate // divisor, rate // divisor, axis=0)


def render_plot(path, audio, rate, duration_max, frequency_max, analysis_rate):
    analyzed = analysis_audio(audio, rate, analysis_rate)
    freq, time, db = spectrum(analyzed, analysis_rate)
    duration = len(audio) / rate
    fig, ax = plt.subplots(figsize=(4.4, 3.0), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f0eef2")
    # Pad-to-display areas remain gray (unavailable data), never synthesized silence.
    valid_time = time <= duration
    t = time[valid_time]
    half_hop = HOP / analysis_rate / 2
    edges = np.r_[max(0, t[0] - half_hop), (t[:-1] + t[1:]) / 2, duration]
    f_edges = np.r_[0, (freq[:-1] + freq[1:]) / 2, analysis_rate / 2]
    ax.pcolormesh(edges, f_edges / 1000, db[:, valid_time], cmap="magma",
                  vmin=DB_MIN, vmax=DB_MAX, rasterized=True, shading="flat")
    ax.set_xlim(0, duration_max)
    ax.set_ylim(0, frequency_max / 1000)
    ax.set_xticks(np.linspace(0, duration_max, 4))
    ax.set_xticklabels([f"{value:.0f}" if abs(value - round(value)) < .02 else f"{value:.1f}" for value in np.linspace(0, duration_max, 4)])
    ax.set_yticks(np.linspace(0, frequency_max / 1000, 5))
    ax.set_xlabel("Time (s)", fontsize=9, labelpad=5, color="#736b80")
    ax.set_ylabel("Frequency (kHz)", fontsize=9, labelpad=5, color="#736b80")
    ax.tick_params(axis="both", labelsize=8, colors="#847b8e", length=2.5, width=.5, pad=3)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.subplots_adjust(left=.15, right=.978, bottom=.2, top=.96)
    fig.savefig(path, facecolor="white", metadata={"Software": "Music restoration demo / SciPy STFT"})
    plt.close(fig)


def escape(value):
    return html.escape(str(value), quote=True)


def method_html(sample_id, key, name, role, record):
    class_name = f"method {key}" if key in ("clean", "noisy", "proposed") else "method"
    tag = '<span class="ours-tag">OURS</span>' if key == "proposed" else ""
    head = f'<article class="{class_name}"><div class="method-head"><div><h4>{escape(name)}</h4></div>{tag}</div>'
    if record:
        title = f"Sample {sample_id.zfill(2)} · {name}"
        return head + f'''<a class="plot" href="{escape(record['spectrogram'])}" data-title="{escape(title)}" aria-label="Enlarge spectrogram: {escape(title)}"><img src="{escape(record['spectrogram'])}" alt="{escape(title)} spectrogram; time in seconds and frequency in kHz, fixed minus 80 to zero dBFS scale" width="880" height="600" loading="lazy"></a>
<div class="audio-wrap"><audio controls preload="none" src="{escape(record['audio'])}" data-duration="{record['duration']}" data-label="{escape(title)}" aria-label="{escape(title)}"></audio><div class="audio-bottom"><span class="time">0:00</span></div></div></article>'''
    return head + '<div class="pending-plot" role="img" aria-label="Spectrogram pending"><svg viewBox="0 0 28 28" aria-hidden="true"><rect x="4" y="4" width="20" height="20" rx="4"/><path d="M9 14h10m-5-5v10"/></svg><span>Awaiting results</span></div><div class="pending-footer">Audio pending</div></article>'


def build(frequency_max, analysis_rate):
    image_dir = ROOT / "assets/spectrograms"
    image_dir.mkdir(parents=True, exist_ok=True)
    plt.imsave(ROOT / "assets/scale.png", np.tile(np.linspace(0, 1, 256), (12, 1)), cmap="magma", vmin=0, vmax=1)
    sample_ids = sorted((p.stem for p in (ROOT / "audio/clean").glob("*.wav")), key=natural_key)
    manifest = {"spectrogram": {"fft_size": N_FFT, "hop_length": HOP, "window": "hann", "analysis_sample_rate": analysis_rate, "resampling": "polyphase resampling of analysis copies only", "scale": "dB relative to STFT amplitude 1.0; scipy spectrum scaling", "limits_db": [DB_MIN, DB_MAX], "frequency_max_hz": frequency_max, "channel_aggregation": "mean power", "colormap": "magma", "time_alignment": "not applied", "audio_gain_normalization": False}, "samples": []}
    sections, nav, notes = [], [], []
    for sample_id in sample_ids:
        signals, records = {}, {}
        for key, name, role, _ in METHODS:
            path = ROOT / "audio" / key / f"{sample_id}.wav"
            if path.exists():
                rate, audio = read_audio(path)
                signals[key] = (rate, audio)
                records[key] = {"audio": path.relative_to(ROOT).as_posix(), "spectrogram": f"assets/spectrograms/{sample_id}-{key}.png", "duration": len(audio) / rate, "sample_rate": rate, "channels": audio.shape[1], "frames": len(audio), "sha256": sha256(path)}
            else:
                records[key] = None
        maximum = max(record["duration"] for record in records.values() if record)
        reference_duration = records["clean"]["duration"]
        unequal = [key for key, record in records.items() if record and abs(record["duration"] - reference_duration) > max(.5, reference_duration * .05)]
        if unequal:
            names = {key: name for key, name, _, _ in METHODS}
            details = "; ".join(f'{names[key]} {records[key]["duration"]:.2f} s' for key in unequal)
            notes.append(f'Sample {sample_id.zfill(2)}: clean reference {reference_duration:.2f} s; {details}.')
        for key, (rate, audio) in signals.items():
            render_plot(ROOT / records[key]["spectrogram"], audio, rate, maximum, frequency_max, analysis_rate)
        columns = "\n".join(method_html(sample_id, key, name, role, records[key]) for key, name, role, _ in METHODS)
        flag = '<span class="duration-flag">Different clip lengths</span>' if unequal else ""
        sections.append(f'''<section class="sample" id="sample-{escape(sample_id)}" aria-labelledby="title-{escape(sample_id)}"><div class="sample-header"><h3 class="sample-number" id="title-{escape(sample_id)}"><span class="sr-only">Sample </span>{escape(sample_id.zfill(2))}</h3>{flag}</div><div class="method-grid">{columns}</div></section>''')
        nav.append(f'<a href="#sample-{escape(sample_id)}" aria-label="Go to Sample {escape(sample_id.zfill(2))}">{escape(sample_id.zfill(2))}</a>')
        manifest["samples"].append({"id": sample_id, "plot_duration": maximum, "duration_warning_methods": unequal, "methods": records})
        print(f"Sample {sample_id}: {len(signals)} audio files / plots")
    page = (ROOT / "templates/index.html").read_text(encoding="utf-8")
    replacements = {"@@SAMPLES@@": "\n".join(sections), "@@NAV@@": "".join(nav), "@@COUNT@@": str(len(sample_ids)).zfill(2), "@@FREQUENCY@@": f"{frequency_max / 1000:g}", "@@ANALYSIS_RATE@@": f"{analysis_rate / 1000:g}", "@@DURATION_NOTE@@": '<p><strong>Duration note.</strong> ' + escape(" ".join(notes)) + '</p>' if notes else ""}
    for key, value in replacements.items():
        page = page.replace(key, value)
    (ROOT / "index.html").write_text(page, encoding="utf-8")
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "data/samples.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Built {ROOT / 'index.html'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio-root", type=Path, help="Tree of method folders, or one flat WAV folder with --method")
    parser.add_argument("--method", choices=[m[0] for m in METHODS], help="Import flat WAV folder as this method")
    parser.add_argument("--analysis-rate", type=int, default=16000, help="Common sample rate for spectrogram analysis copies only (default: 16000)")
    parser.add_argument("--max-frequency", type=float, default=None, help="Shared frequency-axis maximum in Hz (default: analysis rate / 2)")
    args = parser.parse_args()
    if args.method and args.audio_root is None:
        parser.error("--method requires --audio-root")
    if args.analysis_rate <= 0:
        parser.error("--analysis-rate must be positive")
    if args.max_frequency is None:
        args.max_frequency = args.analysis_rate / 2
    if not math.isfinite(args.max_frequency) or not 0 < args.max_frequency <= args.analysis_rate / 2:
        parser.error("--max-frequency must be positive and no greater than half --analysis-rate")
    try:
        imports = import_audio(args.audio_root, args.method)
        for key, _, _, _ in METHODS:
            (ROOT / "audio" / key).mkdir(parents=True, exist_ok=True)
        print(f"Imported {len(imports)} WAV files (verified SHA-256). Existing methods retained.")
        build(args.max_frequency, args.analysis_rate)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
