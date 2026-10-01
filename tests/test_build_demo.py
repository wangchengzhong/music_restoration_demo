"""Regression checks for audio replacement, placeholders, and comparable spectra.

Run: python -m unittest discover -s tests -v
All audio is generated in temporary test directories; published examples are untouched.
"""
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import wavfile

SITE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_demo", SITE / "scripts/build_demo.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class DemoTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="aatcc-demo-test-")
        self.base = Path(self.temporary.name)
        self.site = self.base / "site"
        self.site.mkdir()
        for folder in ("templates", "scripts"):
            shutil.copytree(SITE / folder, self.site / folder)
        (self.site / "assets").mkdir()

    def tearDown(self):
        self.temporary.cleanup()

    def write_wav(self, folder, seconds=1, sample="2", amplitude=.5):
        folder.mkdir(parents=True, exist_ok=True)
        signal = amplitude * np.sin(2 * np.pi * 1000 * np.arange(int(16000 * seconds)) / 16000)
        path = folder / f"{sample}.wav"
        wavfile.write(path, 16000, (signal * 32767).astype(np.int16))
        return path

    def run_cli(self, *arguments, expect_ok=True):
        result = subprocess.run([sys.executable, str(self.site / "scripts/build_demo.py"), *map(str, arguments)], capture_output=True, text=True)
        if expect_ok:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
        return result

    def manifest(self):
        return json.loads((self.site / "data/samples.json").read_text())["samples"][0]

    def test_corrected_clip_and_incremental_baseline(self):
        source = self.base / "source"
        clean = self.write_wav(source / "clean")
        self.write_wav(source / "noisy")
        short = self.write_wav(source / "enh", .3)
        self.run_cli("--audio-root", source)
        initial = self.manifest()
        self.assertEqual(initial["duration_warning_methods"], ["proposed"])
        self.assertIsNone(initial["methods"]["apollo"])
        self.assertEqual(builder.sha256(short), initial["methods"]["proposed"]["sha256"])
        corrected = self.write_wav(self.base / "corrected")
        self.run_cli("--audio-root", corrected.parent, "--method", "proposed")
        apollo = self.write_wav(self.base / "baseline", amplitude=.25)
        self.run_cli("--audio-root", apollo.parent, "--method", "apollo")
        final = self.manifest()
        self.assertEqual(final["duration_warning_methods"], [])
        self.assertEqual(final["methods"]["clean"]["sha256"], builder.sha256(clean))
        self.assertEqual(final["methods"]["proposed"]["sha256"], builder.sha256(corrected))
        self.assertEqual(final["methods"]["apollo"]["sha256"], builder.sha256(apollo))
        self.assertIsNone(final["methods"]["sonicmaster"])
        page = (self.site / "index.html").read_text()
        self.assertEqual(page.count("<audio controls"), 4)
        self.assertEqual(page.count("Awaiting results"), 2)
        self.assertNotIn("Duration note.", page)
        self.assertTrue((self.site / final["methods"]["apollo"]["spectrogram"]).is_file())

    def test_unmatched_file_rejected_before_any_copy(self):
        source = self.base / "bad"
        self.write_wav(source / "clean", sample="1")
        self.write_wav(source / "enh", sample="2")
        result = self.run_cli("--audio-root", source, expect_ok=False)
        self.assertIn("No matching clean reference", result.stderr)
        self.assertFalse((self.site / "audio/clean/1.wav").exists())

    def test_duplicate_method_alias_rejected(self):
        source = self.base / "ambiguous"
        for name in ("clean", "enh", "proposed"):
            self.write_wav(source / name)
        result = self.run_cli("--audio-root", source, expect_ok=False)
        self.assertIn("Multiple folders for proposed", result.stderr)

    def test_spectra_keep_level_and_stereo_phase(self):
        time = np.arange(16000) / 16000
        signal = np.sin(2 * np.pi * 1000 * time)
        freq, _, mono = builder.spectrum(signal[:, None], 16000)
        _, _, quieter = builder.spectrum(.5 * signal[:, None], 16000)
        _, _, stereo = builder.spectrum(np.column_stack([signal, -signal]), 16000)
        index = np.argmin(abs(freq - 1000))
        self.assertAlmostEqual(float(np.mean(mono[index, 4:-4] - quieter[index, 4:-4])), 6.0206, places=3)
        np.testing.assert_allclose(mono, stereo, atol=1e-10)
        self.assertAlmostEqual(float(np.mean(mono[index, 4:-4])), -6.0206, places=3)

    def test_mixed_sample_rates_share_analysis_timebase(self):
        low = np.sin(2 * np.pi * 1000 * np.arange(16000) / 16000)[:, None]
        high = np.sin(2 * np.pi * 1000 * np.arange(48000) / 48000)[:, None]
        path = self.base / "high.wav"
        wavfile.write(path, 48000, high.astype(np.float32))
        before = builder.sha256(path)
        rate, data = builder.read_audio(path)
        converted = builder.analysis_audio(data, rate, 16000)
        f1, t1, db1 = builder.spectrum(low, 16000)
        f2, t2, db2 = builder.spectrum(converted, 16000)
        np.testing.assert_array_equal(f1, f2)
        np.testing.assert_array_equal(t1, t2)
        peak = np.argmin(abs(f1 - 1000))
        np.testing.assert_allclose(db1[peak, 4:-4], db2[peak, 4:-4], atol=.02)
        self.assertEqual(before, builder.sha256(path))


if __name__ == "__main__":
    unittest.main()
