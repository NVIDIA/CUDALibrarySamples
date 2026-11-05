# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU integration tests; usage: python test_decoder.py /path/to/nvjpegDecoder."""

import pathlib
import subprocess
import sys
import tempfile
import unittest

from PIL import Image


EXE = str(pathlib.Path(sys.argv.pop(1)).resolve())
WIDTH, HEIGHT = 129, 97


class HardwareDecodeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="nvjpeg hardware ")
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name)
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        rgb = Image.new("RGB", (WIDTH, HEIGHT))
        rgb.putdata([((x * 3) % 256, (y * 5) % 256, (x + y) % 256)
                     for y in range(HEIGHT) for x in range(WIDTH)])
        for sampling, name in [(0, "444"), (1, "422"), (2, "420")]:
            rgb.save(self.inputs / (name + ".jpg"), subsampling=sampling, quality=90)
        Image.new("L", (WIDTH, HEIGHT), 37).save(self.inputs / "gray.jpg")
        rgb.save(self.root / "progressive.jpg", progressive=True)

    def invoke(self, *args, code=0):
        result = subprocess.run([EXE, "-backend", "hardware", *map(str, args)], capture_output=True, text=True, timeout=60)
        log = result.stdout + result.stderr
        self.assertEqual(result.returncode, code, log)
        if code:
            self.assertIn("RESULT status=FAIL", log)
            self.assertNotIn("RESULT status=PASS", log)
        return log

    def decode(self, name, fmt, scale=1):
        out = self.root / "out"
        self.invoke("-i", self.inputs / name, "-fmt", fmt, "-scale", scale, "-raw", "-o", out)
        return (out / f"{name}.{fmt}.s{scale}.raw").read_bytes()

    def test_cli_errors_and_help(self):
        self.assertIn("Usage:", self.invoke("-h"))
        for args in [[], ["-i"], ["-i", "a", "-fmt", "bad"],
                     ["-i", "a", "-scale", "3"], ["-i", "a", "-t", "0"],
                     ["-i", "a", "-t", "1x"], ["-i", "a", "-device", "-1"],
                     ["-i", "a", "-t", "999999999999999999999"], ["-bad", "a"]]:
            with self.subTest(args=args):
                self.invoke(*args, code=2)

    def test_grayscale_geometry_and_independent_pixels(self):
        for scale in [1, 2, 4, 8]:
            w, h = (WIDTH + scale - 1) // scale, (HEIGHT + scale - 1) // scale
            for fmt in ["y", "yuv", "unchanged"]:
                with self.subTest(scale=scale, fmt=fmt):
                    data = self.decode("gray.jpg", fmt, scale)
                    self.assertEqual(data, bytes([37]) * (w * h))

    def test_nv12_matches_planar_yuv(self):
        for scale in [1, 2, 4, 8]:
            with self.subTest(scale=scale):
                w, h = (WIDTH + scale - 1) // scale, (HEIGHT + scale - 1) // scale
                ysize, csize = w * h, ((w + 1) // 2) * ((h + 1) // 2)
                planar = self.decode("420.jpg", "yuv", scale)
                nv12 = self.decode("420.jpg", "nv12", scale)
                self.assertEqual(len(planar), ysize + 2 * csize)
                self.assertEqual(len(nv12), len(planar))
                self.assertEqual(nv12[:ysize], planar[:ysize])
                self.assertEqual(nv12[ysize::2], planar[ysize:ysize + csize])
                self.assertEqual(nv12[ysize + 1::2], planar[ysize + csize:])

    def test_yuy2_matches_planar_yuv_with_odd_width(self):
        for scale in [1, 2, 4, 8]:
            with self.subTest(scale=scale):
                w, h = (WIDTH + scale - 1) // scale, (HEIGHT + scale - 1) // scale
                cw = (w + 1) // 2
                planar = self.decode("422.jpg", "yuv", scale)
                packed = self.decode("422.jpg", "yuy2", scale)
                self.assertEqual(len(planar), w * h + 2 * cw * h)
                self.assertEqual(len(packed), 4 * cw * h)
                for row in range(h):
                    line = packed[row * 4 * cw:(row + 1) * 4 * cw]
                    # The last pair's second Y is outside the image for odd widths.
                    if w % 2:
                        self.assertEqual(line[-2], 0)
                    self.assertEqual(line[::2][:w], planar[row * w:(row + 1) * w])
                    self.assertEqual(line[1::4], planar[w * h + row * cw:w * h + (row + 1) * cw])
                    self.assertEqual(line[3::4], planar[w * h + cw * h + row * cw:
                                                     w * h + cw * h + (row + 1) * cw])

    def test_rgb_interleaving_and_luma(self):
        for fmt, packed_fmt in [("rgb", "rgbi"), ("bgr", "bgri")]:
            planar = self.decode("444.jpg", fmt)
            packed = self.decode("444.jpg", packed_fmt)
            size = WIDTH * HEIGHT
            self.assertEqual(len(planar), 3 * size)
            for c in range(3):
                self.assertEqual(packed[c::3], planar[c * size:(c + 1) * size])
        self.assertEqual(self.decode("420.jpg", "y"), self.decode("420.jpg", "yuv")[:WIDTH * HEIGHT])

    def test_goldens_repetition_and_mismatch(self):
        image = self.inputs / "420.jpg"
        golden = self.root / "goldens"
        self.invoke("-i", image, "-fmt", "nv12", "-scale", 2, "-o", golden)
        log = self.invoke("-i", image, "-fmt", "nv12", "-scale", 2, "-t", 3, "-reference", golden)
        self.assertIn("decoded=3", log)
        path = golden / "420.jpg.nv12.s2.raw"
        data = bytearray(path.read_bytes())
        data[0] ^= 1
        path.write_bytes(data)
        self.invoke("-i", image, "-fmt", "nv12", "-scale", 2, "-reference", golden, code=4)
        path.unlink()
        self.invoke("-i", image, "-fmt", "nv12", "-scale", 2, "-reference", golden, code=1)
        # Odd-width YUY2 goldens exclude the last pair's undefined second Y.
        image = self.inputs / "422.jpg"
        self.invoke("-i", image, "-fmt", "yuy2", "-o", golden)
        self.invoke("-i", image, "-fmt", "yuy2", "-t", 3, "-reference", golden)

    def test_recursive_paths_and_exact_counts(self):
        nested = self.inputs / "nested"
        nested.mkdir()
        (nested / "420.jpg").write_bytes((self.inputs / "420.jpg").read_bytes())
        out = self.root / "output with spaces"
        log = self.invoke("-i", self.inputs, "-fmt", "yuv", "-t", 10, "-o", out)
        self.assertIn("decoded=10", log)
        self.assertEqual(len(list(out.rglob("*.raw"))), 5)
        self.assertTrue((out / "nested" / "420.jpg.yuv.s1.raw").is_file())

    def test_default_mode_bmp_and_mixed_batch(self):
        mixed = self.root / "mixed"
        mixed.mkdir()
        (mixed / "a_baseline.jpg").write_bytes((self.inputs / "420.jpg").read_bytes())
        (mixed / "z_progressive.jpg").write_bytes((self.root / "progressive.jpg").read_bytes())
        out = self.root / "bmp"
        log = self.invoke("-backend", "default", "-i", mixed, "-b", 3, "-t", 5, "-w", 1, "-o", out)
        self.assertIn("decoded=5", log)
        for name in ["a_baseline.bmp", "z_progressive.bmp"]:
            with Image.open(out / name) as image:
                self.assertEqual(image.size, (WIDTH, HEIGHT))

    def test_default_mode_raw_and_partial_batch(self):
        # Goldens belong to a backend/configuration; CPU and HW rounding can differ.
        self.invoke("-backend", "default", "-i", self.inputs / "420.jpg",
                    "-fmt", "nv12", "-o", self.root / "out")
        log = self.invoke("-backend", "default", "-i", self.inputs / "420.jpg",
                          "-fmt", "nv12", "-b", 2, "-t", 3, "-reference", self.root / "out")
        self.assertIn("decoded=3", log)

    def test_failures_never_report_pass(self):
        self.invoke("-i", self.root / "progressive.jpg", code=3)
        bad = self.root / "bad.jpg"
        bad.write_bytes(b"this is not a JPEG")
        self.invoke("-i", bad, code=1)
        self.invoke("-i", self.root / "missing.jpg", code=1)
        empty = self.root / "empty"
        empty.mkdir()
        self.invoke("-i", empty, code=1)
        self.invoke("-i", self.inputs / "420.jpg", "-o", bad, code=1)
        # An incompatible format is rejected either by support query or decode.
        result = subprocess.run([EXE, "-backend", "hardware", "-i", str(self.inputs / "444.jpg"), "-fmt", "nv12"],
                                capture_output=True, text=True, timeout=60)
        self.assertIn(result.returncode, [1, 3], result.stdout + result.stderr)
        self.assertNotIn("RESULT status=PASS", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
