"""Offline tests for UI helpers and worker integration (no Streamlit required)."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.ui.alignment_view import expand_alignment, parse_alignment
from src.processing.pipeline.config import folder
from src.ui.file_browser import (
    assign_to_raw, available_files, parse_csv_preview, safe_file, store_import,
)
from src.processing.pipeline.runner import run_stage


class BrowserTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "data" / "raw").mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def test_list_and_safe_read(self):
        p = self.root / "data" / "raw" / "x.txt"
        p.write_text("Test", encoding="utf-8")
        self.assertEqual(available_files(self.root), [p])
        self.assertEqual(safe_file(self.root, "data/raw/x.txt"), p)
        with self.assertRaises(ValueError):
            safe_file(self.root, "../unsafe.txt")

    def test_import_and_assign_without_overwrite(self):
        p = store_import(self.root, "../sample.txt", "Text".encode())
        self.assertEqual(p.read_text(), "Text")
        dest = assign_to_raw(self.root, p, "original-rus")
        self.assertEqual(dest.read_text(), "Text")
        with self.assertRaises(FileExistsError):
            assign_to_raw(self.root, p, "original-rus")
        p2 = store_import(self.root, "sample.txt", "Again".encode())
        self.assertNotEqual(p, p2)

    def test_csv(self):
        heads, rows, truncated = parse_csv_preview("a,b\n1,2\n3,4\n5,6\n", max_rows=2)
        self.assertEqual(heads, ["a", "b"])
        self.assertEqual(len(rows), 2)
        self.assertTrue(truncated)


class AlignmentTests(unittest.TestCase):
    def test_parse_and_expand_gaps(self):
        parsed = parse_alignment("[0, 1]:[0]:0.428256\n[2]:[]:0.000000\n[3]:[1, 2]:0.315\n")
        self.assertEqual(len(parsed), 3)
        expanded = expand_alignment(parsed, ["a", "b", "c", "d"], ["z", "x", "y"])
        self.assertEqual(expanded[0]["Оригинал"], "a\nb")
        self.assertEqual(expanded[1]["Перевод"], "∅")
        self.assertTrue(expanded[1]["Пропуск"])
        self.assertFalse(expanded[2]["Индекс вне диапазона"])

    def test_out_of_range(self):
        item = [{"ru_ids": [31], "en_ids": [0], "cost": 0.1}]
        self.assertTrue(expand_alignment(item, ["a"], ["b"])[0]["Индекс вне диапазона"])

    def test_invalid_format_rejected(self):
        with self.assertRaises(ValueError):
            parse_alignment("this is not an alignment")


class PipelineTests(unittest.TestCase):
    def test_clean_standalone(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            p = folder(root, "raw") / "original-rus.txt"
            p.parent.mkdir(parents=True)
            p.write_text("Привет", encoding="utf-8")
            import types
            fake = types.ModuleType("text_normalizer")
            fake.clean_text = lambda content: content + "!"
            with patch.dict(sys.modules, {"src.processing.preprocessing.cleaning": fake}):
                run_stage(root, "clean", ["original-rus"], 2)
            self.assertEqual((folder(root, "cleaned") / "original-rus.txt").read_text(), "Привет!")

    def test_all_requires_russian_and_translation(self):
        with self.assertRaisesRegex(ValueError, "оригинал"):
            run_stage(Path("."), "all", ["glenny-translation-en"], 2)

    def test_worker_subprocess_log_and_status(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "src").mkdir()
            shutil.copytree(Path(__file__).resolve().parents[1], root / "src" / "ui",
                            ignore=shutil.ignore_patterns("__pycache__", "tests"))
            shutil.copytree(Path(__file__).resolve().parents[2] / "external_tools",
                            root / "src" / "external_tools",
                            ignore=shutil.ignore_patterns("__pycache__"))
            shutil.copytree(Path(__file__).resolve().parents[2] / "processing",
                            root / "src" / "processing",
                            ignore=shutil.ignore_patterns("__pycache__", "data"))
            raw = folder(root, "raw") / "original-rus.txt"
            raw.parent.mkdir(parents=True)
            raw.write_text("  Текст  ", encoding="utf-8")
            from src.ui.jobs import read_status, start_job, tail_log
            job = start_job(root, "clean", ["original-rus"], 2)
            result = job["process"].wait(timeout=15)
            self.assertEqual(result, 0, tail_log(root, job["id"]))
            self.assertEqual(read_status(root, job["id"])["state"], "done")
            self.assertIn("Очистка", tail_log(root, job["id"]))
            self.assertEqual((folder(root, "cleaned") / "original-rus.txt").read_text(), "Текст")


if __name__ == "__main__":
    unittest.main()
