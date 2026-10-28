#!/usr/bin/env python3
"""Small staging/identity regressions; no native build, GPU or MD execution."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from python_overlay import inventory, source_lock, stage_python_overlay, verify_python_overlay


class OverlayTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[2]
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.stage = Path(self.tmp.name)
        self.engine = {'environment': {'PYTHONPATH': str(self.stage / 'python')}}
        stage_python_overlay(self.root, self.stage)

    def test_complete_source_locked_stage(self):
        _, lock = source_lock(self.root)
        self.assertEqual(inventory(self.stage / 'python'), lock['files'])
        record = verify_python_overlay(self.root, self.stage, self.engine)
        self.assertEqual(len(record['files']), 15)
        self.assertEqual(verify_python_overlay(self.root, self.stage, self.engine, record), record)

    def test_missing_and_modified_core_rejected(self):
        core = self.stage / 'python/lammps/core.py'
        original = core.read_bytes()
        core.write_bytes(original + b'\n# altered\n')
        with self.assertRaises(ValueError): verify_python_overlay(self.root, self.stage, self.engine)
        core.unlink()
        with self.assertRaises(ValueError): verify_python_overlay(self.root, self.stage, self.engine)

    def test_import_path_and_stale_manifest_rejected(self):
        record = verify_python_overlay(self.root, self.stage, self.engine)
        wrong = copy.deepcopy(self.engine)
        wrong['environment']['PYTHONPATH'] += ':/tmp/other-package'
        with self.assertRaises(ValueError): verify_python_overlay(self.root, self.stage, wrong)
        record['core_sha256'] = '0' * 64
        with self.assertRaises(ValueError): verify_python_overlay(self.root, self.stage, self.engine, record)

    def test_overwrite_and_extra_package_member_rejected(self):
        with self.assertRaises(ValueError): stage_python_overlay(self.root, self.stage)
        (self.stage / 'python/lammps/extra.py').write_text('')
        with self.assertRaises(ValueError): verify_python_overlay(self.root, self.stage, self.engine)


if __name__ == '__main__':
    unittest.main()
