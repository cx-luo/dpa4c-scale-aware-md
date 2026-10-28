"""Isolated state-machine regressions for the explicitly opted-in GC policy."""
import ast
from pathlib import Path
import threading
import types
import unittest


class FakeGC:
    def __init__(self):
        self.thresholds = (2000, 10, 0)
        self.enabled = True
    def get_threshold(self): return self.thresholds
    def set_threshold(self, *values): self.thresholds = tuple(values)
    def isenabled(self): return self.enabled


class GCPolicyTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[2]
        text = (root/'components/lammps-python/package/lammps/core.py').read_text()
        tree = ast.parse(text)
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in {'_acquire_gc_policy', '_release_gc_policy'}]
        self.assertEqual(len(functions), 2)
        self.gc, self.env = FakeGC(), {}
        self.ns = {'_gc': self.gc, 'os': types.SimpleNamespace(environ=self.env),
                   '_LAMMPS_GC_LOCK': threading.RLock(), '_LAMMPS_GC_STATE': None}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(root/'core-policy'), 'exec'), self.ns)
        self.acquire, self.release = self.ns['_acquire_gc_policy'], self.ns['_release_gc_policy']

    def test_opt_out_and_nonlowering(self):
        self.assertFalse(self.acquire())
        self.env['LAMMPS_PYTHON_GC_THRESHOLD'] = '1000'
        self.assertFalse(self.acquire())
        self.assertEqual(self.gc.thresholds, (2000, 10, 0))

    def test_shared_ownership_restores_on_last_release(self):
        self.env['LAMMPS_PYTHON_GC_THRESHOLD'] = '524288'
        self.assertTrue(self.acquire()); self.assertTrue(self.acquire())
        self.assertEqual(self.gc.thresholds, (524288, 10, 0))
        self.release(); self.assertEqual(self.ns['_LAMMPS_GC_STATE']['owners'], 1)
        self.release(); self.assertIsNone(self.ns['_LAMMPS_GC_STATE'])
        self.assertEqual(self.gc.thresholds, (2000, 10, 0))

    def test_disabled_and_zero_threshold_remain_disabled(self):
        self.env['LAMMPS_PYTHON_GC_THRESHOLD'] = '524288'
        self.gc.enabled = False
        self.assertTrue(self.acquire()); self.release()
        self.assertFalse(self.gc.enabled)
        self.gc.thresholds = (0, 17, 2)
        self.assertFalse(self.acquire())
        self.assertEqual(self.gc.thresholds, (0, 17, 2))

    def test_serial_external_threshold_and_enabled_changes_win(self):
        self.env['LAMMPS_PYTHON_GC_THRESHOLD'] = '524288'
        self.assertTrue(self.acquire())
        self.gc.thresholds = (10000, 12, 3)
        self.assertFalse(self.acquire())
        self.release(); self.assertEqual(self.gc.thresholds, (10000, 12, 3))
        self.assertIsNone(self.ns['_LAMMPS_GC_STATE'])
        self.assertTrue(self.acquire()); self.gc.enabled = False
        self.release(); self.assertFalse(self.gc.enabled)
        self.assertEqual(self.gc.thresholds, (10000, 12, 3))

    def test_invalid_and_active_escalation_leave_ownership_intact(self):
        for value in ['invalid', '-1']:
            self.env['LAMMPS_PYTHON_GC_THRESHOLD'] = value
            with self.assertRaises(ValueError): self.acquire()
            self.assertIsNone(self.ns['_LAMMPS_GC_STATE'])
        self.env['LAMMPS_PYTHON_GC_THRESHOLD'] = '524288'
        self.assertTrue(self.acquire())
        self.env['LAMMPS_PYTHON_GC_THRESHOLD'] = '1048576'
        with self.assertRaises(ValueError): self.acquire()
        self.assertEqual(self.ns['_LAMMPS_GC_STATE']['owners'], 1)
        self.release(); self.assertEqual(self.gc.thresholds, (2000, 10, 0))


if __name__ == '__main__':
    unittest.main()
