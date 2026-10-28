#!/usr/bin/env python3
"""Exercise the production decoder's real AST for all ABI dispatch branches."""
import ast
import ctypes
from pathlib import Path
import unittest


SOURCE = Path(__file__).resolve().parents[2] / 'components/lammps-python/package/lammps/core.py'
TREE = ast.parse(SOURCE.read_text())
CLASS = next(node for node in TREE.body if isinstance(node, ast.ClassDef) and node.name == 'lammps')
INIT = next(node for node in CLASS.body if isinstance(node, ast.FunctionDef) and node.name == '__init__')
DECODE = next(node for node in CLASS.body if isinstance(node, ast.FunctionDef) and node.name == 'decode_image_flags')


class NativeFallback:
    def __init__(self):
        self.calls = []

    def lammps_decode_image_flags(self, value, output):
        self.calls.append(value)
        target = ctypes.cast(output, ctypes.POINTER(ctypes.c_int))
        target[0], target[1], target[2] = 7, 8, 9


def holder(size):
    namespace = {'c_int': ctypes.c_int, 'byref': ctypes.byref}
    exec(compile(ast.Module(body=[DECODE], type_ignores=[]), str(SOURCE), 'exec'), namespace)
    obj = type('Holder', (), {'decode_image_flags': namespace['decode_image_flags']})()
    assignments = [node for node in INIT.body if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Attribute) and t.attr in
                           {'_image_decode_layout', '_image_decode_zero'} for t in node.targets)]
    assert len(assignments) == 2
    exec(compile(ast.Module(body=assignments, type_ignores=[]), str(SOURCE), 'exec'),
         {'self': obj, 'imageint_size': size})
    obj.lib = NativeFallback()
    return obj


class DecoderTests(unittest.TestCase):
    def test_known_abi_roundtrips_and_oversize(self):
        for size, bits in [(4, 10), (8, 21)]:
            obj = holder(size)
            half = 1 << (bits - 1)
            word = 1 << (size * 8)
            for xyz in [(0, 0, 0), (-half, 0, half-1), (half-1, -half, 3), (-1, 2, -7)]:
                encoded = sum((v + half) << (bits * i) for i, v in enumerate(xyz))
                for offset in [0, word, -word, word * (1 << 1000), -word * (1 << 1000)]:
                    self.assertEqual(obj.decode_image_flags(encoded + offset), list(xyz))
            self.assertEqual(obj.lib.calls, [])

    def test_signed_boundaries_keep_unmasked_z(self):
        for size, bits in [(4, 10), (8, 21)]:
            obj = holder(size)
            half = 1 << (bits - 1)
            sign = 1 << (size * 8 - 1)
            self.assertEqual(obj.decode_image_flags(-1), [half-1, half-1, -half-1])
            self.assertEqual(obj.decode_image_flags(-sign), [-half, -half, -(sign >> (2*bits))-half])
            self.assertEqual(obj.decode_image_flags(sign), obj.decode_image_flags(-sign))
            self.assertEqual(obj.decode_image_flags(sign-1), [half-1, half-1, ((sign-1) >> (2*bits))-half])

    def test_zero_returns_a_fresh_list(self):
        for size in [4, 8]:
            obj = holder(size)
            a = obj.decode_image_flags(obj._image_decode_zero)
            b = obj.decode_image_flags(obj._image_decode_zero)
            self.assertEqual(a, [0, 0, 0])
            self.assertIsNot(a, b)
            a[0] = 99
            self.assertEqual(obj.decode_image_flags(obj._image_decode_zero), [0, 0, 0])

    def test_non_exact_integer_uses_native_fallback(self):
        class IntSubclass(int):
            pass
        for size in [4, 8]:
            obj = holder(size)
            for value in [True, False, IntSubclass(obj._image_decode_zero), ctypes.c_int(3)]:
                self.assertEqual(obj.decode_image_flags(value), [7, 8, 9])
            self.assertEqual(len(obj.lib.calls), 4)

    def test_unknown_abi_uses_native_fallback(self):
        for size in [0, 16, None]:
            obj = holder(size)
            self.assertIsNone(obj._image_decode_layout)
            self.assertIsNone(obj._image_decode_zero)
            for value in [0, 537395712, -1, 1 << 1000]:
                self.assertEqual(obj.decode_image_flags(value), [7, 8, 9])
            self.assertEqual(len(obj.lib.calls), 4)


if __name__ == '__main__':
    unittest.main()
