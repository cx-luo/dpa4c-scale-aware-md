#!/usr/bin/env python3
"""Check the selected Python API against its loaded native library; no MD steps."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import random
import struct
import sys

import lammps.core as core
from lammps import lammps
from lammps.constants import LAMMPS_DOUBLE


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for data in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def main():
    values = [0, 1, -1, 2**31-1, -2**31, 2**32, 2**55+17, -2**55-17, 1 << 1000, -(1 << 1000)]
    rng = random.Random(20261019)
    values += [rng.randrange(-2**40, 2**40) for _ in range(1024)]
    cases = {'Si': [28.0855], 'MgO': [24.305, 15.999], 'CuNi': [63.546, 58.6934]}
    records = []
    with lammps(cmdargs=['-log', 'none', '-screen', 'none']) as lmp:
        assert lmp.extract_atom('mass') is None
        assert lmp._extract_atom_double._flags_ & 4, 'mass getter must retain the Python GIL'
        for label, masses in cases.items():
            lmp.command('clear')
            lmp.command('units metal')
            lmp.command('atom_style atomic')
            lmp.command('region box block 0 4 0 4 0 4 units box')
            lmp.command('create_box ' + str(len(masses)) + ' box')
            for index, mass in enumerate(masses, 1):
                lmp.command(f'mass {index} {mass}')
                lmp.command(f'create_atoms {index} single {index} 1 1 units box')
            states = []
            for mass_delta in [0.0, 0.125]:
                lmp.command('mass 1 ' + str(masses[0] + mass_delta))
                lmp.extract_atom('type')
                generic_restype = lmp.lib.lammps_extract_atom.restype
                current = lmp.extract_atom('mass')
                assert lmp.lib.lammps_extract_atom.restype is generic_restype
                explicit = lmp.extract_atom('mass', LAMMPS_DOUBLE)
                assert ctypes.cast(current, ctypes.c_void_p).value == ctypes.cast(explicit, ctypes.c_void_p).value
                observed = [current[i] for i in range(1, len(masses) + 1)]
                assert observed == [explicit[i] for i in range(1, len(masses) + 1)]
                assert observed == [masses[0] + mass_delta] + masses[1:]
                states.append(observed)
            records.append({'material': label, 'mass_states': states})
        double_bits = []
        for value in [0.0, -0.0, 1e-300, 1e300, float('inf'), float('-inf')]:
            lmp.extract_atom('mass', LAMMPS_DOUBLE)[1] = value
            actual = struct.pack('d', lmp.extract_atom('mass')[1])
            assert actual == struct.pack('d', value)
            double_bits.append(actual.hex())
        lmp.command('mass 1 63.546')
        lmp.lib.lammps_command(lmp.lmp, b'mass 2 59.0')
        assert lmp.extract_atom('mass')[2] == 59.0
        zero = lmp._image_decode_zero
        assert lmp.decode_image_flags(zero) == [0, 0, 0]
        first = lmp.decode_image_flags(zero)
        second = lmp.decode_image_flags(zero)
        assert first is not second
        first[0] = 99
        assert second == lmp.decode_image_flags(zero) == [0, 0, 0]
        word = 1 << (lmp.extract_setting('imageint') * 8)
        values += [zero, zero + word, zero - word]
        native = (ctypes.c_int * 3)()
        digest = hashlib.sha256()
        for value in values:
            lmp.lib.lammps_decode_image_flags(lmp.c_imageint(value), native)
            expected = list(native)
            assert lmp.decode_image_flags(value) == expected
            digest.update(json.dumps(expected, separators=(',', ':')).encode())
        # Non-exact integers retain native argument conversion behavior.
        class IntSubclass(int):
            pass
        for value in [False, True, IntSubclass(5)]:
            lmp.lib.lammps_decode_image_flags(lmp.c_imageint(value), native)
            assert lmp.decode_image_flags(value) == list(native)
        layout = lmp._image_decode_layout
        lmp._image_decode_layout = None
        for value in [zero, -1, 1 << 1000, IntSubclass(zero)]:
            lmp.lib.lammps_decode_image_flags(lmp.c_imageint(value), native)
            assert lmp.decode_image_flags(value) == list(native)
        lmp._image_decode_layout = layout
        maps = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                       if line.split() and line.split()[-1].startswith('/')})
        loaded = [p for p in maps if 'liblammps.so' in p]
        assert loaded, 'LAMMPS library not present in process maps'
        report = {'passed': True, 'scope': 'native API smoke; no MD steps or score',
                  'python': sys.executable, 'pythonpath': os.environ.get('PYTHONPATH'),
                  'core_path': str(Path(core.__file__).resolve()), 'core_sha256': sha(core.__file__),
                  'lammps_version': lmp.version(), 'imageint_bytes': lmp.extract_setting('imageint'),
                  'decode_values': len(values), 'decode_digest': digest.hexdigest(),
                  'native_lammps_maps': {p: sha(p) for p in loaded}, 'materials': records,
                  'mass_ffi_flags': lmp._extract_atom_double._flags_, 'mass_double_bits': double_bits,
                  'mass_current_pointer_and_generic_restype': True, 'zero_fresh_list': True,
                  'unknown_abi_native_fallback': True}
    assert lmp.extract_atom('mass') is None
    report['closed_handle_fallback'] = True
    if len(sys.argv) > 1:
        destination = Path(sys.argv[1])
        with destination.open('x') as handle:
            json.dump(report, handle, indent=2)
            handle.write('\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
