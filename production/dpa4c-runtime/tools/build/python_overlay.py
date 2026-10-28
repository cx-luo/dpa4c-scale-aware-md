#!/usr/bin/env python3
"""Stage and verify the source-owned LAMMPS Python package without native writes."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory(directory):
    result = {}
    for path in sorted(directory.rglob('*')):
        if '__pycache__' in path.parts or path.suffix == '.pyc':
            continue
        if path.is_symlink():
            raise ValueError('Python package must not contain symlinks: ' + str(path))
        if path.is_file():
            result[path.relative_to(directory).as_posix()] = sha(path)
    return result


def source_lock(root):
    component = Path(root) / 'components/lammps-python'
    lock_path = component / 'package-manifest.json'
    lock = json.loads(lock_path.read_text())
    if lock.get('schema_version') != 1 or lock.get('package') != 'lammps':
        raise ValueError('unsupported Python package lock')
    files = lock.get('files', {})
    if not files or 'lammps/core.py' not in files:
        raise ValueError('missing locked Python core')
    for rel, digest in files.items():
        path = Path(rel)
        if path.is_absolute() or '..' in path.parts or path.parts[0] != 'lammps' or path.suffix != '.py':
            raise ValueError('unsafe Python package member: ' + rel)
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('invalid Python SHA: ' + rel)
    if inventory(component / 'package') != files:
        raise ValueError('source Python package differs from its lock')
    return lock_path, lock


def stage_python_overlay(root, stage):
    _, lock = source_lock(root)
    source = Path(root) / 'components/lammps-python/package'
    target = Path(stage) / 'python'
    if target.exists() and any(target.iterdir()):
        raise ValueError('refusing to replace a populated Python overlay')
    for rel in lock['files']:
        dest = target / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / rel, dest)
    if inventory(target) != lock['files']:
        raise ValueError('staged Python package SHA mismatch')


def verify_python_overlay(root, stage, engine, recorded=None):
    lock_path, lock = source_lock(root)
    stage = Path(stage).resolve()
    target = stage / 'python'
    if target.is_symlink() or not target.is_dir() or inventory(target) != lock['files']:
        raise ValueError('staged Python package missing or different from source lock')
    if engine.get('environment', {}).get('PYTHONPATH') != str(target):
        raise ValueError('engine PYTHONPATH must select only the staged Python package')
    record = {
        'schema_version': 1,
        'source_lock': 'components/lammps-python/package-manifest.json',
        'source_lock_sha256': sha(lock_path),
        'pythonpath': str(target),
        'core_sha256': lock['files']['lammps/core.py'],
        'files': {'python/' + rel: digest for rel, digest in lock['files'].items()},
    }
    if recorded is not None and recorded != record:
        raise ValueError('manifest Python overlay identity differs from source, stage or engine')
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stage-dir', type=Path, required=True)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if not args.verify:
        stage_python_overlay(args.root, args.stage_dir)
    engine = json.loads((args.stage_dir / 'config/engine.json').read_text())
    print(json.dumps(verify_python_overlay(args.root, args.stage_dir, engine), indent=2))


if __name__ == '__main__':
    main()
