"""Validate pinned source archives, optionally copy them into the source tree.

Does not download, extract, compile, or execute dependency contents.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / 'production/dpa4c-runtime'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--from-dir', type=Path, help='directory containing deepmd.tar.gz and lammps.tar.gz')
    group.add_argument('--check', action='store_true', help='verify already staged archives')
    parser.add_argument('--verify-only', action='store_true', help='with --from-dir, validate without copying')
    args = parser.parse_args()
    lock = json.loads((RUNTIME / 'third_party/sources.lock.json').read_text())
    validated = []
    for row in lock['sources']:
        dst = RUNTIME / row['archive']
        if not dst.resolve().is_relative_to((RUNTIME / 'third_party/archives').resolve()):
            raise ValueError('Archive lock escapes the dependency directory.')
        src = args.from_dir / dst.name if args.from_dir else dst
        if not src.is_file() or digest(src) != row['archive_sha256']:
            raise SystemExit('Missing archive or SHA256 mismatch: ' + str(src))
        if args.from_dir and dst.exists() and digest(dst) != row['archive_sha256']:
            raise SystemExit('Refusing to replace a different destination: ' + str(dst))
        validated.append((src, dst, row))
    # All input and existing destination hashes were verified before any copy.
    if args.from_dir and not args.verify_only:
        for src, dst, row in validated:
            if dst.exists():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            with src.open('rb') as inp, dst.open('xb') as out:
                shutil.copyfileobj(inp, out)
            if digest(dst) != row['archive_sha256']:
                raise SystemExit('Copied archive verification failed: ' + str(dst))
    print(json.dumps({'status': 'PASS', 'copied': bool(args.from_dir and not args.verify_only),
                      'archives': [{'name': row['name'], 'bytes': src.stat().st_size,
                                    'sha256': row['archive_sha256']} for src, dst, row in validated]}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
