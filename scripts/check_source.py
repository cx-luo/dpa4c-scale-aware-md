"""Verify frozen source and curated data without importing native libraries."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def safe_path(relative):
    rel = Path(relative)
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError('Unsafe manifest path: ' + relative)
    path = ROOT / rel
    if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
        raise ValueError('Source path escapes repository: ' + relative)
    return path


def main():
    source = json.loads((ROOT / 'metadata/source-files.json').read_text())
    failures = []
    for row in source['files']:
        path = safe_path(row['path'])
        if not path.is_file():
            failures.append({'path': row['path'], 'error': 'missing'})
            continue
        raw = path.read_bytes()
        blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        if len(raw) != row['bytes'] or hashlib.sha256(raw).hexdigest() != row['sha256'] or blob != row['git_blob_sha1']:
            failures.append({'path': row['path'], 'error': 'content differs from frozen Git object'})
    expected = {r['path'] for r in source['files']} | {'production/dpa4c-runtime/docs/MIGRATION.md'}
    actual = {p.relative_to(ROOT).as_posix() for p in (ROOT / 'production/dpa4c-runtime').rglob('*')
              if p.is_file() and '__pycache__' not in p.parts and not p.relative_to(ROOT).as_posix().startswith('production/dpa4c-runtime/third_party/archives/')}
    for path in sorted(actual - expected):
        failures.append({'path': path, 'error': 'untracked addition within frozen runtime'})
    data = json.loads((ROOT / 'metadata/data-files.json').read_text())
    for row in data:
        p = safe_path(row['path'])
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != row['sha256']:
            failures.append({'path': row['path'], 'error': 'curated data digest mismatch'})
    component = ROOT / 'production/dpa4c-runtime/components/lammps-python'
    lock = json.loads((component / 'package-manifest.json').read_text())
    for rel, digest in lock['files'].items():
        p = component / 'package' / rel
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            failures.append({'path': rel, 'error': 'Python overlay lock mismatch'})
    result = {'status': 'PASS' if not failures else 'FAIL', 'source_commit': source['source_commit'],
              'source_files': len(source['files']), 'data_files': len(data),
              'python_overlay_files': len(lock['files']), 'failures': failures,
              'scope': 'source/data integrity; no native execution or physical validation'}
    print(json.dumps(result, indent=2))
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
