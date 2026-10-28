"""Run only archived CPU component tests and data checks in an isolated copy."""
from pathlib import Path
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    if sys.version_info < (3, 12):
        raise SystemExit('Use Python 3.12 or newer.')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1')
    results = []
    for name in ['check_source.py', 'verify_results.py']:
        r = subprocess.run([sys.executable, '-B', str(ROOT / 'scripts' / name)],
                           cwd=ROOT, env=env, capture_output=True, text=True, encoding='utf-8', timeout=60)
        if r.returncode:
            print(r.stdout + r.stderr)
            return r.returncode
        results.append({'check': name, 'status': 'PASS', 'details': json.loads(r.stdout)})
    syntax_files = list((ROOT / 'production/dpa4c-runtime').rglob('*.py')) + list((ROOT / 'scripts').glob('*.py'))
    for path in syntax_files:
        ast.parse(path.read_bytes(), filename=path.relative_to(ROOT).as_posix())
    scratch_parent = ROOT / 'checks'
    scratch_parent.mkdir(exist_ok=True)
    if not scratch_parent.resolve().is_relative_to(ROOT) or scratch_parent.is_symlink():
        raise ValueError('Temporary check directory must stay inside the repository.')
    with tempfile.TemporaryDirectory(prefix='cpu-', dir=scratch_parent) as name:
        scratch = Path(name).resolve()
        if not scratch.is_relative_to(scratch_parent.resolve()):
            raise ValueError('Unexpected temporary path.')
        target = scratch / 'runtime'
        shutil.copytree(ROOT / 'production/dpa4c-runtime', target,
                        ignore=shutil.ignore_patterns('__pycache__', 'archives'))
        env.update(TMP=str(scratch), TEMP=str(scratch), TMPDIR=str(scratch))
        for test in ['test_python_overlay.py', 'test_python_api.py', 'test_python_gc_policy.py']:
            r = subprocess.run([sys.executable, '-B', str(target / 'tools/build' / test)],
                               cwd=target, env=env, capture_output=True, text=True, encoding='utf-8', timeout=60)
            if r.returncode:
                print(r.stdout + r.stderr)
                return r.returncode
            count = re.search(r'Ran (\d+) tests?', r.stdout + r.stderr)
            results.append({'check': test, 'status': 'PASS', 'tests': int(count.group(1)) if count else None})
    print(json.dumps({'status': 'PASS', 'python': sys.version.split()[0], 'syntax_files': len(syntax_files),
                      'checks': results, 'native_build': 'not_run', 'md_or_accelerator_tests': 'not_run'}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
