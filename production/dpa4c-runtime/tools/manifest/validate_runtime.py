#!/usr/bin/env python3
"""Run bounded runtime gates or verify an existing, hash-bound validation."""
import argparse
import json
import pathlib
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=pathlib.Path, required=True)
    parser.add_argument('--stage-dir', type=pathlib.Path, required=True)
    parser.add_argument('--run-gates', action='store_true')
    parser.add_argument('--assets', type=pathlib.Path)
    parser.add_argument('--control-engine', type=pathlib.Path)
    parser.add_argument('--evidence-dir', type=pathlib.Path)
    args, checks = parser.parse_known_args()
    root, stage = args.root.resolve(), args.stage_dir.resolve()
    verify = [sys.executable, str(root/'tools/manifest/verify_runtime.py'),
              '--root', str(root), '--stage-dir', str(stage)]
    if not args.run_gates:
        return subprocess.call(verify + checks)
    if checks:
        parser.error('check-only options cannot be combined with --run-gates')
    if not all([args.assets, args.control_engine, args.evidence_dir]):
        parser.error('--run-gates requires --assets, --control-engine and --evidence-dir')
    evidence = args.evidence_dir.resolve()
    if not evidence.is_relative_to('/workspace'):
        parser.error('validation output must be under /workspace')
    evidence.mkdir(parents=True, exist_ok=False)
    engine = json.loads((stage/'config/engine.json').read_text())
    python = engine['python']
    subprocess.run(verify + ['--static-only'], check=True)
    common = ['--assets', str(args.assets.resolve()), '--control-engine',
              str(args.control_engine.resolve()), '--stage-dir', str(stage)]
    for script, name in [('validate_behavior.py', 'correctness'),
                         ('benchmark_abba.py', 'performance')]:
        output = evidence/name
        with (evidence/(name+'.log')).open('w') as log:
            try:
                subprocess.run([python, str(root/'tests'/script), *common,
                                '--output-dir', str(output)], stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            finally:
                receipt = output/(name+'.json')
                if receipt.is_file():
                    shutil.copy2(receipt, stage/'manifest'/receipt.name)
        print('RUNTIME_GATE_PASSED', name, flush=True)
    identity = stage/'var/tmp/validation-build-identity.json'
    manifest = json.loads((stage/'manifest/runtime-manifest.json').read_text())
    identity.write_text(json.dumps(manifest['build'], indent=2)+'\n')
    create = [python, str(root/'tools/manifest/create_manifest.py'), '--root',
              str(root), '--stage-dir', str(stage), '--identity', str(identity)]
    for key in ['git_commit','git_branch']:
        if manifest.get(key):
            create += ['--'+key.replace('_','-'), manifest[key]]
    subprocess.run(create, check=True)
    subprocess.run(verify, check=True)
    # The verification receipt itself belongs to the final manifest.
    subprocess.run(create, check=True)
    subprocess.run(verify, check=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
