#!/usr/bin/env python3
import argparse,hashlib,io,json,pathlib,subprocess,sys,tarfile
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'build'))
from python_overlay import source_lock,verify_python_overlay
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=pathlib.Path,required=True);ap.add_argument('--stage-dir',type=pathlib.Path,required=True);ap.add_argument('--output',type=pathlib.Path,required=True);a=ap.parse_args()
    if a.output.exists():raise ValueError('refusing to overwrite an existing runtime package')
    manifest=json.loads((a.stage_dir/'manifest/runtime-manifest.json').read_text())
    engine=json.loads((a.stage_dir/'config/engine.json').read_text())
    if 'python_overlay' not in manifest:raise ValueError('missing Python overlay manifest record')
    overlay=verify_python_overlay(a.root,a.stage_dir,engine,manifest['python_overlay'])
    subprocess.run(['bash',str(a.root/'scripts/verify_runtime.sh'),'--stage-dir',str(a.stage_dir)],check=True)
    if a.output.resolve().is_relative_to(a.root.resolve()) or a.output.resolve().is_relative_to(a.stage_dir.resolve()):raise ValueError('package output must be outside source and stage')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(a.output,'w:gz',compresslevel=6) as tf:
        manifest=json.loads((a.stage_dir/'manifest/runtime-manifest.json').read_text())
        provenance=json.dumps({k:manifest[k] for k in ['git_commit','git_branch','source_tree_sha256']},indent=2).encode()+b'\n'
        item=tarfile.TarInfo('source/source-provenance.json');item.size=len(provenance);item.mtime=int(manifest['build']['source_date_epoch']);tf.addfile(item,io.BytesIO(provenance))
        for root,prefix in [(a.root,'source/production/dpa4c-runtime'),(a.stage_dir,'runtime')]:
            for p in sorted(root.rglob('*')):
                if not p.is_file() or '__pycache__' in p.parts or p.suffix in ['.pyc','.o','.ncu-rep','.qdrep','.nsys-rep']:continue
                if any(part in ['trace','tensors','cache','objects','var'] for part in p.relative_to(root).parts):continue
                tf.add(p,arcname=prefix+'/'+str(p.relative_to(root)),recursive=False)
    with tarfile.open(a.output,'r:gz') as tf:
        _,lock=source_lock(a.root)
        expected={'runtime/'+rel:digest for rel,digest in overlay['files'].items()}
        expected.update({'source/production/dpa4c-runtime/components/lammps-python/package/'+rel:digest for rel,digest in lock['files'].items()})
        for name,digest in expected.items():
            member=tf.getmember(name)
            if not member.isfile() or hashlib.sha256(tf.extractfile(member).read()).hexdigest()!=digest:raise ValueError('packaged Python member mismatch: '+name)
        if not (tf.getmember('runtime/lib/liblammps.so').isfile() or tf.getmember('runtime/lib/liblammps.so').issym()):raise ValueError('packaged LAMMPS load entry missing')
    digest=sha(a.output);a.output.with_suffix(a.output.suffix+'.sha256').write_text(digest+'  '+a.output.name+'\n')
    print(json.dumps({'path':str(a.output),'sha256':digest,'bytes':a.output.stat().st_size},indent=2))
if __name__=='__main__':main()
