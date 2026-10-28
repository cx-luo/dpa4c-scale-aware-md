#!/usr/bin/env python3
import argparse,hashlib,json,os,pathlib,re,subprocess,sys
from stage_model import prepare_model
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'build'))
from python_overlay import verify_python_overlay
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def is_elf(p):
    with open(p,'rb') as f:return f.read(4)==b'\x7fELF'
def elf(p):
    if not is_elf(p):return {}
    out=subprocess.check_output(['readelf','-d',p],text=True)
    def field(key):
        m=re.search(r'\('+key+r'\).*\[(.*?)\]',out);return m.group(1) if m else None
    return {'SONAME':field('SONAME'),'RUNPATH':field('RUNPATH'),'RPATH':field('RPATH')}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=pathlib.Path,required=True);ap.add_argument('--stage-dir',type=pathlib.Path,required=True);ap.add_argument('--identity',type=pathlib.Path,required=True);ap.add_argument('--git-commit');ap.add_argument('--git-branch');a=ap.parse_args();root=a.root.resolve();stage=a.stage_dir.resolve()
    engine=json.loads((stage/'config/engine.json').read_text());python_overlay=verify_python_overlay(root,stage,engine);model_members=prepare_model(stage,engine);env=dict(os.environ,**engine['environment']);env.pop('LD_PRELOAD',None)
    products=[];external={};errors=[]
    identity=json.loads(a.identity.read_text());identity['build_dir']=None
    identity['driver_kernel_module']={'sha256':None,'provided_by_runtime_host':True,'reason':'kernel module binary is not exposed in the audited PPU container; driver version and user-space library hashes are recorded'}
    def external_row(path,observed_sha=None):
        q=pathlib.Path(path)
        owner=subprocess.run(['dpkg-query','-S',str(q.resolve())],capture_output=True,text=True).stdout.strip().split(': ',1)[0]
        version=subprocess.run(['dpkg-query','-W','-f=${Version}',owner],capture_output=True,text=True).stdout.strip() if owner else None
        origin='system runtime image'
        if 'PPU_SDK' in path:
            origin='PPU SDK';version=identity['sdk'].strip()
        elif 'torch' in path:
            origin='PyTorch distribution';version=identity['torch']
        elif '/opt/' in path:
            origin='Python/runtime image distribution'
        return {'path':path,'realpath':str(q.resolve()),'sha256':observed_sha or sha(q),'bytes':q.stat().st_size,**elf(q),'version':version or 'not embedded; pinned by SHA256','origin':origin,'provided_by_runtime_image':True}
    for p in sorted(stage.rglob('*')):
        if not p.is_file() or 'var' in p.relative_to(stage).parts or '__pycache__' in p.parts or p.suffix=='.pyc' or p.name in ['runtime-manifest.json','SHA256SUMS'] or p.name.startswith('.'):continue
        rel=str(p.relative_to(stage));row={'path':rel,'realpath':str(p.resolve()),'sha256':sha(p),'bytes':p.stat().st_size,'source':'unified-source-build','build_target':p.name,**elf(p)}
        if p.is_symlink():row['symlink_target']=str(p.readlink())
        if rel.startswith('models/'):row.update(source='fixed-immutable-model',build_target=None)
        if rel in model_members:row.update(model_members[rel],source='fixed-immutable-model')
        elif rel.startswith('python/'):row.update(source='owned-python-package',build_target='lammps-python')
        elif p.name=='python':row.update(source='generated-python-launcher',build_target='runtime-config')
        elif p.suffix not in ['.so','.0'] and '.so.' not in p.name and p.name!='lmp':row.update(source='generated-runtime-config',build_target='runtime-config')
        if is_elf(p):
            output=subprocess.run(['ldd',p],env=env,text=True,capture_output=True).stdout
            row['ldd']=output
            if 'not found' in output:errors.append(rel+': unresolved dependency')
            for path in re.findall(r'(?:=>\s*)?(/[^\s()]+)',output):
                q=pathlib.Path(path)
                if not q.is_file() or q.resolve().is_relative_to(stage):continue
                if path not in external:external[path]=external_row(path)
        products.append(row)
    for path,origin in [(engine['environment']['CUDA_HOME']+'/bin/nvcc','PPU compiler'),('/opt/dpa-production-venv/bin/python','Python virtual environment')]:
        q=pathlib.Path(path)
        if q.is_file():external[path]={'path':path,'realpath':str(q.resolve()),'sha256':sha(q),'bytes':q.stat().st_size,**elf(q),'version':subprocess.getoutput(path+' --version'),'origin':origin,'provided_by_runtime_image':True}
    sourcefiles=[]
    for p in sorted(root.rglob('*')):
        if p.is_file() and p.name not in ['runtime-manifest.json'] and '__pycache__' not in p.parts and p.suffix!='.pyc':sourcefiles.append({'path':str(p.relative_to(root)),'sha256':sha(p)})
    source_id=hashlib.sha256(json.dumps(sourcefiles,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    def git(args):
        r=subprocess.run(['git','-C',root,*args],capture_output=True,text=True);return r.stdout.strip() if r.returncode==0 else None
    validation={}
    for name in ['static-verification','correctness','performance']:
        p=stage/'manifest'/f'{name}.json'
        if p.is_file():validation[name]={'sha256':sha(p),'passed':json.loads(p.read_text()).get('passed',False)}
    observed={};validation_only={}
    evidence=stage/'manifest/correctness.json'
    if evidence.is_file():
        for case in json.loads(evidence.read_text()).get('cases',{}).values():
            for name,receipt in case['runs'].items():
                if not name.startswith('candidate'):continue
                for row in receipt['loaded_libraries']:
                    path=row['path'];q=pathlib.Path(path)
                    if q.name=='force_observer.so':validation_only[path]=dict(row,source='organizer read-only validation observer');continue
                    observed[path]=row
                    if not q.resolve().is_relative_to(stage) and q.is_file() and path not in external:external[path]=external_row(path,row['sha256'])
    provenance_path=root.parent.parent/'source-provenance.json'
    provenance=json.loads(provenance_path.read_text()) if provenance_path.is_file() else {}
    commit=a.git_commit or git(['rev-parse','HEAD']) or provenance.get('git_commit')
    branch=a.git_branch or git(['branch','--show-current']) or provenance.get('git_branch')
    manifest={'schema_version':1,'status':'validated-independent-stage' if len(validation)==3 and all(v['passed'] for v in validation.values()) else 'build-complete-validation-pending','git_commit':commit,'git_branch':branch,'source_tree_sha256':source_id,'source_files':sourcefiles,'build':identity,'engine_path':str(stage/'config/engine.json'),'engine_sha256':sha(stage/'config/engine.json'),'model_sha256':engine['model_sha256'],'files':products,'external_dependencies':list(external.values()),'actual_worker_loaded_libraries':list(observed.values()),'validation_only_libraries':list(validation_only.values()),'errors':errors,'validation':validation,'postlink':{'required':True,'implementation':'two fail-closed fixed-ISA transforms: reverse traversal/dead gradient tail, then registered unit-seed load elimination','receipts':[{'path':'manifest/lowreg-reverse-patch.json','sha256':sha(stage/'manifest/lowreg-reverse-patch.json')},{'path':'manifest/unit-seed-fixed-isa-patch.json','sha256':sha(stage/'manifest/unit-seed-fixed-isa-patch.json')}]},'not_official_evaluation':True}
    manifest['python_overlay']=python_overlay
    dest=stage/'manifest/runtime-manifest.json';dest.write_text(json.dumps(manifest,indent=2)+'\n')
    (stage/'manifest/SHA256SUMS').write_text(''.join(r['sha256']+'  '+r['path']+'\n' for r in products)+sha(dest)+'  manifest/runtime-manifest.json\n')
    print('MANIFEST_CREATED',dest,'products',len(products),'external',len(external),'errors',errors)
    if errors:sys.exit(1)
if __name__=='__main__':main()
