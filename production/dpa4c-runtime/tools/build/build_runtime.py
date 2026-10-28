#!/usr/bin/env python3
"""Build every owned runtime library from locked source; never import objects."""
import argparse,concurrent.futures,hashlib,json,os,pathlib,shutil,subprocess,sys,tarfile,time
from python_overlay import source_lock,stage_python_overlay,verify_python_overlay

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def run(args,**kw):
    print('+',' '.join(map(str,args)),flush=True)
    subprocess.run(list(map(str,args)),check=True,**kw)
def workspace(p):
    p=pathlib.Path(p).resolve()
    if not p.is_relative_to('/workspace') or p==pathlib.Path('/workspace'):raise ValueError('build/stage must be descendants of /workspace')
    return p
def extract(archive,dest):
    dest.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive) as tf:
        prefix=tf.getmembers()[0].name.split('/')[0]
        for m in tf:
            parts=pathlib.PurePosixPath(m.name).parts
            if not parts or parts[0]!=prefix:raise ValueError('archive has inconsistent root')
            if len(parts)==1:continue
            m.name='/'.join(parts[1:])
            if m.isdev() or '..' in pathlib.PurePosixPath(m.name).parts:raise ValueError('unsafe archive member')
            target=dest/m.name
            if m.issym():
                link=(target.parent/m.linkname).resolve()
                if not link.is_relative_to(dest.resolve()):raise ValueError('unsafe source symlink')
            tf.extract(m,dest,filter='data')
def overlay(src,dst):
    for p in src.rglob('*'):
        if p.is_file():
            q=dst/p.relative_to(src);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=pathlib.Path,required=True)
    ap.add_argument('--build-dir',default='/workspace/dpa4c-production-build');ap.add_argument('--stage-dir',default='/workspace/dpa4c-production-stage')
    ap.add_argument('--config',default='Release');ap.add_argument('--jobs',type=int,default=8)
    for f in ['incremental','configure-only','stage-only']:ap.add_argument('--'+f,action='store_true')
    a=ap.parse_args();root=a.root.resolve();work=workspace(a.build_dir);stage=workspace(a.stage_dir)
    source_lock(root)
    if work==stage or stage.is_relative_to(work) or work.is_relative_to(stage):raise ValueError('build and stage must be disjoint')
    if a.config!='Release':raise ValueError('only Release validated')
    work.mkdir(parents=True,exist_ok=True)
    env_epoch='1767225600'
    os.environ['SOURCE_DATE_EPOCH']=env_epoch
    os.environ['TMPDIR']=str(work/'tmp');(work/'tmp').mkdir(exist_ok=True)
    python=pathlib.Path(os.environ.get('DPA_PYTHON_ENV','/opt/dpa-production-venv'))/'bin/python'
    cuda=pathlib.Path(os.environ.get('CUDA_HOME','/usr/local/PPU_SDK/CUDA_SDK'))
    sdk=pathlib.Path(os.environ.get('PPU_SDK','/usr/local/PPU_SDK'))
    torch=pathlib.Path(subprocess.check_output([python,'-c','import torch,pathlib; print(pathlib.Path(torch.__file__).parent)'],text=True).strip())
    nvcc=cuda/'bin/nvcc';env=os.environ.copy();env['PATH']=str(cuda/'bin')+':'+env['PATH']
    host_adapter=root/'tools/build/host_cxx.sh';host_adapter.chmod(0o755)
    env['NVCC_WRAPPER_DEFAULT_COMPILER']=str(host_adapter)
    targets={};source_ids={}
    lock=json.loads((root/'third_party/sources.lock.json').read_text())
    if not a.stage_only:
        for name in ['sources','deepmd-build','lammps-build','install','objects','output']:
            p=work/name
            if p.exists() and not a.incremental:shutil.rmtree(p)
        if not a.incremental:(work/'tmp/host-source-fixes.jsonl').unlink(missing_ok=True)
        for entry in lock['sources']:
            archive=root/entry['archive']
            if sha(archive)!=entry['archive_sha256']:raise ValueError('source archive SHA mismatch: '+str(archive))
            source_ids[entry['name']]=entry
            dest=work/'sources'/entry['name']
            if not dest.exists():extract(archive,dest)
        lmp=work/'sources/lammps';dp=work/'sources/deepmd/source'
        overlay(root/'components/lammps/overlays',lmp)
        overlay(root/'components/deepmd-op/overlays',dp)
        shutil.copy2(root/'include/dpa_chain/graph_compress_launch.h',dp/'op/pt/dpa4c/graph_compress_launch.h')
        wrapper=lmp/'lib/kokkos/bin/nvcc_wrapper';wrapper.chmod(0o755)
        install=work/'install';install.mkdir(exist_ok=True)
        run(['cmake','-S',lmp/'cmake','-B',work/'lammps-build','-G','Ninja',
             '-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CXX_FLAGS=-w',f'-DCMAKE_CXX_COMPILER={wrapper}',f'-DCMAKE_INSTALL_PREFIX={install}',f'-DCMAKE_PROJECT_INCLUDE={root}/cmake/ppu_compiler.cmake',
             '-DBUILD_SHARED_LIBS=ON','-DBUILD_MPI=OFF','-DBUILD_OMP=OFF','-DBUILD_TESTING=OFF',
             '-DPKG_KOKKOS=ON','-DPKG_PLUGIN=ON','-DKokkos_ENABLE_CUDA=ON','-DKokkos_ARCH_AMPERE80=ON',
             '-DLAMMPS_EXCEPTIONS=ON','-DCMAKE_INSTALL_RPATH=$ORIGIN'],env=env)
        run(['cmake','--build',work/'lammps-build','-j',a.jobs],env=env)
        run(['cmake','--install',work/'lammps-build'],env=env)
        run(['cmake','-S',dp,'-B',work/'deepmd-build','-G','Ninja','-DCMAKE_BUILD_TYPE=Release',
             '-DBUILD_PY_IF=OFF','-DBUILD_CPP_IF=ON','-DENABLE_TENSORFLOW=OFF','-DENABLE_PYTORCH=ON',
             '-DUSE_PT_PYTHON_LIBS=ON','-DUSE_TF_PYTHON_LIBS=OFF','-DUSE_CUDA_TOOLKIT=ON',
             f'-DCUDAToolkit_ROOT={cuda}','-DCMAKE_CUDA_ARCHITECTURES=80','-DDEEPMD_CUDA_PORTABLE_PTX=OFF',
             f'-DCMAKE_INSTALL_PREFIX={install}',f'-DCMAKE_INSTALL_RPATH=$ORIGIN:{torch}/lib:{cuda}/lib64',
             f'-DPython_EXECUTABLE={python}',f'-DCMAKE_CUDA_HOST_COMPILER={host_adapter}',f'-DCMAKE_PROJECT_INCLUDE={root}/cmake/ppu_compiler.cmake'],env=env)
        if a.configure_only:return
        run(['cmake','--build',work/'deepmd-build','-j',a.jobs],env=env)
        run(['cmake','--install',work/'deepmd-build'],env=env)
        output=work/'output';(output/'lib').mkdir(parents=True,exist_ok=True);(output/'plugin').mkdir(exist_ok=True)
        objects=work/'objects';objects.mkdir(exist_ok=True)
        common=['-O3','-std=c++17','-arch=sm_80','-ccbin',str(host_adapter),'-Xcompiler=-fPIC','--cudart=shared','-D_GLIBCXX_USE_CXX11_ABI=1']
        inc=['-I'+str(root/'include/dpa_chain')]+['-I'+str(p) for p in (root/'src').rglob('*') if p.is_dir()]
        torchinc=['-I'+str(torch/'include'),'-I'+str(torch/'include/torch/csrc/api/include')]
        units=['force/packed/force_runtime.cu','force/cartesian/force_runtime.cu','force/baseline/force_runtime.cu',
               'forward/forward_runtime.cu','forward/fallback/forward_runtime.cu','entry/chain_entry.cu',
               'integration/force_widen.cu','integration/reclaim_widen.cu','integration/widen_dispatch.cu',
               'fitting/fitting_runtime.cu','node/node_runtime.cu','fit_node/fit_node_runtime.cu','pipeline/model_pipeline.cu']
        chain_inputs=[(str(p.relative_to(root)),sha(p)) for tree in ['include','src'] for p in sorted((root/tree).rglob('*')) if p.is_file()]
        def compile_unit(rel):
            obj=objects/(rel.replace('/','_')+'.o')
            flags=[]
            if rel.startswith('fitting/'):
                flags=['-DFITTING_IMAGE_PRODUCTS=3','-DFITTING_IMAGE_KPERIOD=2','-DSPLIT_FITTING_LOAD64=1']
            if rel not in ['entry/chain_entry.cu','fitting/fitting_runtime.cu','pipeline/model_pipeline.cu']:flags+=['--use_fast_math']
            stamp=obj.with_suffix('.inputs.sha256')
            fingerprint=hashlib.sha256(json.dumps([chain_inputs,common,flags,inc,torchinc,sha(nvcc)],sort_keys=True).encode()).hexdigest()
            if a.incremental and obj.is_file() and stamp.is_file() and stamp.read_text()==fingerprint:
                print('REUSE_OWN_OBJECT',obj,flush=True);return obj
            run([nvcc,*common,*flags,*inc,*torchinc,'-c',root/'src'/rel,'-o',obj],env=env)
            stamp.write_text(fingerprint)
            return obj
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(a.jobs,4)) as pool:chainobj=list(pool.map(compile_unit,units))
        run([nvcc,*common,'-shared',*chainobj,'-L'+str(torch/'lib'),'-L'+str(install/'lib'),
             '-Xlinker','--soname=libdpa-chain.so','-Xlinker',f'--rpath={install}/lib:{torch}/lib:{cuda}/lib64',
             '-Xlinker','--version-script='+str(root/'cmake/dpa-chain.exports'),
             '-Xlinker','--no-as-needed','-ldeepmd_op_pt','-ltorch','-ltorch_cpu','-lc10','-Xlinker','--as-needed','-ldl',
             '-o',output/'lib/libdpa-chain.so'],env=env)
        chain_out=output/'lib/libdpa-chain.so'
        postlink_tool=shutil.which('patchelf')
        if not postlink_tool:raise RuntimeError('patchelf is required for deterministic chain normalization')
        chain_rpath='$ORIGIN:'+str(torch/'lib')+':'+str(cuda/'lib64')
        run([postlink_tool,'--set-soname','libdpa-chain.so','--set-rpath',chain_rpath,chain_out])
        patched_chain=output/'lib/libdpa-chain.patched.so'
        postlink_receipt=work/'lowreg-reverse-patch.json'
        run([python,root/'tools/postlink/patch_lowreg_reverse.py',chain_out,patched_chain,'--manifest',postlink_receipt])
        unit_chain=output/'lib/libdpa-chain.unit-seed.so'
        unit_receipt=work/'unit-seed-fixed-isa-patch.json'
        run([python,root/'tools/postlink/patch_unit_seed_fixed_isa.py',patched_chain,unit_chain,'--manifest',unit_receipt])
        os.replace(unit_chain,chain_out)
        patched_chain.unlink()
        kin=[lmp/'src',lmp/'src/KOKKOS',lmp/'src/PLUGIN',lmp/'src/KSPACE',lmp/'src/EXTRA-FIX',lmp/'src/STUBS',install/'include',install/'include/deepmd']
        kin += [lmp/'third_party',work/'lammps-build/styles',work/'lammps-build/lib/kokkos',
                lmp/'lib/kokkos/tpls/desul/include',lmp/'lib/kokkos/tpls/mdspan/include']
        for part in ['core','containers','algorithms','simd']:
            kin += [lmp/f'lib/kokkos/{part}/src',work/f'lammps-build/lib/kokkos/{part}/src']
        kflags=['-DKOKKOS_DEPENDENCE','-DLAMMPS_VERSION_NUMBER=20250722','-DLMPPLUGIN','-DLMP_KOKKOS','-Ddeepmd_lmp_EXPORTS',
                '-DLAMMPS_SMALLBIG','-DLMP_PLUGIN',
                '-fopenmp','-O3','-DNDEBUG','-fPIC','-std=c++17','-extended-lambda','-Wext-lambda-captures-this','-arch=sm_80','--cudart','shared',
                *['-I'+str(p) for p in kin]]
        psrc=root/'components/lammps-deepmd-plugin/src'
        generated=work/'plugin-generated';generated.mkdir(exist_ok=True)
        shutil.copy2(work/'deepmd-build/lmp/deepmd_version.h',generated/'deepmd_version.h')
        pluginsrc=sorted(psrc.glob('*.cpp'))+[lmp/'src/KSPACE'/n for n in ['pppm.cpp','fft3d.cpp','fft3d_wrap.cpp','remap.cpp','remap_wrap.cpp']]+[lmp/'src/EXTRA-FIX/fix_ttm.cpp']
        def plugin_compile(p):
            obj=objects/('plugin_'+p.name+'.o');run([wrapper,*kflags,'-I'+str(psrc),'-I'+str(generated),'-c',p,'-o',obj],env=env);return obj
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(a.jobs,4)) as pool:pobjs=list(pool.map(plugin_compile,pluginsrc))
        run([wrapper,'-fPIC','-fopenmp','-O3','-DNDEBUG','-rdynamic','-shared','--cudart','shared',*pobjs,'-L'+str(install/'lib'),
             '-Wl,--no-as-needed','-ldeepmd_backend_ptexpt','-ldeepmd_c','-Wl,--as-needed',
             f'-Wl,--rpath=$ORIGIN/../lib:{torch}/lib:{cuda}/lib64','-Wl,--soname=dpplugin.so','-o',output/'plugin/dpplugin.so'],env=env)
        for name,files in [('nvt-integration-plugin',['components/runtime-plugins/nvt/fix_nvt_fused_kokkos.cpp','components/runtime-plugins/nvt/nvt_plugin.cpp']),('npt-integration-plugin',['components/runtime-plugins/npt/npt_coroutine.cpp'])]:
            run([wrapper,*kflags,'-I'+str(root/'components/runtime-plugins/nvt'),'-shared',*[root/f for f in files],
                 f'-Wl,--rpath=$ORIGIN/../lib:{cuda}/lib64',f'-Wl,--soname={name}.so','-o',output/'plugin'/f'{name}.so','-ldl'],env=env)
        for rel,name,loc in [('src/runtime/device_observer.cu','libdeviceobserver','lib'),('src/runtime/device_restart.cu','libdevicerestart','lib'),('components/runtime-plugins/state/native_state.cu','native-state-plugin','plugin')]:
            run([nvcc,*common,'-shared','-DLAMMPS_SMALLBIG',*['-I'+str(p) for p in kin],root/rel,'-L'+str(install/'lib'),'-llammps',
                 '-Xlinker',f'--rpath={install}/lib:{cuda}/lib64','-Xlinker',f'--soname={name}.so','-pthread','-ldl','-o',output/loc/f'{name}.so'],env=env)
        run(['g++','-std=c++17','-O3','-fPIC','-shared','-pthread','-I'+str(cuda/'include'),root/'src/runtime/zerocopy_restart.cpp','-L'+str(cuda/'lib64'),'-Wl,--no-as-needed','-lcudart','-Wl,--as-needed','-ldl',f'-Wl,-rpath={cuda}/lib64','-Wl,-soname,libzerocopyrestart.so','-o',output/'lib/libzerocopyrestart.so'])
        run(['g++','-std=c++17','-O3','-fPIC','-shared','-I'+str(cuda/'include'),root/'src/runtime/fitting_grid.cpp','-ldl','-Wl,-soname,libgrid.so','-o',output/'lib/libgrid.so'])
        compiler_files={str(p):sha(p) for p in [nvcc,cuda/'bin/ppu_clang++',pathlib.Path('/usr/bin/g++-13'),cuda/'include/hggc_runtime.h'] if p.is_file()}
        fixes=work/'tmp/host-source-fixes.jsonl'
        transforms=[json.loads(line) for line in fixes.read_text().splitlines()] if fixes.is_file() else []
        if not transforms:raise RuntimeError('the pinned SDK host declaration transform did not execute')
        driver=subprocess.check_output([sdk/'ppu-smi/bin/ppu-smi','--query-ppu=name,uuid,driver_version','--format=csv,noheader'],text=True).strip()
        (work/'build-identity.json').write_text(json.dumps({'source_ids':source_ids,'compiler':subprocess.getoutput(str(nvcc)+' --version'),'cxx':subprocess.getoutput('g++ --version'),'compiler_files_sha256':compiler_files,'source_date_epoch':env_epoch,'compiler_source_transforms':transforms,'sdk':(sdk/'VERSION.txt').read_text(),'torch':subprocess.check_output([python,'-c','import torch;print(torch.__version__)'],text=True).strip(),'driver':driver,'build_dir':str(work),'config':a.config},indent=2)+'\n')
    # A stage contains only this build's products and the fixed model.
    marker=stage/'.dpa4c-stage'
    if stage.exists():
        if (stage/'.dpa4c-validation-active').exists():raise ValueError('refusing to replace a stage used by active validation')
        if any(stage.iterdir()) and not marker.is_file():raise ValueError('refusing to overwrite unmarked stage')
        shutil.rmtree(stage)
    for d in ['lib','plugin','config','models','manifest','bin','var/tmp']: (stage/d).mkdir(parents=True,exist_ok=True)
    marker.write_text('owned by dpa4c-runtime\n')
    for p in (work/'install/lib').glob('*.so*'):
        if p.name not in {'liblammps.so','liblammps.so.0','libdeepmd.so','libdeepmd_op_pt.so','libdeepmd_op_cuda.so','libdeepmd_dyn_cudart.so','libdeepmd_cc.so','libdeepmd_cc_low.so','libdeepmd_c.so','libdeepmd_backend_ptexpt.so'}:continue
        q=stage/'lib'/p.name
        if p.is_symlink():q.symlink_to(p.readlink())
        else:shutil.copy2(p,q)
    for loc in ['lib','plugin']:
        for p in (work/'output'/loc).glob('*.so'):shutil.copy2(p,stage/loc/p.name)
    stage_python_overlay(root,stage)
    shutil.copy2(work/'install/bin/lmp',stage/'bin/lmp')
    py=stage/'bin/python';py.write_text('#!/usr/bin/env bash\nexec '+str(python)+' "${@}"\n');py.chmod(0o755)
    model=pathlib.Path(os.environ.get('DPA_MODEL','/opt/dpa-production/models/compressed.pt2'))
    template=(root/'config/engine.production.json.in').read_text()
    substitutions={'STAGE':str(stage),'PYTHON':str(py),'TORCH_LIB':str(torch/'lib'),'CUDA_HOME':str(cuda),'PPU_SDK':str(sdk),'CUDA_LIB':str(cuda/'lib64'),'SDK_LIB':str(sdk/'lib')}
    for key,value in substitutions.items():template=template.replace('@'+key+'@',value)
    engine=json.loads(template);engine['environment'].update(json.loads((root/'config/runtime-defaults.json').read_text()))
    engine['environment']['TMPDIR']=str(stage/'var/tmp')
    verify_python_overlay(root,stage,engine)
    if sha(model)!=engine['model_sha256']:raise ValueError('fixed model mismatch')
    shutil.copy2(model,stage/'models/compressed.pt2')
    (stage/'config/engine.json').write_text(json.dumps(engine,indent=2)+'\n')
    sys.path.insert(0,str(root/'tools/manifest'))
    from stage_model import prepare_model
    prepare_model(stage,engine)
    (stage/'README.md').write_text('DPA4C runtime built entirely from production/dpa4c-runtime.\nUse config/engine.json; do not modify the live runtime.\n')
    # ELF paths never retain a disposable build directory.
    patchelf=shutil.which('patchelf')
    if not patchelf:raise RuntimeError('patchelf is required for final relocatable ELF layout')
    for p in sorted(stage.rglob('*.so*')):
        if p.is_symlink() or p.is_relative_to(stage/'models'):continue
        name=p.name;relative='$ORIGIN' if p.parent.name=='lib' else '$ORIGIN/../lib'
        run([patchelf,'--set-soname',name,'--set-rpath',relative+':'+str(torch/'lib')+':'+str(cuda/'lib64'),p])
        needed=subprocess.check_output([patchelf,'--print-needed',p],text=True).splitlines()
        for dep in needed:
            if dep.startswith('/'):run([patchelf,'--replace-needed',dep,pathlib.Path(dep).name,p])
    run([patchelf,'--set-rpath','$ORIGIN/../lib:'+str(cuda/'lib64'),stage/'bin/lmp'])
    shutil.copy2(work/'lowreg-reverse-patch.json',stage/'manifest/lowreg-reverse-patch.json')
    shutil.copy2(work/'unit-seed-fixed-isa-patch.json',stage/'manifest/unit-seed-fixed-isa-patch.json')
    run([python,root/'tools/manifest/create_manifest.py','--root',root,'--stage-dir',stage,'--identity',work/'build-identity.json'])
    print('BUILD_ALL_OK',stage,flush=True)
if __name__=='__main__':main()
