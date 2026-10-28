#!/usr/bin/env python3
import argparse,hashlib,json,os,pathlib,re,subprocess,sys
from create_manifest import sha,is_elf
from python_overlay import source_lock,verify_python_overlay
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=pathlib.Path,required=True);ap.add_argument('--stage-dir',type=pathlib.Path,required=True);ap.add_argument('--source-only',action='store_true');ap.add_argument('--static-only',action='store_true');a=ap.parse_args();root=a.root.resolve();stage=a.stage_dir.resolve();errors=[];checks={}
    historical_paths=['/'.join(['experiments','']),'/'.join(['candidate','variants',''])]
    forbidden=re.compile(r'R[0-9]+|round[0-9]+|patch_r[0-9]+|libr[0-9]+|'+('|'.join(map(re.escape,historical_paths))))
    for p in root.rglob('*'):
        if not p.is_file() or p.relative_to(root)==pathlib.Path('docs/MIGRATION.md') or '__pycache__' in p.parts or 'archives' in p.parts:continue
        if p.suffix in ['.pyc','.tar','.gz']:continue
        try:text=p.read_text()
        except UnicodeError:continue
        scan_text=text
        if p.relative_to(root)==pathlib.Path('components/lammps-python/package/lammps/formats.py'):
            # Upstream linter identifiers are not historical runtime dependencies.
            scan_text=re.sub(r'(?m)^ *# pylint: disable=[A-Z]\d+(?:,[A-Z]\d+)*\s*$','',text)
        if forbidden.search(scan_text) or forbidden.search(str(p.relative_to(root))):errors.append('source naming/dependency violation: '+str(p.relative_to(root)))
        if re.search(r'\bmprotect\s*\('+'|'+'PROT_'+'EXEC',text):errors.append('runtime binary mutation: '+str(p.relative_to(root)))
    try:source_lock(root)
    except (ValueError,OSError,KeyError) as exc:errors.append('Python source lock: '+str(exc))
    checks['source_clean']=not errors
    if a.source_only:
        print(json.dumps({'checks':checks,'errors':errors},indent=2));return sys.exit(bool(errors))
    manifest=json.loads((stage/'manifest/runtime-manifest.json').read_text());engine=json.loads((stage/'config/engine.json').read_text());env=dict(os.environ,**engine['environment']);env.pop('LD_PRELOAD',None)
    try:
        if 'python_overlay' not in manifest:raise ValueError('missing Python overlay manifest record')
        overlay=verify_python_overlay(root,stage,engine,manifest['python_overlay'])
        import_env=dict(env,PYTHONDONTWRITEBYTECODE='1')
        actual=subprocess.check_output([engine['python'],'-c','import json,lammps.core,pathlib; print(json.dumps({"core":str(pathlib.Path(lammps.core.__file__).resolve())}))'],env=import_env,text=True)
        if json.loads(actual)['core']!=str(stage/'python/lammps/core.py'):raise ValueError('actual import bypassed staged Python core')
        checks['python_overlay']=True
    except (ValueError,OSError,KeyError,subprocess.CalledProcessError) as exc:
        errors.append('Python overlay: '+str(exc));checks['python_overlay']=False;overlay={'files':{}}
    required_files=['bin/lmp','bin/python','models/compressed.pt2','config/engine.json',
        'lib/liblammps.so','lib/liblammps.so.0','lib/libdeepmd.so','lib/libdeepmd_c.so','lib/libdeepmd_cc.so',
        'lib/libdeepmd_backend_ptexpt.so','lib/libdeepmd_op_pt.so','lib/libdeepmd_op_cuda.so',
        'lib/libdeepmd_dyn_cudart.so','lib/libdpa-chain.so','lib/libdeviceobserver.so',
        'lib/libdevicerestart.so','lib/libzerocopyrestart.so','lib/libgrid.so',
        'plugin/dpplugin.so','plugin/native-state-plugin.so','plugin/nvt-integration-plugin.so',
        'plugin/npt-integration-plugin.so']
    required_files+=list(overlay['files'])
    recorded={row['path'] for row in manifest['files']}
    for rel in required_files:
        if rel not in recorded or not (stage/rel).is_file():errors.append('required runtime file missing '+rel)
    if sha(stage/'config/engine.json')!=manifest['engine_sha256']:errors.append('engine hash mismatch')
    if sha(engine['model_file'])!=engine['model_sha256'] or engine['model_sha256']!=manifest['model_sha256']:errors.append('model hash mismatch')
    for row in manifest['files']:
        p=stage/row['path']
        if not p.is_file():errors.append('missing '+row['path']);continue
        if sha(p)!=row['sha256']:errors.append('SHA mismatch '+row['path'])
        if str(p.resolve())!=row['realpath']:errors.append('symlink real target mismatch '+row['path'])
        if not is_elf(p):continue
        dynamic=subprocess.check_output(['readelf','-d',p],text=True)
        if forbidden.search(dynamic) or '/sources/' in dynamic or '/install/' in dynamic or '/objects/' in dynamic:errors.append('ELF path retains source/build dependency '+row['path'])
        ldd=subprocess.run(['ldd',p],env=env,capture_output=True,text=True).stdout
        if 'not found' in ldd:errors.append('ldd missing '+row['path'])
        if '.so' in p.name and '(SONAME)' not in dynamic and row['source']!='fixed-immutable-model':errors.append('SONAME missing '+row['path'])
        for path in re.findall(r'(?:=>\s*)?(/[^\s()]+)',ldd):
            q=pathlib.Path(path)
            if q.is_file() and ('deepmd' in q.name or 'lammps' in q.name or 'libdpa' in q.name) and not q.resolve().is_relative_to(stage):errors.append('owned dependency outside stage '+path)
    for row in manifest['external_dependencies']:
        if not pathlib.Path(row['path']).is_file() or sha(row['path'])!=row['sha256']:errors.append('external hash mismatch '+row['path'])
    for row in manifest['source_files']:
        p=root/row['path']
        if not p.is_file() or sha(p)!=row['sha256']:errors.append('source hash mismatch '+row['path'])
    for path,expected in manifest['build'].get('compiler_files_sha256',{}).items():
        if not pathlib.Path(path).is_file() or sha(path)!=expected:errors.append('compiler hash mismatch '+path)
    if not manifest['build'].get('compiler_source_transforms'):errors.append('missing SDK compiler transform evidence')
    for key in ['LD_PRELOAD','LD_LIBRARY_PATH','LAMMPS_PLUGIN_PATH']:
        for path in engine['environment'][key].split(':'):
            if not pathlib.Path(path).exists():errors.append('engine path missing '+path)
            if forbidden.search(path):errors.append('engine historical path '+path)
            if key in ['LD_PRELOAD','LAMMPS_PLUGIN_PATH'] and not pathlib.Path(path).resolve().is_relative_to(stage):errors.append('owned engine path outside stage '+path)
    for key in ['model_file','python']:
        p=pathlib.Path(engine[key])
        if not p.is_file() or not p.is_relative_to(stage):errors.append('engine file not staged '+key)
    for key in ['DPA_AOTI_MODEL_LIBRARY','DPA_AOTI_MODEL_DIRECTORY']:
        p=pathlib.Path(engine['environment'][key])
        if not p.exists() or not p.resolve().is_relative_to(stage/'models'):errors.append('engine model package path invalid '+key)
    if any(forbidden.search(k) for k in engine['environment']):errors.append('historical runtime switch')
    symbols={}
    for p in sorted(stage.rglob('*.so*')):
        if p.is_symlink():continue
        output=subprocess.check_output(['nm','-D','--defined-only',p],text=True)
        for line in output.splitlines():
            fields=line.split()
            if len(fields)==3 and fields[1] in ['T','D','B']:symbols.setdefault(fields[2],[]).append(str(p.relative_to(stage)))
    required=['_ZN12deepmd_dpa4c19launch_raw_force_c8ERKNS_9ArgumentsERK21DpaDirectNeighborViewP11CUstream_stPfS8_','_ZN12deepmd_dpa4c21launch_raw_forward_c8ERKNS_9ArgumentsERK21DpaDirectNeighborViewP11CUstream_st','_ZN12deepmd_dpa4c21launch_direct_node_c8ERKNS_9ArgumentsEP11CUstream_st','dpa_chain_set_need_virial_stream','dpa_chain_requested_virial_device_pointer','dpa_chain_try_fitting_range','dpa_chain_try_widen','dpa_chain_small_nvt_materialize_force','dpa_direct_energy_force','dpa_direct_destination_only_ready','dpa_get_direct_neighbor_view','dpa_device_restart_pack_batch']
    if engine['environment'].get('DPA_SHARED_OUTPUT_SNAPSHOT','0')!='0':
        required += ['dpa_zerocopy_restart_shared_snapshot_available',
                     'dpa_device_restart_register_state_snapshot',
                     'dpa_device_restart_release_state_snapshot']
    for sym in required:
        if len(symbols.get(sym,[]))!=1:errors.append('ABI missing/duplicate '+sym+': '+str(symbols.get(sym)))
    duplicates={k:v for k,v in symbols.items() if len(v)>1}
    interposed={
        'cudaLaunchKernel':{'lib/libdeepmd_dyn_cudart.so','lib/libgrid.so'},
        '_ZN9LAMMPS_NS12ModifyKokkos10post_forceEi':{'lib/libdeviceobserver.so','lib/liblammps.so.0'},
        '_ZN9LAMMPS_NS3Fix13modify_paramsEiPPc':{'lib/libdeviceobserver.so','lib/liblammps.so.0'},
        '_ZN9LAMMPS_NS7AtomVec12pack_restartEiPd':{'lib/libdevicerestart.so','lib/liblammps.so.0'},
        '_ZN9LAMMPS_NS10AtomKokkos4syncENS_14ExecutionSpaceEj':{'lib/liblammps.so.0','plugin/native-state-plugin.so'},
        '_ZN9LAMMPS_NS12WriteRestart5writeERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE':{'lib/liblammps.so.0','lib/libzerocopyrestart.so'},
        '_ZN9LAMMPS_NS6Memory5sfreeEPv':{'lib/liblammps.so.0','lib/libzerocopyrestart.so'},
        '_ZN9LAMMPS_NS6Memory7smallocElPKc':{'lib/liblammps.so.0','lib/libzerocopyrestart.so'},
        'lammpsplugin_init':{'plugin/dpplugin.so','plugin/native-state-plugin.so','plugin/npt-integration-plugin.so','plugin/nvt-integration-plugin.so'}}
    unexpected={k:v for k,v in duplicates.items() if set(v)!=interposed.get(k,set())}
    if unexpected:errors.append('unexpected strong duplicate exports '+str(unexpected))
    if not a.static_only:
        for name in ['correctness','performance']:
            p=stage/'manifest'/f'{name}.json'
            if not p.is_file():errors.append('missing '+name+' validation evidence');continue
            data=json.loads(p.read_text())
            if not data.get('passed'):errors.append(name+' evidence did not pass')
            if set(data.get('cases',{}))!={'Si-small','MgO-small','Si-medium','CuNi-large','MgO-large'}:errors.append(name+' validation case coverage incomplete')
            if name=='performance' and any([r['label'] for r in case['runs']]!=['A','B','B','A'] for case in data.get('cases',{}).values()):errors.append('performance validation order mismatch')
            hashes=data.get('engines',data.get('engine_sha256',{}))
            if hashes.get('candidate',hashes.get('B'))!=manifest['engine_sha256']:errors.append(name+' candidate engine identity mismatch')
            if name=='performance':
                by_path={str((stage/r['path']).resolve()):r['sha256'] for r in manifest['files']}
                validation_only={row['realpath']:row['sha256'] for row in manifest.get('validation_only_libraries',[])}
                for case in data.get('cases',{}).values():
                    for receipt in case['runs']:
                        if receipt['label']!='B':continue
                        for row in receipt['loaded_project_libraries']:
                            p=pathlib.Path(row['path']);real=str(p.resolve())
                            if validation_only.get(real)==row['sha256']:continue
                            if real not in by_path or row['sha256']!=by_path[real]:errors.append('candidate loaded performance library identity mismatch '+str(p))
            if name=='correctness':
                by_path={str((stage/r['path']).resolve()):r['sha256'] for r in manifest['files']}
                for case in data.get('cases',{}).values():
                    for label,receipt in case['runs'].items():
                        if not label.startswith('candidate'):continue
                        for row in receipt['loaded_libraries']:
                            p=pathlib.Path(row['path']);real=str(p.resolve())
                            if real in by_path and row['sha256']!=by_path[real]:errors.append('candidate loaded library identity mismatch '+str(p))
                            if ('libdeepmd' in p.name or 'liblammps' in p.name or 'libdpa-chain' in p.name) and not p.resolve().is_relative_to(stage):errors.append('candidate loaded owned library outside stage '+str(p))
                            if p.name.endswith('.wrapper.so') and not p.resolve().is_relative_to(stage/'models'):errors.append('candidate loaded model outside stage '+str(p))
        checks['saved_behavior_and_performance']=not any('evidence' in e or 'validation' in e or 'candidate engine identity' in e for e in errors)
    for name,pattern in [('manifest_hashes','SHA mismatch|engine hash mismatch|model hash mismatch|real target mismatch'),('external_hashes','external hash mismatch'),('source_hashes','source hash mismatch'),('ldd','ldd missing|owned dependency'),('elf_paths','ELF path|SONAME|engine path'),('abi','ABI|duplicate exports'),('actual_worker_library_identity','candidate loaded')]:checks[name]=not any(re.search(pattern,e) for e in errors)
    result={'passed':not errors,'checks':checks,'required_symbols':{s:symbols.get(s,[]) for s in required},'intentional_duplicate_exports':{k:v for k,v in duplicates.items() if k not in unexpected},'errors':errors}
    (stage/'manifest/static-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    sys.exit(bool(errors))
if __name__=='__main__':main()
