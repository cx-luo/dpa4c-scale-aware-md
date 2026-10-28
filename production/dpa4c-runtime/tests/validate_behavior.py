#!/usr/bin/env python3
"""Bounded production equivalence and native restart gate; never nine-case scoring."""
import argparse,hashlib,json,pathlib,re,subprocess,sys,traceback
import numpy as np

LIMITS={'energy_per_atom_abs_eV':2e-5,'force_rmse_eV_A':4e-5,'force_max_abs_eV_A':5e-4,'stress_max_abs_GPa':4e-4,'position_max_abs_A':1e-4,'velocity_max_abs':1.2e-4,'integrator_state_max_abs':2.05e-5}
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def compare(left,right,atoms,load_frame):
    a,b=load_frame(left),load_frame(right);d={}
    for key in ['forces','stress_GPa','unwrapped_positions','velocities','integrator_state','cell','energy','total_energy','temperature','volume']:
        x=np.asarray(b[key],dtype=np.float64)-np.asarray(a[key],dtype=np.float64)
        d[key]={'rmse':float(np.sqrt(np.mean(x*x))) if x.size else 0.0,'max_abs':float(np.max(np.abs(x))) if x.size else 0.0}
    d['energy_per_atom_abs_eV']=d['energy']['max_abs']/atoms
    d['max_force_component_control_eV_A']=float(np.max(np.abs(a['forces'])))
    d['max_force_component_candidate_eV_A']=float(np.max(np.abs(b['forces'])))
    d['identity_equal']=all(np.array_equal(a[k],b[k]) for k in ['ids','types','masses','elements','step'])
    d['passed']=d['identity_equal'] and d['energy_per_atom_abs_eV']<=LIMITS['energy_per_atom_abs_eV'] and d['forces']['rmse']<=LIMITS['force_rmse_eV_A'] and d['forces']['max_abs']<=LIMITS['force_max_abs_eV_A'] and d['stress_GPa']['max_abs']<=LIMITS['stress_max_abs_GPa'] and d['unwrapped_positions']['max_abs']<=LIMITS['position_max_abs_A'] and d['velocities']['max_abs']<=LIMITS['velocity_max_abs'] and d['integrator_state']['max_abs']<=LIMITS['integrator_state_max_abs']
    return d
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--assets',type=pathlib.Path,required=True);ap.add_argument('--control-engine',type=pathlib.Path,required=True);ap.add_argument('--stage-dir',type=pathlib.Path,required=True);ap.add_argument('--output-dir',type=pathlib.Path,required=True);ap.add_argument('--cases',nargs='+',default=['Si-small','MgO-small','Si-medium','CuNi-large','MgO-large']);a=ap.parse_args()
    sys.path.insert(0,str(a.assets/'tools'))
    from metrics import load_frame
    from runner import MDRun
    from continuous_states import convert
    from checks import require_state
    if len(a.cases)>5:raise ValueError('this gate is deliberately bounded')
    a.output_dir.mkdir(parents=True,exist_ok=False)
    engine_paths={'control':a.control_engine,'candidate':a.stage_dir/'config/engine.json'}
    engines={k:json.loads(p.read_text()) for k,p in engine_paths.items()}
    cases={c['id']:c for c in json.loads((a.assets/'config/dev.json').read_text())['cases']}
    result={'scope':'bounded 1-step, uninterrupted 32-step, native 16+16 self-replay','full_nine_run':False,'official_submission':False,'limits':LIMITS,'engines':{k:sha(p) for k,p in engine_paths.items()},'cases':{}}
    lease=a.stage_dir/'.dpa4c-validation-active'
    with lease.open('x') as f:f.write('behavior verification active\n')
    def receipt(directory,run):
        maps=pathlib.Path(f'/proc/{run.process.pid}/maps').read_text()
        loaded=sorted(set(l.split()[-1] for l in maps.splitlines() if '.so' in l and l.split()[-1].startswith('/')))
        libraries=[]
        for p in loaded:
            dynamic=subprocess.check_output(['readelf','-d',p],text=True)
            soname=re.search(r'\(SONAME\).*\[(.*?)\]',dynamic)
            libraries.append({'path':p,'realpath':str(pathlib.Path(p).resolve()),'sha256':sha(p),'SONAME':soname.group(1) if soname else None})
        return {'loaded_libraries':libraries,'neighbor_builds':run.last_neighbor_builds,'step':run.step,'files':[{'path':p.name,'sha256':sha(p),'bytes':p.stat().st_size} for p in directory.iterdir() if p.is_file() and (p.suffix in ['.bin','.dump','.restart','.npz'] or p.name=='output-layout.json')]}
    try:
        for cid in a.cases:
            c=cases[cid];seed=a.assets/'data-full'/cid/'seed-00';data=seed/'equilibrated.data';restart=seed/'equilibrated.restart'
            row={'atoms':c['atoms'],'ensemble':c['ensemble'],'input_sha256':sha(data),'input_restart_sha256':sha(restart),'runs':{}};frames={};continuous={};checkpoint=None
            for name in ['control','candidate']:
                directory=a.output_dir/cid/(name+'-one')
                with MDRun(engines[name],c,directory,2400,data=data,restart=restart,outputs={'every':1,'restarts':True,'binary':True}) as md:
                    md.run(1);frames[name]=md.snapshot('step-one');require_state(load_frame(frames[name]),c);row['runs'][name+'-one']=receipt(directory,md)
                directory=a.output_dir/cid/(name+'-continuous')
                with MDRun(engines[name],c,directory,2400,data=data,restart=restart,outputs={'every':16,'restarts':True,'binary':True}) as md:
                    start=md.step;initial=md.snapshot('initial');wall=md.run(32)
                    native=convert(directory,c,load_frame(initial),start,start+32)
                    continuous[name]=native[-1];require_state(load_frame(continuous[name]),c)
                    if len(native)!=2:raise ValueError('expected native frames at steps 16 and 32')
                    checkpoint=directory/f'native-state.{start+16}.restart' if name=='candidate' else checkpoint
                    row['runs'][name+'-continuous']=dict(receipt(directory,md),wall_seconds=wall,native_frames=len(native),continuous_run_calls=1)
                text=(directory/'worker.log').read_text(errors='replace')
                log_text=(directory/'log.lammps').read_text(errors='replace')
                row['runs'][name+'-continuous']['path_markers']=[l for l in (text+chr(10)+log_text).splitlines() if any(k in l for k in ['active','captured','DIRECT_ENGINE','DIRECT_PRIME','VIRIAL_MODE','DPA_SMALL_NPT_COROUTINE','DPA_SMALL_EPOCH_PHASES','DPA_NPT_FINITE_EPOCH'])]
                if name=='candidate' and cid=='MgO-small':
                    row['small_npt_coroutine_route']={
                        'active':'DPA_SMALL_NPT_COROUTINE active=1' in log_text,
                        'fused':'DPA_SMALL_NPT_COROUTINE fused=1' in log_text,
                        'finite_epoch_active':(
                            'DPA_SMALL_EPOCH_PHASES active=1' in text
                            and 'DPA_NPT_FINITE_EPOCH eligible=1' in log_text
                        ),
                        'legacy_cross_step_active':'NPT_CROSS_STEP active=1' in log_text,
                    }
            directory=a.output_dir/cid/'candidate-replay'
            if not checkpoint.is_file():raise ValueError('native checkpoint missing')
            with MDRun(engines['candidate'],c,directory,2400,data=data,restart=checkpoint,outputs={'every':16,'restarts':True,'binary':True}) as md:
                start=md.step;initial=md.snapshot('initial');md.run(16);native=convert(directory,c,load_frame(initial),start,start+16);replayed=native[-1];require_state(load_frame(replayed),c);row['runs']['candidate-replay']=receipt(directory,md)
            row['one_step']=compare(frames['control'],frames['candidate'],c['atoms'],load_frame)
            row['continuous_32']=compare(continuous['control'],continuous['candidate'],c['atoms'],load_frame)
            row['native_restart_16_plus_16']=compare(continuous['candidate'],replayed,c['atoms'],load_frame)
            row['neighbor_counts_equal']=row['runs']['control-continuous']['neighbor_builds']==row['runs']['candidate-continuous']['neighbor_builds']
            route=row.get('small_npt_coroutine_route',{})
            route_ok = cid!='MgO-small' or (
                route.get('active')
                and (route.get('fused') or route.get('finite_epoch_active'))
                and not route.get('legacy_cross_step_active')
            )
            row['passed']=all(row[k]['passed'] for k in ['one_step','continuous_32','native_restart_16_plus_16']) and row['neighbor_counts_equal'] and route_ok
            result['cases'][cid]=row
            (a.output_dir/'correctness.json').write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps({'case':cid,'passed':row['passed'],'one_step':row['one_step'],'continuous_32':row['continuous_32'],'restart':row['native_restart_16_plus_16']},indent=2),flush=True)
            if not row['passed']:raise RuntimeError('behavior gate failed: '+cid)
        result['passed']=True
    except BaseException as e:
        result.update(passed=False,error=str(e),traceback=traceback.format_exc());raise
    finally:
        (a.output_dir/'correctness.json').write_text(json.dumps(result,indent=2)+'\n')
        lease.unlink()
if __name__=='__main__':main()
