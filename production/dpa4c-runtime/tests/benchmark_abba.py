#!/usr/bin/env python3
import argparse,hashlib,json,pathlib,statistics,sys
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--assets',type=pathlib.Path,required=True);ap.add_argument('--control-engine',type=pathlib.Path,required=True);ap.add_argument('--stage-dir',type=pathlib.Path,required=True);ap.add_argument('--output-dir',type=pathlib.Path,required=True);ap.add_argument('--cases',nargs='+',default=['Si-small','MgO-small','Si-medium','CuNi-large','MgO-large']);a=ap.parse_args()
    if len(a.cases)>5:raise ValueError('no full nine-case benchmark')
    sys.path.insert(0,str(a.assets/'tools'));from runner import MDRun
    a.output_dir.mkdir(parents=True,exist_ok=False)
    paths={'A':a.control_engine,'B':a.stage_dir/'config/engine.json'};engines={k:json.loads(p.read_text()) for k,p in paths.items()}
    config={c['id']:c for c in json.loads((a.assets/'config/dev.json').read_text())['cases']}
    result={'scope':'finite A-B-B-A engineering regression screen','not_a_nine_case_score':True,'official_submission':False,'obvious_regression_threshold_percent':5.0,'engine_sha256':{k:sha(p) for k,p in paths.items()},'cases':{}}
    for cid in a.cases:
        c=config[cid];seed=a.assets/'data-full'/cid/'seed-00'
        warm,steps={'small':(512,8192),'medium':(128,1024),'large':(64,256)}[c['scale']]
        runs=[]
        for i,label in enumerate(['A','B','B','A']):
            directory=a.output_dir/cid/f'{i}-{label}'
            with MDRun(engines[label],c,directory,2400,data=seed/'equilibrated.data',restart=seed/'equilibrated.restart',outputs={'every':c['dump_every'],'restarts':False,'binary':True}) as md:
                md.run(warm);before=md.last_neighbor_builds;seconds=md.run(steps)
                maps=pathlib.Path(f'/proc/{md.process.pid}/maps').read_text()
                loaded_paths=sorted(set(l.split()[-1] for l in maps.splitlines() if '.so' in l and l.split()[-1].startswith('/workspace/') and '/test-assets/' not in l.split()[-1] and '/var/' not in l.split()[-1]))
                runs.append({'order':i,'label':label,'warmup_steps':warm,'measure_steps':steps,'wall_seconds':seconds,'ms_per_step':seconds*1000/steps,'neighbor_builds':md.last_neighbor_builds,'neighbor_builds_after_warmup':before,'loaded_project_libraries':[{'path':p,'sha256':sha(p)} for p in loaded_paths]})
            print(json.dumps({'case':cid,**runs[-1]}),flush=True)
        med={k:statistics.median(x['ms_per_step'] for x in runs if x['label']==k) for k in ['A','B']}
        change=(med['B']/med['A']-1)*100
        result['cases'][cid]={'runs':runs,'A_median_ms_per_step':med['A'],'B_median_ms_per_step':med['B'],'time_change_percent':change,'throughput_change_percent':(med['A']/med['B']-1)*100,'neighbor_counts_equal':len({r['neighbor_builds'] for r in runs})==1,'no_obvious_regression':change<=5.0}
        (a.output_dir/'performance.json').write_text(json.dumps(result,indent=2)+'\n')
    result['passed']=all(r['no_obvious_regression'] and r['neighbor_counts_equal'] for r in result['cases'].values())
    (a.output_dir/'performance.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    if not result['passed']:raise SystemExit('bounded performance screen failed')
if __name__=='__main__':main()
