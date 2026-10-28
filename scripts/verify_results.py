"""Recompute the frozen nine-case result from all 27 retained timing pairs.

Uses Python's standard library only. Does not run molecular dynamics.
"""
from pathlib import Path
import csv,json,math,statistics
P=Path(__file__).resolve().parents[1]
def rows(name):
    with (P/'data'/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
pairs=rows('nine-v1-repeats.csv');reported=rows('nine-results.csv')
assert len(pairs)==27 and len(reported)==9
scores={};detail=[]
for case in reported:
    legs=[p for p in pairs if p['case']==case['case']]
    assert len(legs)==3 and len({p['repeat'] for p in legs})==3
    assert all(p['correctness_passed'].lower()=='true' for p in legs)
    ratios=[float(p['baseline_seconds'])/float(p['candidate_seconds']) for p in legs]
    value=statistics.median(ratios);scores[case['case']]=value
    assert math.isclose(value,float(case['speedup']),rel_tol=1e-12)
    for key,col in [('baseline_seconds','baseline_s'),('candidate_seconds','candidate_s')]:
        assert math.isclose(statistics.median(float(p[key]) for p in legs),float(case[col]),rel_tol=1e-12)
    assert math.isclose(100*(value-1),float(case['throughput_pct']),rel_tol=1e-12)
    assert math.isclose(100*(1-1/value),float(case['time_reduction_pct']),rel_tol=1e-12)
    detail.append({'case':case['case'],'ratios':ratios,'median':value,'range':[min(ratios),max(ratios)]})
score=math.exp(statistics.mean(math.log(s) for s in scores.values()))
frozen=json.loads((P/'data/frozen-version.json').read_text())
assert math.isclose(score,frozen['score'],rel_tol=1e-14)
groups={scale:math.exp(statistics.mean(math.log(v) for k,v in scores.items() if k.endswith(scale))) for scale in ['small','medium','large']}
for row in rows('group-results.csv'):assert math.isclose(groups[row['scale']],float(row['geometric_speedup']),rel_tol=1e-12)
gates=rows('nine-gates.csv');assert len(gates)==36 and all(g['passed'].lower()=='true' for g in gates)
assert frozen['verified'] is False and frozen['audit']['status']=='not_run'
print(json.dumps({'status':'PASS','pairs':27,'cases':detail,'group_geomeans':groups,'score':score,'physical_stage_pass_flags':36,'independent_timed_interval_audit':'not_run'},indent=2))
