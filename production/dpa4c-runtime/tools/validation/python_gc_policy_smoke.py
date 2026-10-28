#!/usr/bin/env python3
"""Real-library GC ownership/lifecycle regressions, zero MD steps.

Scope: isolated process with serial GC policy control and no active-policy fork.
The RLock coordinates owners; it does not serialize arbitrary external setters.
"""
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import weakref
from lammps import lammps
import lammps.core as core

ENV = 'LAMMPS_PYTHON_GC_THRESHOLD'
BASE = gc.get_threshold()
ENABLED = gc.isenabled()
ORIGINAL_ENV = os.environ.get(ENV)
PLAN = {'primary_threshold': 524288, 'only_allowed_fallback_threshold': 1048576}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def make(value=None):
    os.environ[ENV] = str(PLAN['primary_threshold'] if value is None else value)
    return lammps(cmdargs=['-log', 'none', '-screen', 'none', '-nocite'])

def check_restored():
    assert gc.get_threshold() == BASE and gc.isenabled() == ENABLED
    assert core._LAMMPS_GC_STATE is None

def lifecycle():
    checks = []
    obj = make(0); assert gc.get_threshold() == BASE; obj.close();check_restored();checks.append('opt_out')
    obj = make();assert gc.get_threshold() == (PLAN['primary_threshold'], BASE[1], BASE[2]);assert gc.isenabled()==ENABLED
    obj.close();obj.close();check_restored();checks.append('single_and_repeated_close')
    a,b=make(),make();assert core._LAMMPS_GC_STATE['owners']==2
    a.close();a.close();assert core._LAMMPS_GC_STATE['owners']==1
    b.close();check_restored();checks.append('two_instances_interleaved')
    try:
        with make():raise LookupError('context sentinel')
    except LookupError:pass
    check_restored();checks.append('exception_context_manager')
    gc.disable();obj=make();assert not gc.isenabled();obj.close();assert not gc.isenabled() and gc.get_threshold()==BASE
    gc.enable() if ENABLED else gc.disable();check_restored();checks.append('initially_disabled')
    gc.set_threshold(0,BASE[1],BASE[2]);obj=make();assert gc.get_threshold()[0]==0 and not obj._gc_policy_active
    obj.close();assert gc.get_threshold()[0]==0;gc.set_threshold(*BASE);check_restored();checks.append('initially_zero_threshold')
    for value in ['invalid', '-1']:
        try:make(value)
        except ValueError:pass
        else:raise AssertionError('invalid policy accepted')
        gc.collect(2);check_restored()
    checks.append('failed_policy_constructor')
    os.environ[ENV]=str(PLAN['primary_threshold'])
    try:lammps(name='gc-policy-intentionally-missing-library',cmdargs=['-log','none','-screen','none'])
    except OSError:pass
    else:raise AssertionError('missing library unexpectedly loaded')
    gc.collect(2);check_restored();checks.append('failed_native_initialization')
    obj=make();original_lib=obj.lib
    class CloseErrorProxy:
        def lammps_close(self,pointer):raise RuntimeError('synthetic native-close failure')
    obj.lib=CloseErrorProxy()
    try:obj.close()
    except RuntimeError:pass
    else:raise AssertionError('close did not raise')
    check_restored();assert obj.lmp and obj.opened
    obj.lib=original_lib;obj.close();check_restored();checks.append('native_close_exception_finally')
    obj=make();external=(BASE[0]+37,BASE[1]+1,BASE[2]);gc.set_threshold(*external)
    assert gc.get_threshold()==external, 'external test policy must be observable on this interpreter'
    obj.close();assert gc.get_threshold()==external and core._LAMMPS_GC_STATE is None
    gc.set_threshold(*BASE);check_restored();checks.append('external_threshold_wins')
    obj=make();gc.disable();obj.close();assert not gc.isenabled() and gc.get_threshold()==BASE
    gc.enable() if ENABLED else gc.disable();check_restored();checks.append('external_enabled_change_wins')
    obj=make();ref=weakref.ref(obj);del obj;gc.collect(2)
    assert ref() is None;check_restored();checks.append('destructor_no_new_cycle')
    obj=make();wrapper=obj.numpy;ref=weakref.ref(obj);del obj,wrapper;gc.collect(2)
    assert ref() is None;check_restored();checks.append('existing_numpy_cycle_collected')
    a=make()
    try:make(PLAN['only_allowed_fallback_threshold'])
    except ValueError:pass
    else:raise AssertionError('active policy escalation accepted')
    gc.collect(2);assert core._LAMMPS_GC_STATE['owners']==1;a.close();check_restored();checks.append('active_policy_escalation_rejected')
    return checks

def main():
    checks=lifecycle()
    before={'thresholds':gc.get_threshold(),'enabled':gc.isenabled()}
    obj=make();during={'thresholds':gc.get_threshold(),'enabled':gc.isenabled()}
    obj.close();after={'thresholds':gc.get_threshold(),'enabled':gc.isenabled()}
    check_restored()
    maps=sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if 'liblammps.so' in line})
    result={'passed':True,'scope':'real-library GC lifecycle, no MD or score; serial GC control/no active-policy fork',
            'core_path':str(Path(core.__file__).resolve()),'core_sha256':sha(core.__file__),
            'python':sys.version,'engine_environment_value':ORIGINAL_ENV,'checks':checks,
            'before':before,'during':during,'after':after,'native_maps':{path:sha(path) for path in maps}}
    if len(sys.argv)>1:Path(sys.argv[1]).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    try:main()
    finally:
        gc.set_threshold(*BASE);gc.enable() if ENABLED else gc.disable()
        if ORIGINAL_ENV is None:os.environ.pop(ENV,None)
        else:os.environ[ENV]=ORIGINAL_ENV
