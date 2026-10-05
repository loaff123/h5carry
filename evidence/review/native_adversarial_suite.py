"""Original independent fixture and mutation probes. Native-runner execution only."""
import copy
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

if os.environ.get('H5CARRY_TEST_NATIVE_CHILD')!='1':
    raise SystemExit('Run only through tests.native_runner')
import h5py
import numpy as np
from h5carry.model import Limits,fingerprint
from h5carry.plan import validate_plan
from h5carry.scan import make_plan_native
from h5carry.write import write_staging
from h5carry.verify import verify_export,inspect_output

root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=True)
source=root/'source.h5';output=root/'output.h5';limits=Limits(chunk_bytes=32)
with h5py.File(source,'w') as f:
    f.attrs['root_note']='review fixture'
    calibrate=f.create_dataset('calibration/value',data=np.array(-0.0,dtype='>f8'))
    f.attrs['calibration']=calibrate.ref
    scale=f.create_dataset('axis',data=np.array([0,1,2],dtype='>i4'));scale.make_scale('repeated label')
    g=f.create_group('selected')
    d=g.create_dataset('signal',data=np.arange(6,dtype='>i4').reshape(2,3),chunks=(1,3),maxshape=(None,3),compression='gzip',compression_opts=4,shuffle=True,fletcher32=True,fillvalue=-4)
    g['alias']=d; d.dims[1].attach_scale(scale);d.dims[1].label='coordinate'
    g['to_calibration']=h5py.SoftLink('/calibration/value')
    refs=g.create_dataset('refs',(2,),dtype=h5py.ref_dtype,chunks=(1,),compression='gzip',compression_opts=6)
    refs[:]=[calibrate.ref,h5py.Reference()]
    g.attrs['null_reference']=h5py.Reference()
    g.attrs['bytes']=np.array([b'a\0b',b''],dtype='S5')
    g.attrs['empty']=h5py.Empty('>f4')
    excluded=f.create_dataset('unselected/signal',data=[991,992,993]);excluded.attrs['excluded_marker']='not retained';excluded.dims[0].attach_scale(scale)
original_hash=fingerprint(source)
graph=make_plan_native(str(source),['/selected'],limits)
plan={'format':'h5carry-plan','version':1,'source':original_hash,'selections':['/selected'],'limits':limits.to_dict(),'graph':graph}
validate_plan(plan);write_staging(str(source),graph,str(output),limits)
rows=[]
verified=verify_export(str(source),str(output),plan,limits)
assert verified['status']=='verified',verified
expected_paths={'axis','calibration','selected','calibration/value','selected/alias','selected/refs','selected/signal','selected/to_calibration'}
with h5py.File(output,'r') as f:
    # Explicit original expectations, independent of the plan and verifier.
    seen=set()
    for base in ('/','/calibration','/selected'):
        group=f[base]
        for name in group:seen.add((base.strip('/')+'/' if base!='/' else '')+name)
    assert seen==expected_paths,(seen,expected_paths)
    assert f['selected/alias'].id==f['selected/signal'].id
    assert f['selected/signal'].dtype.str=='>i4'
    assert f['selected/signal'][...].tolist()==[[0,1,2],[3,4,5]]
    assert f['calibration/value'][()].tobytes()==np.float64(-0.).tobytes()
    assert f['selected/signal'].chunks==(1,3)
    assert f['selected/signal'].compression_opts==4
    assert f['selected/signal'].maxshape==(None,3)
    assert f['selected/signal'].fillvalue==-4
    assert f['selected/refs'].chunks==(1,) and f['selected/refs'].compression_opts==6
    assert f[f['selected/refs'][0]].id==f['calibration/value'].id
    assert not f['selected/refs'][1]
    assert not f['selected'].attrs['null_reference']
    assert f['selected'].attrs['bytes'].tobytes()==np.array([b'a\0b',b''],dtype='S5').tobytes()
    assert isinstance(f['selected'].attrs['empty'],h5py.Empty)
    assert f['selected/signal'].dims[1][0].id==f['axis'].id
    reverse=f['axis'].attrs['REFERENCE_LIST'];assert len(reverse)==1
    assert f[reverse[0]['dataset']].id==f['selected/signal'].id and int(reverse[0]['dimension'])==1
    assert f['selected'].get('to_calibration',getlink=True).path=='/calibration/value'
rows.append({'case':'manual_complete_subset','status':'passed','verification':verified})

def mutate(name, fn, changed_plan=None):
    target=root/(name+'.h5');shutil.copyfile(output,target)
    with h5py.File(target,'r+') as f:fn(f)
    result=verify_export(str(source),str(target),changed_plan or plan,limits)
    assert result['status']!='verified',(name,result)
    rows.append({'case':name,'status':'rejected','verification':result})

mutate('same_value_split_alias',lambda f:(f.__delitem__('selected/alias'),f.copy('selected/signal','selected/alias')))
mutate('reference_to_equal_wrong_identity',lambda f:(f.create_dataset('equal_calibration',data=np.array(-0.,dtype='>f8')),f['selected/refs'].__setitem__(0,f['equal_calibration'].ref)))
mutate('negative_zero_to_positive',lambda f:f['calibration/value'].__setitem__((),0.))
mutate('normal_root_attribute_omission',lambda f:f.attrs.__delitem__('root_note'))
mutate('excluded_marker_under_new_name',lambda f:f.create_dataset('renamed_extra',data=[991,992,993]))
mutate('soft_target_changed_same_value',lambda f:(f['selected'].__delitem__('to_calibration'),f['selected'].__setitem__('to_calibration',h5py.SoftLink('/selected/alias'))))
# Remove child from BOTH plan and output. Fresh source closure must reject it.
changed=copy.deepcopy(plan)
changed['graph']['links']=[x for x in changed['graph']['links'] if x['path']!='/selected/to_calibration']
validate_plan(changed)
mutate('joint_plan_output_child_omission',lambda f:f['selected'].__delitem__('to_calibration'),changed)
assert fingerprint(source)==original_hash
inspected=inspect_output(str(output),limits);assert inspected['status']=='inspected' and inspected['coverage']['source_equality'] is False
rows.append({'case':'source_unchanged_and_inspect_scope','status':'passed','source':original_hash,'inspection':inspected})
(root/'results.json').write_text(json.dumps(rows,indent=2))
print(json.dumps({'cases':len(rows),'statuses':[r['status'] for r in rows],'runtime':verified['runtime']}))
