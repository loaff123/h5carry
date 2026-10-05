"""Original source-derived budget guard checks, after the review repair."""
import copy,json,os,sys
from pathlib import Path
if os.environ.get('H5CARRY_TEST_NATIVE_CHILD')!='1':raise SystemExit('Use native runner')
import h5py,numpy as np
from h5carry.model import Limits,fingerprint,CarryError
from h5carry.scan import make_plan_native
from h5carry.plan import validate_plan
from h5carry.write import write_staging
root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=True)
s=root/'source.h5';o=root/'staged.h5'
with h5py.File(s,'w') as f:f['values']=np.arange(1024,dtype='u1')
g=make_plan_native(str(s),['/values'],Limits());g['payload_bytes']=1
limits=Limits(max_payload_bytes=1)
p={'format':'h5carry-plan','version':1,'source':fingerprint(s),'selections':['/values'],'limits':limits.to_dict(),'graph':g}
rows=[]
try:validate_plan(p)
except CarryError as exc:rows.append({'case':'metadata_total_contradiction','rejected_by':'codec','code':exc.code})
else:raise AssertionError('contradictory payload total accepted')
p=copy.deepcopy(p)
for item in p['graph']['objects']:
 if item['kind']=='dataset':item['metadata']['shape']=[1];item['metadata']['creation']['maxshape']=[1]
validate_plan(p)
sentinel=b'original staging sentinel';o.write_bytes(sentinel)
try:write_staging(str(s),p['graph'],str(o),limits)
except CarryError as exc:
 assert exc.code=='RESOURCE',exc.to_dict()
 assert o.read_bytes()==sentinel,'staging modified before true source budget check'
 rows.append({'case':'forged_shape','rejected_by':'actual_source_preflight','code':exc.code,'staging_unchanged':True})
else:raise AssertionError('actual over-budget source copied')
(root/'results.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows))
