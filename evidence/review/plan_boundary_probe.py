"""Original exact-byte-cap plan roundtrip and malformed-field classification."""
import json,tempfile
from pathlib import Path
from h5carry.model import Limits,canonical_json,CarryError
from h5carry.plan import save_plan,load_plan,validate_plan

def empty():
 return {'format':'h5carry-plan','version':1,'source':{'sha256':'0'*64,'size':0},'selections':['/'],'limits':Limits().to_dict(),'graph':{'objects':[{'id':'/','kind':'group','metadata':{'attributes':[]}}],'links':[],'references':[],'scales':[],'payload_bytes':0,'reasons':{'/':['selected']}}}
p=empty()
for _ in range(5):p['limits']['max_plan_bytes']=len(canonical_json(p))
assert p['limits']['max_plan_bytes']==len(canonical_json(p))
rows=[]
with tempfile.TemporaryDirectory() as d:
 path=Path(d)/'p.json';save_plan(p,path)
 try:
  load_plan(path,Limits(max_plan_bytes=p['limits']['max_plan_bytes']))
  status='accepted'
 except CarryError as exc:status={'code':exc.code,'message':exc.message}
 rows.append({'case':'exact_active_byte_cap','limit':p['limits']['max_plan_bytes'],'saved_bytes':path.stat().st_size,'result':status})
for field in ('hard_target','reference_owner','reference_target','scale_consumer','reason_value'):
 p=empty()
 if field=='hard_target':p['graph']['links']=[{'path':'/x','kind':'hard','target':[]}]
 elif field=='reference_owner':p['graph']['references']=[{'owner':[],'attribute':None,'index':0,'target':None}]
 elif field=='reference_target':p['graph']['references']=[{'owner':'/','attribute':None,'index':0,'target':[]}]
 elif field=='scale_consumer':p['graph']['scales']=[{'consumer':[],'axis':0,'scale':'/'}]
 elif field=='reason_value':p['graph']['reasons']['/']=[{}]
 try:validate_plan(p);status='accepted'
 except Exception as exc:status={'exception':type(exc).__name__,'code':getattr(exc,'code',None)}
 rows.append({'case':field,'result':status})
print(json.dumps(rows,indent=2))
