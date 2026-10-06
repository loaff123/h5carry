from pathlib import Path
import hashlib
import importlib.resources
import json
import sys
import h5carry
root=Path(h5carry.__file__).resolve()
assert Path(sys.prefix).resolve() in root.parents, (root,sys.prefix)
assert 'site-packages' in root.parts
assert 'h5py' not in sys.modules and 'numpy' not in sys.modules
assert h5carry.__version__ == '0.2.0a1'
schemas={}
for name in ('plan-v1','plan-v2','selection-v1'):
    raw=importlib.resources.files('h5carry').joinpath('schemas/'+name+'.schema.json').read_bytes()
    json.loads(raw)
    schemas[name]=hashlib.sha256(raw).hexdigest()
schema=importlib.resources.files('h5carry').joinpath('schemas/plan-v1.schema.json').read_bytes()
assert json.loads(schema)['title']=='H5Carry plan v1'
print(json.dumps({'version':h5carry.__version__,'installed_site_package':True,'native_free_import':True,'schema_present':True,'schema_sha256':hashlib.sha256(schema).hexdigest(),'schemas':schemas,'python':sys.version.split()[0]},sort_keys=True))
