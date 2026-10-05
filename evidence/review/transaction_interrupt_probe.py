"""Original no-native fault probe for interruption immediately after link commit."""
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
from h5carry.model import Limits,fingerprint
from h5carry.transaction import export_checked,_publish

def plan_for(source):
    return {'format':'h5carry-plan','version':1,'source':fingerprint(source),'selections':['/'],'limits':Limits().to_dict(),'graph':{'objects':[{'id':'/','kind':'group','metadata':{'attributes':[]}}],'links':[],'references':[],'scales':[],'payload_bytes':0,'reasons':{'/':['selected']}}}

def native(req):
    if req['operation']=='write':
        Path(req['staging']).write_bytes(b'original verified fixture output')
        return {'ok':True,'result':None}
    return {'ok':True,'result':{'status':'verified','diagnostics':[],'coverage':{},'runtime':{}}}

rows=[]
for interrupted in ('output','report'):
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory); source=root/'source'; source.write_bytes(b'original fixture source')
        out=root/'out'; report=root/'report'
        def commit_then_interrupt(stage,destination):
            _publish(stage,destination)
            if Path(destination)==(out if interrupted=='output' else report):
                raise KeyboardInterrupt('interrupt after kernel commit')
        try:
            with patch('h5carry.transaction.run_native',side_effect=native),patch('h5carry.transaction._publish',side_effect=commit_then_interrupt):
                export_checked(source,plan_for(source),out,report)
        except BaseException as exc:
            rows.append({'interrupted_after':interrupted,'exception':type(exc).__name__,'reported_publication':getattr(exc,'publication',None),'actual_publication':{'output_published':out.exists(),'report_published':report.exists()}})
print(json.dumps(rows,indent=2))
