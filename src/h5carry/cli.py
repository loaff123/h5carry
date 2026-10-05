"""Four bounded native-HDF5 commands for trusted, quiescent local data."""
from __future__ import annotations

import argparse
from dataclasses import fields
import sys

from . import __version__
from .model import CarryError, Limits, canonical_json
from .plan import load_plan, make_plan, save_plan
from .transaction import PublicationError, assert_distinct, export_checked, inspect_checked, verify_checked


def _limit_options(parser):
    for field in fields(Limits):
        parser.add_argument('--'+field.name.replace('_','-'), type=int, default=field.default,
                            help=f'lower the {field.name} ceiling (default {field.default})')


def parser():
    result=argparse.ArgumentParser(prog='h5carry', description='Plan, export and verify complete objects from trusted local HDF5 archives. Keep source files closed to writers.')
    result.add_argument('--version',action='version',version='h5carry '+__version__)
    commands=result.add_subparsers(dest='command',required=True)
    plan=commands.add_parser('plan',help='inspect selected dependencies without exporting')
    plan.add_argument('source'); plan.add_argument('--select',action='append',required=True)
    plan.add_argument('--out',required=True); _limit_options(plan)
    export=commands.add_parser('export',help='create a new independently verified HDF5 file')
    export.add_argument('source'); export.add_argument('--plan',required=True)
    export.add_argument('--out',required=True); export.add_argument('--report',required=True)
    verify=commands.add_parser('verify',help='compare reopened source, plan and output')
    verify.add_argument('source'); verify.add_argument('output')
    verify.add_argument('--plan',required=True); verify.add_argument('--report',required=True)
    inspect=commands.add_parser('inspect',help='check output structure, without claiming source equality')
    inspect.add_argument('output'); inspect.add_argument('--report',required=True); _limit_options(inspect)
    return result


def _emit(value):
    payload=canonical_json(value)
    if len(payload)>1024*1024:
        payload=canonical_json({'status':'incomplete','diagnostics':[{'code':'RESOURCE','message':'CLI result exceeds 1 MiB cap; inspect the report if one was published'}]})
    sys.stdout.write(payload.decode('utf-8')+'\n')


def main(argv=None):
    args=parser().parse_args(argv)
    try:
        if args.command in ('plan','inspect'):
            limits=Limits(**{field.name:getattr(args,field.name) for field in fields(Limits)})
        if args.command=='plan':
            assert_distinct(args.source,args.out)
            value=make_plan(args.source,args.select,limits)
            save_plan(value,args.out)
            _emit({'status':'planned','objects':len(value['graph']['objects']),
                   'links':len(value['graph']['links']),'payload_bytes':value['graph']['payload_bytes'],
                   'source':value['source']})
            return 0
        if args.command=='export':
            value=export_checked(args.source,load_plan(args.plan),args.out,args.report,plan_path=args.plan)
        elif args.command=='verify':
            value=verify_checked(args.source,args.output,load_plan(args.plan),args.report,plan_path=args.plan)
        else:
            value=inspect_checked(args.output,args.report,limits)
        _emit(value)
        return 0 if value['status'] in ('verified','inspected') else (1 if value['status']=='mismatch' else 2)
    except CarryError as exc:
        value={'status':'mismatch' if exc.code=='MISMATCH' else 'incomplete','diagnostics':[exc.to_dict()]}
        if isinstance(exc,PublicationError):
            value['publication']=exc.publication
        _emit(value)
        return 1 if exc.code=='MISMATCH' else 2
    except KeyboardInterrupt:
        _emit({'status':'incomplete','diagnostics':[{'code':'ERROR','message':'operation interrupted'}]})
        return 2
    except Exception as exc:
        # A programming failure or unexpected runtime crash is not unsupported data.
        _emit({'status':'incomplete','diagnostics':[{'code':'ERROR','message':'unexpected '+type(exc).__name__+'; operation did not complete'}]})
        return 2
