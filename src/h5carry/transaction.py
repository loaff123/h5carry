"""No-clobber staged publication. The output/report pair is NOT atomic."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
import stat

from .model import CarryError, Limits, canonical_json, fingerprint
from .plan import validate_plan
from .supervisor import run_native


class PublicationError(CarryError):
    def __init__(self, message, output_published=False, report_published=False):
        super().__init__('ERROR', message)
        self.publication = {'output_published':output_published,'report_published':report_published}


def assert_distinct(*paths):
    values = [Path(path).absolute() for path in paths if path is not None]
    for index, left in enumerate(values):
        for right in values[index+1:]:
            try:
                same = left.resolve() == right.resolve() or (left.exists() and right.exists() and os.path.samefile(left,right))
            except OSError as exc:
                raise CarryError('INVALID', f'cannot resolve file identity: {exc.strerror}') from exc
            if same:
                raise CarryError('INVALID', 'source, plan, output and report must be distinct files')


def _vacant(path):
    # This is an early friendly check only. Atomic link decides the final race.
    if os.path.lexists(path):
        raise CarryError('INVALID', 'destination already exists; no files are overwritten')


def _stage(destination):
    try:
        fd, name = tempfile.mkstemp(prefix='.h5carry-', dir=Path(destination).parent)
        os.close(fd)
        return Path(name)
    except OSError as exc:
        raise CarryError('INVALID', f'cannot create destination staging file: {exc.strerror}') from exc


def _write_json(path, value, maximum):
    data = canonical_json(value) + b'\n'
    if len(data) > maximum:
        raise CarryError('RESOURCE', 'report exceeds byte limit')
    with open(path,'wb') as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())


def _sync_file(path):
    with open(path,'rb') as stream:
        os.fsync(stream.fileno())


def _publish(stage, destination):
    os.link(stage, destination, follow_symlinks=False)


def _same_published_file(stage, destination):
    """Reconcile link completion after an asynchronous interruption.

    None means the filesystem did not let us establish the current state. Never
    follow a final symlink or remove a final path based on this observation.
    """
    if stage is None:
        return False
    try:
        left = os.stat(stage, follow_symlinks=False)
        right = os.stat(destination, follow_symlinks=False)
    except FileNotFoundError:
        return False
    except OSError:
        return None
    return stat.S_ISREG(right.st_mode) and (left.st_dev,left.st_ino)==(right.st_dev,right.st_ino)


def _native(request):
    result = run_native(request)
    if not result['ok']:
        raise CarryError(**result['error'])
    return result['result']


def _source_matches(source, expected):
    if fingerprint(source) != expected:
        raise CarryError('SOURCE_CHANGED', 'source fingerprint changed; recreate the plan after closing writers')


def export_checked(source, plan, output, report, *, plan_path=None):
    """Publish verified output, then report, with explicit partial-state failure.

    After successful output publication, a report publication failure never
    deletes the output. A returned PublicationError records that state. Neither
    file may overwrite any existing directory entry. Pair-level atomicity and
    crash/power-loss durability are not promised.
    """
    validate_plan(plan)
    limits = Limits.from_dict(plan['limits'])
    source, output, report = Path(source), Path(output), Path(report)
    assert_distinct(source, plan_path, output, report)
    _vacant(output); _vacant(report)
    _source_matches(source, plan['source'])
    if plan['version'] == 2:
        derived = _native({'operation':'plan_v2','source':str(source),'request':plan['request'],'limits':limits.to_dict()})
        for field in ('graph','source_objects','selections','transformations'):
            if canonical_json(derived[field]) != canonical_json(plan[field]):
                raise CarryError('MISMATCH','Typed plan differs from current source before staging: '+field)
        _source_matches(source,plan['source'])
    stages=[]
    staged_output=staged_report=None
    output_published=False
    report_published=False
    try:
        staged_output = _stage(output); stages.append(staged_output)
        if plan['version'] == 2:
            _native({'operation':'write_v2','source':str(source),'plan':plan,
                     'staging':str(staged_output),'limits':limits.to_dict()})
        else:
            _native({'operation':'write','source':str(source),'graph':plan['graph'],
                     'staging':str(staged_output),'limits':limits.to_dict()})
        checked = _native({'operation':'verify','source':str(source),'output':str(staged_output),
                           'plan':plan,'limits':limits.to_dict()})
        if checked.get('status') != 'verified':
            code = 'MISMATCH' if checked.get('status') == 'mismatch' else 'ERROR'
            diagnostics=checked.get('diagnostics',[])
            detail = diagnostics[0].get('message','') if diagnostics else ''
            raise CarryError(code, 'staged output did not verify' + (': '+detail if detail else ''))
        _source_matches(source, plan['source'])
        output_fingerprint = fingerprint(staged_output)
        value = dict(checked)
        value.update({'format':'h5carry-report','version':1,'operation':'export',
                      'source':plan['source'],'plan_sha256':hashlib.sha256(canonical_json(plan)).hexdigest(),
                      'output':output_fingerprint,'limits':limits.to_dict(),
                      'publication':{'output_published':True,'report_published':True}})
        staged_report = _stage(report); stages.append(staged_report)
        _write_json(staged_report, value, limits.max_plan_bytes)
        _sync_file(staged_output)
        _publish(staged_output, output); output_published=True
        _publish(staged_report, report); report_published=True
        return value
    except BaseException as exc:
        # A signal can arrive after the link syscall succeeded but before the
        # next Python assignment. Observe stage/final inode identity first.
        output_published=_same_published_file(staged_output,output)
        report_published=_same_published_file(staged_report,report)
        if output_published is not False or report_published is not False:
            raise PublicationError('publication did not finish normally; recorded final-file state was checked against staging identities',
                                   output_published,report_published) from exc
        if isinstance(exc,OSError):
            raise CarryError('INVALID','output was not published: '+str(exc.strerror)) from exc
        raise
    finally:
        cleanup_errors=[]
        for stage in stages:
            try:
                stage.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                cleanup_errors.append(str(stage))
        if cleanup_errors:
            # Only files created by this invocation are attempted. Do not imply
            # successful cleanup or delete anything at final user paths.
            raise PublicationError('could not clean invocation-owned partial staging files: '+', '.join(cleanup_errors),
                                   output_published,report_published)


def verify_checked(source, output, plan, report=None, *, plan_path=None):
    validate_plan(plan)
    limits=Limits.from_dict(plan['limits'])
    assert_distinct(source, output, plan_path, report)
    if report is not None:
        _vacant(report)
    _source_matches(source,plan['source'])
    output_before=fingerprint(output)
    result=_native({'operation':'verify','source':str(source),'output':str(output),'plan':plan,'limits':limits.to_dict()})
    _source_matches(source,plan['source'])
    if fingerprint(output)!=output_before:
        raise CarryError('SOURCE_CHANGED','output changed during verification; close all writers')
    result.update({'format':'h5carry-report','version':1,'operation':'verify','source':plan['source'],
                   'output':output_before,'plan_sha256':hashlib.sha256(canonical_json(plan)).hexdigest(),
                   'limits':limits.to_dict()})
    if report is not None:
        publish_report(result,report,limits)
    return result


def inspect_checked(output, report=None, limits=None):
    limits=limits or Limits()
    assert_distinct(output,report)
    if report is not None:
        _vacant(report)
    before=fingerprint(output)
    result=_native({'operation':'inspect','output':str(output),'limits':limits.to_dict()})
    if fingerprint(output)!=before:
        raise CarryError('SOURCE_CHANGED','file changed during inspection; close all writers')
    result.update({'format':'h5carry-report','version':1,'operation':'inspect','output':before,'limits':limits.to_dict()})
    if report is not None:
        publish_report(result,report,limits)
    return result


def publish_report(value,destination,limits):
    _vacant(destination)
    stage=_stage(destination)
    try:
        _write_json(stage,value,limits.max_plan_bytes)
        _publish(stage,destination)
    except OSError as exc:
        raise CarryError('INVALID','report was not published: '+str(exc.strerror)) from exc
    finally:
        stage.unlink(missing_ok=True)
