"""Native-library-free limits, diagnostics and canonical encodings."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import os
from pathlib import Path
import stat


class CarryError(Exception):
    """An anticipated invalid, unsupported or incomplete operation."""
    def __init__(self, code: str, message: str, path: str | None = None):
        super().__init__(message)
        self.code, self.message, self.path = code, message, path

    def to_dict(self):
        value = {'code': self.code, 'message': self.message}
        if self.path is not None:
            value['path'] = self.path
        return value


@dataclass(frozen=True)
class Limits:
    """Release ceilings; callers may lower them, never silently raise them."""
    max_objects: int = 10_000
    max_edges: int = 50_000
    max_plan_bytes: int = 8 * 1024 * 1024
    max_payload_bytes: int = 256 * 1024 * 1024
    chunk_bytes: int = 1024 * 1024
    max_attribute_bytes: int = 256 * 1024
    max_depth: int = 64
    max_name_bytes: int = 1024
    max_rank: int = 32
    discovery_seconds: int = 60

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if type(value) is not int or value < 1 or value > field.default:
                raise CarryError('INVALID', f'{field.name} must be an integer in [1, {field.default}]')

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if type(value) is not dict or set(value) != {f.name for f in fields(cls)}:
            raise CarryError('INVALID', 'limits must contain exactly the documented fields')
        return cls(**value)


def canonical_json(value) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                          allow_nan=False).encode('utf-8')
    except (TypeError, ValueError, UnicodeError) as exc:
        raise CarryError('INVALID', 'not a finite UTF-8 JSON value') from exc


def validate_path(value: str, limits: Limits | None = None) -> str:
    limits = limits or Limits()
    if type(value) is not str or not value.startswith('/') or '\x00' in value:
        raise CarryError('INVALID', 'HDF5 path must be an absolute UTF-8 name')
    try:
        encoded = value.encode('utf-8')
    except UnicodeError as exc:
        raise CarryError('INVALID', 'HDF5 path is not UTF-8') from exc
    parts = value.split('/')[1:]
    if value != '/' and any(p in ('', '.', '..') for p in parts):
        raise CarryError('INVALID', 'HDF5 path must be normalized', value)
    if len(encoded) > limits.max_name_bytes or len(parts) > limits.max_depth:
        raise CarryError('RESOURCE', 'HDF5 path exceeds name or depth limit')
    return value


def open_regular(path: str | Path):
    """Open read-only without a FIFO handshake, then require a regular file."""
    fd = os.open(path, os.O_RDONLY | getattr(os,'O_NONBLOCK',0))
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise CarryError('INVALID','input must be a regular local file')
        return os.fdopen(fd,'rb',buffering=0)
    except BaseException:
        os.close(fd)
        raise


def fingerprint(path: str | Path) -> dict:
    """Hash a regular local file in bounded blocks and detect observed mutation.

    Hashing is synchronous and outside the native-operation wall deadline. It
    cannot prove quiescence or bound kernel filesystem I/O latency.
    """
    try:
        with open_regular(path) as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise CarryError('INVALID', 'source must be a regular local file')
            digest, total = hashlib.sha256(), 0
            while block := stream.read(1024 * 1024):
                digest.update(block)
                total += len(block)
            after = os.fstat(stream.fileno())
        current = os.stat(path)
    except OSError as exc:
        raise CarryError('INVALID', f'cannot read source: {exc.strerror}') from exc
    def identity(s):
        return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns
    if identity(before) != identity(after) or identity(after) != identity(current) or total != after.st_size:
        raise CarryError('SOURCE_CHANGED', 'source changed while hashing; close all writers')
    return {'sha256': digest.hexdigest(), 'size': total}
