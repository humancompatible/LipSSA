"""One JSON record per experiment cell, keyed by the cell's parameters.

A cell is identified by a `key` dict (network spec, method, seed, budget, ...).
The record is stored at `<root>/<experiment>/<hash of key>.json` and carries the
key inside it, so a directory of records can be listed and filtered without
re-deriving file names. `run_cached` is the only entry point most code needs:
it returns the stored record if there is one and otherwise computes, stores and
returns it. That is what lets a grid be split over many short cluster jobs and
resumed after an interruption: every job walks the same grid and only computes
the cells whose record is missing.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parent.parent / 'experiments_out'


def _jsonable(obj):
    """Fallback for json.dumps: numpy/torch scalars and arrays, paths."""
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    if hasattr(obj, 'item'):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


def _canonical(key: dict) -> str:
    return json.dumps(key, sort_keys=True, separators=(',', ':'), default=_jsonable)


def result_path(experiment: str, key: dict, root: Path = DEFAULT_ROOT) -> Path:
    digest = hashlib.sha1(_canonical(key).encode()).hexdigest()[:16]
    return Path(root) / experiment / f"{digest}.json"


def load(path: Path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    with open(path) as fh:
        return json.load(fh)


def save(path: Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.json.tmp')
    with open(tmp, 'w') as fh:
        json.dump(record, fh, indent=1, default=_jsonable)
    tmp.replace(path)          # atomic on POSIX: a reader never sees a half-written file


def run_cached(experiment: str, key: dict, fn, root: Path = DEFAULT_ROOT, force: bool = False) -> dict:
    """Return the record for `key`, computing it with `fn()` if it is missing.

    `fn` must return a JSON-serialisable dict; the key is merged in under
    'key' before saving so the record is self-describing.
    """
    path = result_path(experiment, key, root)
    if not force:
        record = load(path)
        if record is not None:
            return record
    record = dict(fn())
    record['key'] = key
    save(path, record)
    return record


def collect(experiment: str, root: Path = DEFAULT_ROOT, **filters) -> list[dict]:
    """All records of an experiment whose key matches every `filters` item.

    A filter value that is a list/tuple/set matches any of its members.
    """
    out = []
    for path in sorted((Path(root) / experiment).glob('*.json')):
        record = load(path)
        if record is None:
            continue
        key = record.get('key', {})
        ok = True
        for name, wanted in filters.items():
            have = key.get(name)
            if isinstance(wanted, (list, tuple, set)):
                ok &= have in wanted
            else:
                ok &= have == wanted
        if ok:
            out.append(record)
    return out
