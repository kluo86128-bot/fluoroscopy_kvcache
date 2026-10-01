import csv
from contextlib import contextmanager
import json
import os
import errno
import sys
from pathlib import Path


class RunBusy(RuntimeError):
    pass


@contextmanager
def run_lock(root):
    """OS locks release on process death, avoiding stale resume locks."""
    path = Path(root) / ".run.lock"
    with path.open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RunBusy("该运行目录已有训练或测试进程，不能同时启动或续跑") from error
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def write_error_status(path, value):
    """Failure reporting must never replace the original exception."""
    try:
        write_json(path, value)
    except Exception as error:
        try:
            print(f"无法写入失败状态 {path}: {error}", file=sys.stderr, flush=True)
        except Exception:
            pass


def is_storage_error(error):
    seen, pending = set(), [error]
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, OSError) and current.errno in (errno.ENOSPC, errno.EDQUOT, errno.EIO, errno.EROFS):
            return True
        pending.extend((current.__cause__, current.__context__))
    return False


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_torch(path, value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        # A Python file object preserves actionable OSError details on write failure.
        with temporary.open("wb") as output:
            torch.save(value, output)
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise
