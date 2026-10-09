"""Bounded asynchronous application logs, independent of Qt and acquisition."""

import atexit
from datetime import datetime
import json
import logging
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
import os
from pathlib import Path
from queue import Full, Queue
import sys


_listener = None
_queue_handler = None
_file_handler = None
_log_file = None


class LocalTimeFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        return datetime.fromtimestamp(record.created).astimezone().isoformat(timespec="milliseconds")


class BoundedQueueHandler(QueueHandler):
    """Disk stalls never block serial reception; report overflow when it recovers."""
    dropped = 0

    def enqueue(self, record):
        skipped = self.dropped
        if skipped:
            record.msg = f"日志队列曾丢弃 {skipped} 条记录；{record.msg}"
        try:
            self.queue.put_nowait(record)
            self.dropped = 0
        except Full:
            self.dropped += 1


class DrainingListener(QueueListener):
    def enqueue_sentinel(self):
        self.queue.put(self._sentinel)


def setup_logging(directory=None, max_bytes=5 * 1024 * 1024, backups=4):
    global _listener, _queue_handler, _file_handler, _log_file
    if _listener is not None:
        return _log_file
    root = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[2]
    directories = [Path(directory)] if directory else [
        root / "logs", Path(os.environ.get("LOCALAPPDATA", Path.home())) / "SkinDraw" / "logs"]
    last_error = None
    for folder in directories:
        try:
            folder.mkdir(parents=True, exist_ok=True)
            # Each application instance writes its own files, avoiding rotation races.
            filename = f"skindraw-{datetime.now():%Y%m%d-%H%M%S-%f}-{os.getpid()}.log"
            path = folder / filename
            handler = RotatingFileHandler(path, maxBytes=max_bytes, backupCount=backups, encoding="utf-8")
            break
        except OSError as error:
            last_error = error
    else:
        raise OSError(f"无法创建运行日志：{last_error}")
    # Retain ten runs including this one, each with bounded rotated files.
    previous = sorted(folder.glob("skindraw-*.log"), reverse=True)
    for old in previous[10:]:
        for target in (old, *(old.parent / f"{old.name}.{i}" for i in range(1, backups + 1))):
            try:
                target.unlink(missing_ok=True)
            except OSError:
                pass  # An active second instance may own this file.
    handler.setFormatter(LocalTimeFormatter("%(asctime)s %(levelname)s [%(threadName)s] %(name)s %(message)s"))
    queue_handler = BoundedQueueHandler(Queue(maxsize=1000))
    logger = logging.getLogger("skin_draw")
    logger.setLevel(logging.INFO)
    logger.addHandler(queue_handler)
    _file_handler, _queue_handler, _log_file = handler, queue_handler, path
    _listener = DrainingListener(queue_handler.queue, handler)
    _listener.start()
    return path


def shutdown_logging():
    global _listener, _queue_handler, _file_handler
    if _listener is not None:
        logging.getLogger("skin_draw").removeHandler(_queue_handler)
        _listener.stop()
        if _queue_handler.dropped:
            _file_handler.handle(logging.makeLogRecord({"name": "skin_draw", "levelno": logging.WARNING,
                "levelname": "WARNING", "msg": f"退出时日志队列累计丢弃 {_queue_handler.dropped} 条记录"}))
        _file_handler.close()
        _queue_handler.close()
        _listener = _queue_handler = _file_handler = None


def log_directory():
    return setup_logging().parent


def event(logger, level, message, **fields):
    logger.log(level, "%s %s", message, json.dumps(fields, ensure_ascii=False, default=str))


atexit.register(shutdown_logging)
