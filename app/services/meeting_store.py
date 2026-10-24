import json
import os
import threading
from pathlib import Path
from app.config import MEETINGS_DIR

_lock = threading.RLock()


def meeting_dir(meeting_id: str) -> Path:
    # UUID IDs are validated by callers, and resolve guard protects against traversal.
    if not meeting_id.startswith("meeting_") or len(meeting_id) != 40:
        raise ValueError("Invalid meeting ID")
    root = MEETINGS_DIR.resolve()
    target = (root / meeting_id).resolve()
    if target.parent != root:
        raise ValueError("Invalid meeting ID")
    return target


def save_metadata(meeting_id: str, metadata: dict) -> None:
    folder = meeting_dir(meeting_id)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "meeting.json"
    temp = folder / "meeting.json.tmp"
    with _lock:
        temp.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp, path)


def load_metadata(meeting_id: str) -> dict | None:
    path = meeting_dir(meeting_id) / "meeting.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
