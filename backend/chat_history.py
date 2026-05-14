"""
Disk-persisted chat history.
Each chat is a JSON file under <chroma_db>/chats/<chat_id>.json containing the
list of messages. A sidecar <chroma_db>/chats/index.json stores chat metadata
(id, title, timestamps) so we can list chats without opening every file.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

from . import config


CHAT_DIR = os.path.join(config.CHROMA_DIR, "chats")
INDEX_PATH = os.path.join(CHAT_DIR, "index.json")
os.makedirs(CHAT_DIR, exist_ok=True)


# internal
def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_index() -> Dict[str, dict]:
    if not os.path.exists(INDEX_PATH):
        return {}
    try:
        with open(INDEX_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save_index(idx: Dict[str, dict]) -> None:
    tmp = INDEX_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(idx, f, indent=2, ensure_ascii=False)
    os.replace(tmp, INDEX_PATH)


def _messages_path(chat_id: str) -> str:
    return os.path.join(CHAT_DIR, f"{chat_id}.json")


def _save_messages(chat_id: str, messages: List[dict]) -> None:
    p = _messages_path(chat_id)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(messages, f, indent=2, ensure_ascii=False)
    os.replace(tmp, p)


# public api
def list_chats() -> List[dict]:
    """Return chat metadata sorted newest-updated first."""
    idx = _load_index()
    chats = list(idx.values())
    chats.sort(key=lambda c: c.get("updated_at", ""), reverse=True)
    return chats


def get_chat_choices() -> List:
    """[(label, id), ...] for gr.Radio / gr.Dropdown."""
    return [(c["title"], c["id"]) for c in list_chats()]


def create_chat(title: Optional[str] = None) -> str:
    chat_id = uuid.uuid4().hex[:12]
    now = _now()
    idx = _load_index()
    idx[chat_id] = {
        "id": chat_id,
        "title": title or "New chat",
        "created_at": now,
        "updated_at": now,
    }
    _save_index(idx)
    _save_messages(chat_id, [])
    return chat_id


def get_messages(chat_id: str) -> List[dict]:
    if not chat_id:
        return []
    p = _messages_path(chat_id)
    if not os.path.exists(p):
        return []
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def add_message(chat_id: str, role: str, content: str) -> None:
    """Append a message and update the chat's title (from first user msg) + updated_at."""
    msgs = get_messages(chat_id)
    msgs.append({"role": role, "content": content})
    _save_messages(chat_id, msgs)

    idx = _load_index()
    if chat_id in idx:
        if role == "user" and idx[chat_id].get("title", "New chat") == "New chat":
            t = content.strip().replace("\n", " ")
            idx[chat_id]["title"] = (t[:60] + "…") if len(t) > 60 else t or "New chat"
        idx[chat_id]["updated_at"] = _now()
        _save_index(idx)


def delete_chat(chat_id: str) -> bool:
    idx = _load_index()
    existed = chat_id in idx
    if existed:
        del idx[chat_id]
        _save_index(idx)
    p = _messages_path(chat_id)
    if os.path.exists(p):
        try:
            os.remove(p)
        except OSError:
            pass
    return existed


def chat_exists(chat_id: str) -> bool:
    return chat_id in _load_index()


def clear_all() -> None:
    """Delete every chat and the index. Used at server startup so each launch
    starts with no prior conversations."""
    if not os.path.isdir(CHAT_DIR):
        os.makedirs(CHAT_DIR, exist_ok=True)
        return
    for name in os.listdir(CHAT_DIR):
        path = os.path.join(CHAT_DIR, name)
        if os.path.isfile(path):
            try:
                os.remove(path)
            except OSError:
                pass
