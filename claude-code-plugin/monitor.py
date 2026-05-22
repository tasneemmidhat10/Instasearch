#!/usr/bin/env python3
import hashlib
import json
import platform
import socket
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

HERE = Path(__file__).resolve().parent
STATE_FILE = Path.home() / ".claude_code_usage_monitor_state.json"

DEFAULT_CONFIG = {
    "backend_url": "https://claude-monitor-2p1u.onrender.com",
    "ingest_key": "",
    "device_id": "",
    "device_name": "",
    "poll_seconds": 20,
    "claude_projects_dir": ""
}

def load_config() -> Dict[str, Any]:
    p = HERE / "config.json"
    cfg = DEFAULT_CONFIG.copy()
    if p.exists():
        cfg.update(json.loads(p.read_text(encoding="utf-8")))
    hostname = socket.gethostname()
    if not cfg["device_id"]:
        cfg["device_id"] = hostname
    if not cfg["device_name"]:
        cfg["device_name"] = hostname
    if not cfg["claude_projects_dir"]:
        cfg["claude_projects_dir"] = str(Path.home() / ".claude" / "projects")
    return cfg

def load_state() -> Dict[str, Any]:
    if not STATE_FILE.exists():
        return {"files": {}}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"files": {}}

def save_state(state: Dict[str, Any]) -> None:
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(STATE_FILE)

def parse_time(obj: Dict[str, Any], path: Path) -> str:
    for key in ("timestamp", "created_at", "time"):
        v = obj.get(key)
        if v:
            try:
                return datetime.fromisoformat(str(v).replace("Z", "+00:00")).astimezone(timezone.utc).isoformat()
            except Exception:
                pass
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()

def extract_usage(obj: Dict[str, Any]) -> Optional[Dict[str, int]]:
    usage = obj.get("usage")
    if not usage and isinstance(obj.get("message"), dict):
        usage = obj["message"].get("usage")
    if not usage and isinstance(obj.get("response"), dict):
        usage = obj["response"].get("usage")
    if not isinstance(usage, dict):
        return None
    out = {
        "input_tokens": int(usage.get("input_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0),
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }
    return out if sum(out.values()) > 0 else None

def extract_model(obj: Dict[str, Any]) -> Optional[str]:
    for src in (obj, obj.get("message") if isinstance(obj.get("message"), dict) else {}, obj.get("response") if isinstance(obj.get("response"), dict) else {}):
        for k in ("model", "model_id"):
            if src.get(k):
                return str(src[k])
    return None

def project_name(path: Path, root: Path) -> str:
    try:
        rel = path.relative_to(root)
        return rel.parts[0] if rel.parts else path.parent.name
    except Exception:
        return path.parent.name

def raw_id(path: Path, line_no: int, line: str) -> str:
    h = hashlib.sha256()
    h.update(str(path).encode())
    h.update(str(line_no).encode())
    h.update(line.encode())
    return h.hexdigest()

def post(cfg: Dict[str, Any], payload: Dict[str, Any]) -> None:
    url = cfg["backend_url"].rstrip("/") + "/api/events"
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST", headers={"Content-Type": "application/json", "X-Ingest-Key": cfg.get("ingest_key", "")})
    try:
        with urllib.request.urlopen(req, timeout=12) as r:
            if r.status >= 300:
                print(f"[WARN] backend returned HTTP {r.status}")
    except urllib.error.HTTPError as e:
        print(f"[WARN] backend HTTP {e.code}: {e.read().decode(errors='ignore')}")
    except Exception as e:
        print(f"[WARN] backend error: {e}")

def process_file(path: Path, root: Path, cfg: Dict[str, Any], state: Dict[str, Any]) -> None:
    key = str(path)
    fs = state["files"].setdefault(key, {"offset": 0, "line": 0})
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return
    if size < fs.get("offset", 0):
        fs["offset"] = 0
        fs["line"] = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        f.seek(fs.get("offset", 0))
        while True:
            line = f.readline()
            if not line:
                break
            fs["line"] = int(fs.get("line", 0)) + 1
            line_no = fs["line"]
            try:
                obj = json.loads(line)
            except Exception:
                continue
            usage = extract_usage(obj)
            if not usage:
                continue
            payload = {
                "source": "claude-code",
                "device_id": cfg["device_id"],
                "device_name": cfg["device_name"],
                "hostname": socket.gethostname(),
                "os": f"{platform.system()} {platform.release()}",
                "project": project_name(path, root),
                "model": extract_model(obj),
                "conversation_id": path.stem,
                "event_type": "usage",
                "timestamp": parse_time(obj, path),
                "raw_id": raw_id(path, line_no, line),
                "metadata": {"log_file": str(path), "line": line_no},
                **usage,
            }
            post(cfg, payload)
        fs["offset"] = f.tell()

def main():
    cfg = load_config()
    root = Path(cfg["claude_projects_dir"]).expanduser()
    state = load_state()
    state.setdefault("files", {})
    print("Claude Code Usage Plugin")
    print(f"Device: {cfg['device_name']} ({cfg['device_id']})")
    print(f"Backend: {cfg['backend_url']}")
    print(f"Logs: {root}")
    while True:
        if root.exists():
            for p in root.glob("**/*.jsonl"):
                try:
                    process_file(p, root, cfg, state)
                except Exception as e:
                    print(f"[WARN] failed on {p}: {e}")
            save_state(state)
        else:
            print(f"[WARN] Claude Code logs not found: {root}")
        time.sleep(int(cfg.get("poll_seconds", 20)))

if __name__ == "__main__":
    main()
