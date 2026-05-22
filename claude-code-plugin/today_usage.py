#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import monitor


def local_day(value: str) -> Optional[str]:
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone().date().isoformat()


def iter_usage_events(root: Path) -> Iterable[Dict[str, Any]]:
    for path in root.glob("**/*.jsonl"):
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                for line_no, line in enumerate(f, start=1):
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    usage = monitor.extract_usage(obj)
                    if not usage:
                        continue
                    yield {
                        "path": path,
                        "line_no": line_no,
                        "project": monitor.project_name(path, root),
                        "model": monitor.extract_model(obj) or "unknown",
                        "timestamp": monitor.parse_time(obj, path),
                        **usage,
                    }
        except FileNotFoundError:
            continue
        except OSError as exc:
            print(f"[WARN] failed to read {path}: {exc}")


def fmt_int(value: int) -> str:
    return f"{value:,}"


def add_row(bucket: Dict[str, int], event: Dict[str, Any]) -> None:
    bucket["input_tokens"] += int(event["input_tokens"])
    bucket["output_tokens"] += int(event["output_tokens"])
    bucket["total_tokens"] += int(event["input_tokens"]) + int(event["output_tokens"])
    bucket["events"] += 1


def print_table(title: str, rows: list[tuple[str, Dict[str, int]]]) -> None:
    if not rows:
        return
    print(f"\n{title}")
    print("-" * len(title))
    name_width = min(max(4, max(len(name) for name, _ in rows)), 42)
    print(f"{'Name':<{name_width}}  {'Total':>12}  {'Input':>12}  {'Output':>12}  {'Events':>6}")
    for name, data in rows:
        display = name if len(name) <= name_width else name[: name_width - 1] + "..."
        print(
            f"{display:<{name_width}}  "
            f"{fmt_int(data['total_tokens']):>12}  "
            f"{fmt_int(data['input_tokens']):>12}  "
            f"{fmt_int(data['output_tokens']):>12}  "
            f"{fmt_int(data['events']):>6}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Show today's Claude Code usage from local logs.")
    parser.add_argument("--day", help="Local day to report, in YYYY-MM-DD format. Defaults to today.")
    args = parser.parse_args()

    cfg = monitor.load_config()
    root = Path(cfg["claude_projects_dir"]).expanduser()
    selected_day = args.day or datetime.now().astimezone().date().isoformat()

    if not root.exists():
        print(f"Claude Code logs not found: {root}")
        return

    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "events": 0}
    by_project: Dict[str, Dict[str, int]] = defaultdict(lambda: {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "events": 0})
    by_model: Dict[str, Dict[str, int]] = defaultdict(lambda: {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "events": 0})

    for event in iter_usage_events(root):
        if local_day(str(event["timestamp"])) != selected_day:
            continue
        add_row(totals, event)
        add_row(by_project[str(event["project"])], event)
        add_row(by_model[str(event["model"])], event)

    print(f"Claude Code usage for {selected_day}")
    print(f"Logs: {root}")
    print(f"Events: {fmt_int(totals['events'])}")
    print(f"Input tokens: {fmt_int(totals['input_tokens'])}")
    print(f"Output tokens: {fmt_int(totals['output_tokens'])}")
    print(f"Total tokens: {fmt_int(totals['total_tokens'])}")

    project_rows = sorted(by_project.items(), key=lambda item: item[1]["total_tokens"], reverse=True)
    model_rows = sorted(by_model.items(), key=lambda item: item[1]["total_tokens"], reverse=True)
    print_table("By project", project_rows[:10])
    print_table("By model", model_rows[:10])


if __name__ == "__main__":
    main()
