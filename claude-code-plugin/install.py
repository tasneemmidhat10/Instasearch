#!/usr/bin/env python3
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
PLUGIN_DIR = Path.home() / ".claude-code-usage-plugin"
CLAUDE_COMMANDS_DIR = Path.home() / ".claude" / "commands"
TODAY_USAGE_COMMAND = CLAUDE_COMMANDS_DIR / "today-usage.md"
SERVICE_NAME = "claude-code-usage-plugin"
MAC_LABEL = "com.ayman.claude-code-usage-plugin"
WINDOWS_TASK = "Claude Code Usage Plugin"
WINDOWS_LAUNCHER = PLUGIN_DIR / "run-monitor.cmd"


def prompt(label: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{label}{suffix}: ").strip()
    return value or default


def python_command() -> str:
    return sys.executable or shutil.which("python3") or shutil.which("python") or "python3"


def write_config() -> None:
    existing = {}
    config_path = PLUGIN_DIR / "config.json"
    if config_path.exists():
        try:
            existing = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}

    host = socket.gethostname()
    device_name = prompt("Device name", str(existing.get("device_name") or existing.get("device_id") or host))
    backend_url = str(existing.get("backend_url") or "https://claude-monitor-2p1u.onrender.com")
    ingest_key = str(existing.get("ingest_key") or "")
    poll_seconds = int(existing.get("poll_seconds") or 20)
    claude_projects_dir = str(existing.get("claude_projects_dir") or "")

    config = {
        "backend_url": backend_url.rstrip("/"),
        "ingest_key": ingest_key,
        "device_id": device_name,
        "device_name": device_name,
        "poll_seconds": poll_seconds,
        "claude_projects_dir": claude_projects_dir,
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def copy_files() -> None:
    PLUGIN_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(HERE / "monitor.py", PLUGIN_DIR / "monitor.py")
    shutil.copy2(HERE / "today_usage.py", PLUGIN_DIR / "today_usage.py")
    shutil.copy2(HERE / "config.example.json", PLUGIN_DIR / "config.example.json")


def install_slash_command() -> None:
    CLAUDE_COMMANDS_DIR.mkdir(parents=True, exist_ok=True)
    command = (
        "---\n"
        "description: Show today's Claude Code token usage from local logs\n"
        "allowed-tools: Bash\n"
        "---\n\n"
        "Run the installed local usage reporter and show the output to the user.\n\n"
        f"!`\"{python_command()}\" \"{PLUGIN_DIR / 'today_usage.py'}\"`\n"
    )
    TODAY_USAGE_COMMAND.write_text(command, encoding="utf-8")


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print("+ " + " ".join(cmd))
    return subprocess.run(cmd, check=check)


def is_success(cmd: list[str]) -> bool:
    return subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def windows_startup_dir() -> Path:
    appdata = os.environ.get("APPDATA")
    if appdata:
        return Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
    return Path.home() / "AppData" / "Roaming" / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"


def write_windows_launcher() -> None:
    WINDOWS_LAUNCHER.write_text(
        "@echo off\r\n"
        f'cd /d "{PLUGIN_DIR}"\r\n'
        f'"{python_command()}" "{PLUGIN_DIR / "monitor.py"}" >> "{PLUGIN_DIR / "out.log"}" 2>> "{PLUGIN_DIR / "err.log"}"\r\n',
        encoding="utf-8",
    )


def start_monitor_detached() -> None:
    stdout = (PLUGIN_DIR / "out.log").open("ab")
    stderr = (PLUGIN_DIR / "err.log").open("ab")
    kwargs = {}
    if platform.system().lower() == "windows":
        kwargs["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    subprocess.Popen(
        [python_command(), str(PLUGIN_DIR / "monitor.py")],
        cwd=str(PLUGIN_DIR),
        stdout=stdout,
        stderr=stderr,
        **kwargs,
    )


def install_linux() -> None:
    systemd_dir = Path.home() / ".config" / "systemd" / "user"
    systemd_dir.mkdir(parents=True, exist_ok=True)
    service_path = systemd_dir / f"{SERVICE_NAME}.service"
    service_path.write_text(
        f"""[Unit]
Description=Claude Code Usage Plugin
After=network-online.target

[Service]
Type=simple
WorkingDirectory={PLUGIN_DIR}
ExecStart=/usr/bin/env python3 {PLUGIN_DIR / "monitor.py"}
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
""",
        encoding="utf-8",
    )
    run(["systemctl", "--user", "daemon-reload"])
    run(["systemctl", "--user", "enable", "--now", f"{SERVICE_NAME}.service"])
    if not is_success(["systemctl", "--user", "is-active", "--quiet", f"{SERVICE_NAME}.service"]):
        raise RuntimeError(f"{SERVICE_NAME}.service was installed but did not start")
    print("\nInstalled and running in the background.")


def install_macos() -> None:
    plist_dir = Path.home() / "Library" / "LaunchAgents"
    plist_dir.mkdir(parents=True, exist_ok=True)
    plist_path = plist_dir / f"{MAC_LABEL}.plist"
    plist_path.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>{MAC_LABEL}</string>
  <key>ProgramArguments</key><array><string>{python_command()}</string><string>{PLUGIN_DIR / "monitor.py"}</string></array>
  <key>WorkingDirectory</key><string>{PLUGIN_DIR}</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>{PLUGIN_DIR / "out.log"}</string>
  <key>StandardErrorPath</key><string>{PLUGIN_DIR / "err.log"}</string>
</dict></plist>
""",
        encoding="utf-8",
    )
    run(["launchctl", "unload", str(plist_path)], check=False)
    run(["launchctl", "load", str(plist_path)])
    if not is_success(["launchctl", "list", MAC_LABEL]):
        raise RuntimeError(f"{MAC_LABEL} was installed but did not start")
    print("\nInstalled and running in the background.")


def install_windows() -> None:
    write_windows_launcher()
    task_cmd = f'"{WINDOWS_LAUNCHER}"'
    task_created = run([
        "schtasks",
        "/Create",
        "/TN",
        WINDOWS_TASK,
        "/SC",
        "ONLOGON",
        "/TR",
        task_cmd,
        "/RL",
        "LIMITED",
        "/F",
    ], check=False).returncode == 0

    if task_created and is_success(["schtasks", "/Query", "/TN", WINDOWS_TASK]):
        run(["schtasks", "/Run", "/TN", WINDOWS_TASK], check=False)
        print("\nInstalled and running in the background with Windows Task Scheduler.")
        return

    startup_dir = windows_startup_dir()
    startup_dir.mkdir(parents=True, exist_ok=True)
    startup_cmd = startup_dir / "claude-code-usage-plugin.cmd"
    shutil.copy2(WINDOWS_LAUNCHER, startup_cmd)
    start_monitor_detached()
    print("\nTask Scheduler setup was blocked, so a per-user Startup launcher was installed instead.")
    print(f"Startup launcher:\n  {startup_cmd}")
    print("The monitor was started in the background for this session.")


def start_detached() -> None:
    start_monitor_detached()
    print("\nAutomatic service setup is not available on this OS.")
    print("The monitor was started in the background for this session.")


def main() -> None:
    print("Claude Code Usage Plugin installer\n")
    copy_files()
    write_config()
    install_slash_command()

    system = platform.system().lower()
    if system == "linux":
        install_linux()
    elif system == "darwin":
        install_macos()
    elif system == "windows":
        install_windows()
    else:
        print(f"Unsupported OS for automatic service setup: {platform.system()}")
        start_detached()
        return

    print(f"\nConfig written to:\n  {PLUGIN_DIR / 'config.json'}")
    print(f"Claude Code command installed:\n  /today-usage ({TODAY_USAGE_COMMAND})")


if __name__ == "__main__":
    main()
