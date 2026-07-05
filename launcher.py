# -*- coding: utf-8 -*-
"""Windows launcher for the locally hosted Streamlit desktop distribution."""
from __future__ import annotations
import logging
import os
import sys
import socket
import threading
import time
import traceback
import urllib.request
import webbrowser
from pathlib import Path


def resource_path(relative: str) -> Path:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base / relative


def free_port(preferred: int = 8501) -> int:
    for port in range(preferred, preferred + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return preferred


def user_data_dir() -> Path:
    base = Path(os.environ.get('LOCALAPPDATA', Path.home() / 'AppData' / 'Local'))
    path = base / 'StudentRecordExplorer'
    path.mkdir(parents=True, exist_ok=True)
    return path


def configure_logging(data_dir: Path) -> Path:
    log_path = data_dir / 'StudentRecordExplorer.log'
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        encoding='utf-8',
    )
    return log_path


def open_browser_when_ready(port: int) -> None:
    health_url = f'http://127.0.0.1:{port}/_stcore/health'
    app_url = f'http://127.0.0.1:{port}'
    for _ in range(120):
        try:
            with urllib.request.urlopen(health_url, timeout=1) as response:
                if response.status == 200:
                    webbrowser.open(app_url)
                    return
        except Exception:
            time.sleep(0.25)
    logging.error('The local web server did not become ready: %s', app_url)


def main() -> int:
    app = resource_path("app.py")
    port = free_port()
    data_dir = user_data_dir()
    log_path = configure_logging(data_dir)
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    os.environ["PYTHONUTF8"] = "1"
    os.environ["SRE_DESKTOP_MODE"] = "1"
    os.environ["SRE_DATA_DIR"] = str(data_dir / 'data')

    sys.argv = [
        'streamlit', 'run', str(app),
        "--server.headless=true",
        "--server.address=127.0.0.1",
        f"--server.port={port}",
        "--global.developmentMode=false",
        "--browser.gatherUsageStats=false",
    ]
    logging.info('Starting StudentRecord Explorer on 127.0.0.1:%s', port)
    threading.Thread(target=open_browser_when_ready, args=(port,), daemon=True).start()

    try:
        from streamlit.web import cli as streamlit_cli
        return int(streamlit_cli.main() or 0)
    except SystemExit as exc:
        return int(exc.code or 0)
    except Exception:
        logging.error('Fatal launcher error:\n%s', traceback.format_exc())
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(
                0,
                f'학생부 탐색기를 시작하지 못했습니다.\n로그: {log_path}',
                'StudentRecord Explorer',
                0x10,
            )
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
