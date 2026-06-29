"""Build a macOS .app bundle with PyInstaller and zip it for GitHub Releases."""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path
from shutil import which


ROOT = Path(__file__).resolve().parents[1]
DIST_DIR = ROOT / "dist"
APP_NAME = "Verilume"


def main() -> int:
    uv = which("uv")
    if uv is None:
        raise RuntimeError("uv is required to build the macOS app")

    subprocess.run([uv, "pip", "install", "-e", ".[mac]"], cwd=ROOT, check=True)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            "--name",
            APP_NAME,
            "--windowed",
            "--noconfirm",
            "--clean",
            "--copy-metadata",
            "streamlit",
            "--collect-data",
            "streamlit",
            "--collect-submodules",
            "verilume",
            "--add-data",
            f"{ROOT / 'src' / 'verilume' / 'app.py'}:verilume",
            "--hidden-import",
            "streamlit.runtime.scriptrunner.magic_funcs",
            str(ROOT / "launcher.py"),
        ],
        cwd=ROOT,
        check=False,
    )
    if result.returncode != 0:
        return result.returncode

    archive = _archive_app_bundle()
    print(f"Built {DIST_DIR / (APP_NAME + '.app')}")
    print(f"Built {archive}")
    return 0


def _archive_app_bundle() -> Path:
    app_path = DIST_DIR / f"{APP_NAME}.app"
    if not app_path.exists():
        raise FileNotFoundError(f"Expected app bundle at {app_path}")

    machine = platform.machine().lower() or "unknown"
    archive = DIST_DIR / f"{APP_NAME}-macOS-{machine}.zip"
    archive.unlink(missing_ok=True)

    ditto = which("ditto")
    if ditto is not None and sys.platform == "darwin":
        subprocess.run(
            [
                ditto,
                "-c",
                "-k",
                "--sequesterRsrc",
                "--keepParent",
                str(app_path),
                str(archive),
            ],
            cwd=DIST_DIR,
            check=True,
        )
        return archive

    shutil.make_archive(str(archive.with_suffix("")), "zip", root_dir=DIST_DIR, base_dir=app_path.name)
    return archive


if __name__ == "__main__":
    raise SystemExit(main())
