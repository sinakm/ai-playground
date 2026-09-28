"""Download Blizzard's Melee map pack and install Flat128 into the SC2 Maps folder.

Extracting the pack with password `iagreetotheeula` means you accept the
Blizzard AI and Machine Learning License:
https://blzdistsc2-a.akamaihd.net/AI_AND_MACHINE_LEARNING_LICENSE.html
"""

from __future__ import annotations

import io
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "http://blzdistsc2-a.akamaihd.net/MapPacks/Melee.zip"
PASSWORD = b"iagreetotheeula"
MAP_MEMBER = "Melee/Flat128.SC2Map"
DEFAULT_SC2 = {
    "win32": r"C:\Program Files (x86)\StarCraft II",
    "darwin": "/Applications/StarCraft II",
}


def sc2_dir() -> Path:
    env = os.environ.get("SC2PATH")
    if env:
        return Path(env)
    if sys.platform in DEFAULT_SC2:
        return Path(DEFAULT_SC2[sys.platform])
    return Path.home() / "StarCraftII"


def main() -> None:
    target = sc2_dir() / "Maps" / "Flat128.SC2Map"
    if target.exists():
        print(f"already installed: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {URL}")
    data = urllib.request.urlopen(URL, timeout=60).read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        target.write_bytes(z.read(MAP_MEMBER, pwd=PASSWORD))
    print(f"installed: {target}")


if __name__ == "__main__":
    main()
