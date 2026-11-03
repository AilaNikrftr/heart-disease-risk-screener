"""Download the NHANES 2015-2016 tables the model needs from the CDC.

Usage:  python -m ml.fetch_data            (saves .xpt files to ml/data/)
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

from ml.features import TABLES

DATA_DIR = Path(__file__).parent / "data"

# The CDC has moved these files before, so try the current location first.
URL_PATTERNS = [
    "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/{name}.xpt",
    "https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/{name}.XPT",
]


def download(name: str, dest: Path) -> None:
    last_error: Exception | None = None
    for pattern in URL_PATTERNS:
        url = pattern.format(name=name)
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "heart-risk-screener"})
            with urllib.request.urlopen(request, timeout=60) as response:
                body = response.read()
            # A moved file sometimes returns an HTML page with status 200.
            if body[:6].lower() in (b"<!doct", b"<html>"):
                raise ValueError(f"{url} returned HTML, not an XPT file")
            dest.write_bytes(body)
            print(f"  {name}: {len(body) / 1e6:.1f} MB from {url}")
            return
        except Exception as error:  # try the next location
            last_error = error
    raise RuntimeError(f"could not download {name}: {last_error}")


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading NHANES 2015-2016 tables to {DATA_DIR}")
    for name in TABLES:
        dest = DATA_DIR / f"{name}.xpt"
        if dest.exists():
            print(f"  {name}: already downloaded")
            continue
        download(name, dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
