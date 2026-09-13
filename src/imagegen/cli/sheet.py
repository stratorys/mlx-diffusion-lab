"""Build a contact sheet for a run directory.

    imagegen-sheet runs/beach
    imagegen-sheet runs/beach --columns 4 --cell 400
"""

# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026, Lucas Jahier - Stratorys

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from imagegen import sheet as sheets
from imagegen.storage import atomic_image


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--out", type=Path, default=None, help="defaults to sheet.png in the directory")
    parser.add_argument("--columns", type=int, default=None)
    parser.add_argument("--cell", type=int, default=320, help="longest side of one cell, in pixels")
    parser.add_argument("--per-sheet", type=int, default=64, help="split into pages beyond this many")
    return parser.parse_args(argv), parser


def main(argv: list[str] | None = None) -> int:
    args, parser = parse_args(argv)
    if not args.directory.is_dir():
        parser.error(f"not a directory: {args.directory}")

    cells = sheets.collect(args.directory)
    if not cells:
        print(f"no images in {args.directory}", file=sys.stderr)
        return 1

    pages = sheets.paginate(cells, args.per_sheet)
    base = args.out or args.directory / "sheet.png"
    for number, page in enumerate(pages, start=1):
        target = base if len(pages) == 1 else base.with_name(f"{base.stem}_{number:02d}{base.suffix}")
        atomic_image(target, sheets.build(page, args.columns, args.cell), format="PNG")
        print(f"{target}  {len(page)} image(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
