"""Contact sheets.

Judging forty images by opening forty files is not iteration. A sheet puts a whole
sweep on one page with the entry name and the seed under each cell, so the next
sweep can be decided at a glance.

Pure image work, so it needs no model and is fully testable.
"""

# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026, Lucas Jahier - Stratorys

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

BACKGROUND = (18, 18, 20)
FOREGROUND = (232, 232, 236)
LABEL_BAND = 26
PADDING = 10
NAME = re.compile(r"^(?P<entry>.+?)_(?P<fingerprint>[0-9a-f]{6,})_seed(?P<seed>\d+)$")


@dataclass(frozen=True)
class Cell:
    path: Path
    label: str
    entry: str = ""
    seed: int = 0


def parse_name(path: Path) -> Cell:
    """Recover the entry name and seed from a fingerprinted filename."""
    match = NAME.match(path.stem)
    if not match:
        return Cell(path, path.stem)
    entry, seed = match.group("entry"), int(match.group("seed"))
    return Cell(path, f"{entry}  seed {seed}", entry, seed)


def collect(directory: Path) -> list[Cell]:
    """Every image in a run directory, ordered by entry then seed."""
    cells = [
        parse_name(path)
        for path in sorted(directory.glob("*.png"))
        if not path.name.startswith("sheet")
    ]
    return sorted(cells, key=lambda cell: (cell.entry, cell.seed, cell.path.name))


def _font(size: int) -> Any:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def build(cells: list[Cell], columns: int | None = None, cell_size: int = 320) -> Image.Image:
    """One page. Cells keep their aspect ratio inside a square box."""
    if not cells:
        raise ValueError("no images to put on a sheet")
    columns = columns or max(1, min(len(cells), round(math.sqrt(len(cells) * 1.4))))
    rows = math.ceil(len(cells) / columns)

    step_x = cell_size + PADDING
    step_y = cell_size + LABEL_BAND + PADDING
    sheet = Image.new("RGB", (columns * step_x + PADDING, rows * step_y + PADDING), BACKGROUND)
    draw = ImageDraw.Draw(sheet)
    font = _font(15)

    for index, cell in enumerate(cells):
        column, row = index % columns, index // columns
        left = PADDING + column * step_x
        top = PADDING + row * step_y
        with Image.open(cell.path) as opened:
            thumbnail = opened.convert("RGB")
            thumbnail.thumbnail((cell_size, cell_size), Image.Resampling.LANCZOS)
        sheet.paste(
            thumbnail,
            (left + (cell_size - thumbnail.width) // 2, top + (cell_size - thumbnail.height) // 2),
        )
        draw.text((left + 2, top + cell_size + 5), cell.label[:48], font=font, fill=FOREGROUND)
    return sheet


def paginate(cells: list[Cell], per_sheet: int) -> list[list[Cell]]:
    if per_sheet < 1:
        raise ValueError("per_sheet must be at least 1")
    return [cells[start : start + per_sheet] for start in range(0, len(cells), per_sheet)]
