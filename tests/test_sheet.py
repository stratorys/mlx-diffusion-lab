# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can obtain
# one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026, Lucas Jahier - Stratorys

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from imagegen import sheet as sheets
from imagegen.cli import sheet as cli


def place(directory: Path, name: str, size=(64, 96), colour=(10, 120, 200)) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    Image.new("RGB", size, colour).save(path)
    return path


def test_a_fingerprinted_name_yields_its_entry_and_seed():
    cell = sheets.parse_name(Path("harbour_0742e6399a_seed12.png"))
    assert (cell.entry, cell.seed, cell.label) == ("harbour", 12, "harbour  seed 12")


def test_an_entry_name_may_contain_underscores():
    cell = sheets.parse_name(Path("night_market_02_bc09b61e00_seed3.png"))
    assert cell.entry == "night_market_02" and cell.seed == 3


def test_an_unrecognised_name_still_gets_a_label():
    cell = sheets.parse_name(Path("whatever.png"))
    assert cell.label == "whatever" and cell.entry == ""


def test_cells_are_ordered_by_entry_then_seed(tmp_path):
    for name in ("b_aaaaaaaaaa_seed2.png", "a_aaaaaaaaaa_seed10.png", "a_aaaaaaaaaa_seed2.png"):
        place(tmp_path, name)
    assert [(c.entry, c.seed) for c in sheets.collect(tmp_path)] == [("a", 2), ("a", 10), ("b", 2)]


def test_an_existing_sheet_is_not_pulled_into_the_next_one(tmp_path):
    place(tmp_path, "a_aaaaaaaaaa_seed1.png")
    place(tmp_path, "sheet.png")
    assert [cell.path.name for cell in sheets.collect(tmp_path)] == ["a_aaaaaaaaaa_seed1.png"]


def test_the_grid_holds_every_cell(tmp_path):
    cells = [sheets.parse_name(place(tmp_path, f"a_aaaaaaaaaa_seed{n}.png")) for n in range(5)]
    image = sheets.build(cells, columns=2, cell_size=100)
    # two columns, three rows, each row a cell plus its label band and padding
    assert image.width == 2 * (100 + sheets.PADDING) + sheets.PADDING
    assert image.height == 3 * (100 + sheets.LABEL_BAND + sheets.PADDING) + sheets.PADDING


def test_a_tall_image_keeps_its_aspect_ratio(tmp_path):
    cell = sheets.parse_name(place(tmp_path, "a_aaaaaaaaaa_seed1.png", size=(50, 200)))
    image = sheets.build([cell], columns=1, cell_size=100)
    assert image.size == (100 + 2 * sheets.PADDING, 100 + sheets.LABEL_BAND + 2 * sheets.PADDING)


def test_an_empty_sheet_is_an_error():
    with pytest.raises(ValueError, match="no images"):
        sheets.build([])


def test_pagination_splits_evenly(tmp_path):
    cells = [sheets.parse_name(place(tmp_path, f"a_aaaaaaaaaa_seed{n}.png")) for n in range(7)]
    pages = sheets.paginate(cells, 3)
    assert [len(page) for page in pages] == [3, 3, 1]


def test_the_cli_writes_one_sheet(tmp_path, capsys):
    for n in range(4):
        place(tmp_path, f"a_aaaaaaaaaa_seed{n}.png")
    assert cli.main([str(tmp_path), "--cell", "80"]) == 0
    capsys.readouterr()
    assert (tmp_path / "sheet.png").is_file()


def test_the_cli_paginates_when_asked(tmp_path, capsys):
    for n in range(5):
        place(tmp_path, f"a_aaaaaaaaaa_seed{n}.png")
    assert cli.main([str(tmp_path), "--cell", "80", "--per-sheet", "2"]) == 0
    capsys.readouterr()
    assert sorted(p.name for p in tmp_path.glob("sheet_*.png")) == [
        "sheet_01.png", "sheet_02.png", "sheet_03.png"
    ]


def test_an_empty_directory_reports_rather_than_crashes(tmp_path, capsys):
    assert cli.main([str(tmp_path)]) == 1
    assert "no images" in capsys.readouterr().err
