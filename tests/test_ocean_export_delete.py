"""tools/export_web_assets.remove_tile: the re-export deletes the old tile by
the names its meta.json gives, so the names are data driving a delete. A
name that is not a bare file in the tile's own directory refuses the whole
delete, before anything is removed (the fleet lost a repo to a delete that
followed a path out of its directory, 2026-09-24)."""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

from export_web_assets import remove_tile                      # noqa: E402


def _tile(tmp_path, files):
    """A tile dir with two real levels, a sentinel beside it and one
    elsewhere, and a meta naming `files`."""
    out = tmp_path / "parent" / "ocean"
    out.mkdir(parents=True)
    for name in ("mip0.bin.gz", "mip1.bin.gz"):
        (out / name).write_bytes(b"level")
    (tmp_path / "parent" / "sentinel").write_text("keep")
    (tmp_path / "abs_sentinel").write_text("keep")
    (out / "meta.json").write_text(json.dumps({"mips": 2, "files": files}))
    return out


def test_the_named_levels_go_and_nothing_else(tmp_path):
    out = _tile(tmp_path, ["mip0.bin.gz", "mip1.bin.gz", "mip2.bin.gz"])
    remove_tile(out)
    assert sorted(p.name for p in out.iterdir()) == ["meta.json"]
    assert (tmp_path / "parent" / "sentinel").exists()


@pytest.mark.parametrize("poison", [
    "../sentinel",            # up out of the tile
    "..",                     # the parent itself
    ".",                      # the tile directory
    "",                       # the tile directory again
    "sub/mip0.bin.gz",        # into a subdirectory
    "..\\sentinel",           # a Windows separator
    "ABSOLUTE",               # an absolute path (filled in below)
    7,                        # not a string
])
def test_a_poisoned_name_refuses_the_whole_delete(tmp_path, poison):
    if poison == "ABSOLUTE":
        poison = str(tmp_path / "abs_sentinel")
    out = _tile(tmp_path, ["mip0.bin.gz", poison, "mip1.bin.gz"])
    with pytest.raises(ValueError, match="nothing was deleted"):
        remove_tile(out)
    # Refused before the first unlink: the good names survive too.
    assert (out / "mip0.bin.gz").exists() and (out / "mip1.bin.gz").exists()
    assert (tmp_path / "parent" / "sentinel").read_text() == "keep"
    assert (tmp_path / "abs_sentinel").read_text() == "keep"


def test_a_string_files_is_refused_not_iterated(tmp_path):
    """A string is iterable: unchecked, "mip0" would delete files named
    m, i, p and 0."""
    out = _tile(tmp_path, "mip0.bin.gz")
    (out / "m").write_text("keep")
    with pytest.raises(ValueError, match="not a list"):
        remove_tile(out)
    assert (out / "m").exists() and (out / "mip0.bin.gz").exists()


def test_no_meta_deletes_nothing(tmp_path):
    out = tmp_path / "ocean"
    out.mkdir()
    (out / "mip0.bin.gz").write_bytes(b"level")
    remove_tile(out)
    assert (out / "mip0.bin.gz").exists()
