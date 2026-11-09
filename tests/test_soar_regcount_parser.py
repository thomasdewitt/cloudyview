"""tools/soar_regcount.py's reader of the NVIDIA driver's shader disk cache,
pinned on the CPU.

The format is undocumented: a cache record is (u32 n, u32 size, a zstd frame
of n - 4 bytes that decompresses to size bytes), the frame an "NVuc"
program container whose table holds the code section and a register entry.
The register entry's reading was cross-checked against the driver's own
statistics (VK_KHR_pipeline_executable_properties) at 95, 158, 166 and 168
registers on 2026-11-09, through the Vulkan crate the tool replaced. The
fixture is one record as driver 595.80 wrote it on the RTX 5080: the sun-tau
bake (fs_bake_light), 26 registers, 135 instructions, no spills. A driver
update that changes the format fails here, not in someone's measurement.

A compute program keeps its count elsewhere: the u32 eight bytes before
its code section (checked against the driver's statistics at 16, 128 and
185 registers, 2026-11-09, for the surface_cs experiment's cs_surface).
Its fixture is nvidia_cache_record_cs_tiny.wgsl (a 512-invocation kernel,
16 registers, 119 instructions) as the same driver wrote it.
"""

import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import soar_regcount as rc  # noqa: E402

FIXTURE = Path(__file__).parent / "data" / "nvidia_cache_record_fs_bake_light.bin"
FIXTURE_CS = Path(__file__).parent / "data" / "nvidia_cache_record_cs_tiny.bin"

pytestmark = pytest.mark.skipif(shutil.which("zstd") is None,
                                reason="needs the zstd command")


def test_the_bake_record_parses_to_its_known_program():
    progs = list(rc.cache_frames(FIXTURE.read_bytes()))
    assert len(progs) == 1
    s = rc.program_stats(progs[0])
    assert (s["registers"], s["ldl"], s["stl"], s["instructions"]) == (26, 0, 0, 135)
    assert rc.opcodes(rc.program_code(progs[0]))[rc.OP_IPA] > 0   # a fragment program
    with pytest.raises(SystemExit, match="no register count"):
        rc.program_stats(progs[0], compute=True)                      # 0 there


def test_the_compute_record_parses_to_its_known_program():
    progs = list(rc.cache_frames(FIXTURE_CS.read_bytes()))
    assert len(progs) == 1
    assert rc.opcodes(rc.program_code(progs[0]))[rc.OP_IPA] == 0
    s = rc.program_stats(progs[0], compute=True)
    assert (s["registers"], s["ldl"], s["stl"], s["instructions"]) == (16, 0, 0, 119)
    # Read as a fragment program it has no register entry: refused, not
    # misread.
    with pytest.raises(SystemExit, match="no register count"):
        rc.program_stats(progs[0])


def test_a_record_that_does_not_decompress_to_its_size_refuses():
    blob = bytearray(FIXTURE.read_bytes())
    blob[4:8] = (int.from_bytes(blob[4:8], "little") + 1).to_bytes(4, "little")
    with pytest.raises(SystemExit, match="did not decompress"):
        list(rc.cache_frames(bytes(blob)))


def test_a_program_without_its_register_entry_refuses():
    prog = list(rc.cache_frames(FIXTURE.read_bytes()))[0]
    table = rc.sections(prog)
    # Blank the register entry's 0xff marker (its sixth word).
    k = next(i for i, (t, size, o, w4, w5) in enumerate(table)
             if t in (1, 2) and size == 0 and o == 0 and w5 == 0xff)
    off = 0x28 + 32 * k + 20
    broken = prog[:off] + bytes(4) + prog[off + 4:]
    with pytest.raises(SystemExit, match="no register count"):
        rc.program_stats(broken)


@pytest.mark.parametrize("registers,warps", [
    (26, 48), (94, 20), (96, 20), (104, 16), (128, 16), (136, 12),
    (157, 12), (168, 12), (176, 8), (255, 8)])
def test_warps_are_bounded_per_sub_partition(registers, warps):
    # Four 16K-register sub-partitions, 256 registers a warp's allocation
    # unit, 12 warps each: e.g. 96 registers allow 20 warps, not 64K/3072=21.
    assert rc.warps_per_sm(registers) == warps
