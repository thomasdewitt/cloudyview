#!/usr/bin/env python
"""What the GPU driver compiled a soar shader into: registers, spills, size.

fs_main's speed on this renderer is set as much by how many registers the
driver allocates as by what executes (raymarch.wgsl, "FS_MAIN IS ONE
COMPILE"): code that never runs still costs occupancy, and a kernel over the
driver's register budget runs few warps at once. wgpu does not expose that
number. This reads it off the binary wgpu itself built: it constructs a
SoarRenderer on the shader (which specializes it exactly as the app does)
and builds the entry point's pipeline through wgpu, in a child process
whose NVIDIA shader disk cache is pointed at a private directory; the
driver writes its compiled fragment program there, and this parses it.

    tools/gpulock uv run python tools/soar_regcount.py        # fs_main, day, Low
    tools/gpulock uv run python tools/soar_regcount.py --surface city \\
        --entry fs_surface_refine
    tools/gpulock uv run python tools/soar_regcount.py --shader my_variant.wgsl
    tools/gpulock uv run python tools/soar_regcount.py --a git:HEAD --b tree
    tools/gpulock uv run python tools/soar_regcount.py --surface city \\
        --entry fs_surface --sass /tmp/fs_surface.sass       # and disassemble

Why the binary and not a compile of our own (2026-11-09): the tool before
this one compiled the WGSL itself (naga 26, its own SPIR-V options) and read
the driver's pipeline statistics through Vulkan. wgpu-native 29 builds a
different program from the same source (its own naga, and wgpu-hal's
own SPIR-V writer options), and on the city's first stage the two
disagreed in both directions: a graffiti-weathering variant it read at 128
ran at 157, and a facade-zones variant it read at 161 ran at 128. Register
attributions in commits before that date came from it.

What the numbers mean on this card (NVIDIA, sm_120: 64K registers an SM,
48 warps at most, registers allocated 8 a thread). The driver chooses each
fragment program's register target itself, by its own spill-cost
heuristic, and a stage whose natural demand sits near 128 flips between
two builds on changes that do no work: 128 registers with spills (LDL/STL,
local memory), 16 warps an SM, or ~160 with none, 12 warps — on the city's
first stage, 10-14% apart. The spill counts are static instruction counts,
not traffic. Apple's compiler makes its own choices: a count here is a
statement about this card. Time what it suggests, here and on the Mac.

The binary is read from the driver's on-disk cache, an undocumented
format; every step checks what it expects (frame sizes, the program
container's magic, exactly one new fragment program, a code section that
decodes) and raises rather than guess. The fragment program is told from
the vertex program and from the renderer's own eager pipelines by
difference: one child builds the renderer alone, another the renderer and
the entry, and the one new program (by its code) that interpolates an
input (IPA) is the entry's. Run it under tools/gpulock: it draws nothing,
but each child opens the device.
"""

import argparse
import collections
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# The lighting method per tier (web/soar/constants.js QUALITY_PRESETS):
# (light cache, sky probe).
TIERS = {"max": (False, True), "high": (True, True), "medium": (True, False),
         "low": (True, False), "minimal": (True, False)}
ENTRIES = ("fs_main", "fs_bake_light", "fs_surface", "fs_surface_refine",
           "fs_surface_street")

# sm_120 (RTX 50): what a register count allows an SM to keep in flight.
SM_REGISTERS = 65536
SM_MAX_WARPS = 48
REG_GRANULE = 8
# SM120 opcodes, the low 12 bits of an instruction's first 64-bit word
# (checked against nvdisasm -b SM120 on these binaries).
OP_LDL, OP_STL, OP_IPA = 0x983, 0x387, 0x326
# The driver's program container and the sections of it read here.
NVUC_MAGIC = b"NVDANVVMNVuc"
SEC_CODE = 0x1
ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


def source_for(spec: str) -> str:
    """'tree', 'git:<rev>', or a path — soar_shader_ab.py's spec, without its
    test-suite imports, so this runs in any venv (no pytest)."""
    if spec == "tree":
        return (REPO / "cloudyview" / "soar" / "raymarch.wgsl").read_text()
    if spec.startswith("git:"):
        out = subprocess.run(
            ["git", "show", f"{spec[4:]}:cloudyview/soar/raymarch.wgsl"],
            cwd=REPO, capture_output=True, text=True)
        if out.returncode != 0 or "fn " not in out.stdout:
            raise SystemExit(f"git show failed for {spec!r}: {out.stderr.strip()}")
        return out.stdout
    return Path(spec).read_text()


# --- the child: build one pipeline exactly as the renderer does ---------------

def child(job: dict):
    # The device is opened here; the parent checked the lock, and so does
    # the child (a descendant of the same tools/gpulock).
    from gpulock_guard import require_gpulock
    require_gpulock("tools/soar_regcount.py")
    import wgpu
    from cloudyview.soar_host import SoarRenderer, ViewState
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = adapter.request_device_sync()
    r = SoarRenderer(device=device, periodic=job["periodic"],
                     nested=job["nested"], surface=job["surface"],
                     shader_source=Path(job["source"]).read_text())
    entry = job["entry"]
    if entry is None:
        return                      # the renderer's eager pipelines alone
    light_cache, sky_probe = TIERS[job["tier"]]
    view = ViewState(camera_position=(0.0, 0.0, 1000.0), azimuth=0.0,
                     elevation=0.0, fov=60.0, output_size=(2560, 1440),
                     render_size=(768, 432), light_cache=light_cache,
                     sky_probe=sky_probe)
    if entry == "fs_main":
        r._ray_pipeline_for(view)
    elif entry == "fs_bake_light":
        r._bake_pipeline()
    else:
        stages = r._stage_entries(view)
        if entry not in stages:
            raise SystemExit(f"{entry} is not a stage of {job['surface']} at "
                             f"{job['tier']} (its stages: {stages})")
        r._stage_pipeline_for(entry, stages.index(entry) == 0, view)


def build_programs(job: dict, cache: Path) -> dict:
    """{sha256 of its code: program bytes} of every compiled program the
    driver cached while a child ran `job`. Keyed on the code alone: the
    containers of one program differ from run to run."""
    cache.mkdir(parents=True)
    env = {**os.environ, "__GL_SHADER_DISK_CACHE": "1",
           "__GL_SHADER_DISK_CACHE_PATH": str(cache),
           "__GL_SHADER_DISK_CACHE_SKIP_CLEANUP": "1",
           "__GL_SHADER_DISK_CACHE_SIZE": str(10 ** 10)}
    out = subprocess.run([sys.executable, __file__, "--child", json.dumps(job)],
                         env=env, capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"pipeline build failed ({job['entry']}):\n"
                         f"{out.stderr[-3000:]}")
    bins = sorted(cache.rglob("*.bin"))
    if not bins:
        raise SystemExit(f"the driver wrote no disk cache under {cache}: is "
                         "this the NVIDIA driver, with its shader cache on?")
    programs = {}
    for b in bins:
        for prog in cache_frames(b.read_bytes()):
            programs[hashlib.sha256(program_code(prog)).hexdigest()] = prog
    return programs


def cache_frames(blob: bytes):
    """The driver's cache records: (u32 n, u32 size, a zstd frame of n - 4
    bytes that decompresses to size); yields the compiled programs (the
    NVuc containers) among them."""
    if shutil.which("zstd") is None:
        raise SystemExit("needs the zstd command (dnf install zstd)")
    i = 0
    while (j := blob.find(ZSTD_MAGIC, i)) >= 0:
        i = j + 4
        n, size = struct.unpack_from("<II", blob, j - 8)
        if j + n - 4 > len(blob) or n < 12 or size == 0:
            continue                # the magic inside compressed data
        out = subprocess.run(["zstd", "-d", "-c", "-q"],
                             input=blob[j:j + n - 4], capture_output=True)
        if out.returncode != 0 or len(out.stdout) != size:
            raise SystemExit(f"cache record at {j} did not decompress to its "
                             f"stated {size} bytes: the cache format changed")
        i = j + n - 4
        if out.stdout.startswith(NVUC_MAGIC):
            yield out.stdout


def sections(prog: bytes) -> list:
    """The container's table: 32-byte entries (type, size, offset, _, w4,
    w5, ...) from 0x28 up to where the code (type SEC_CODE, non-empty)
    begins."""
    table, off, code_at = [], 0x28, None
    while code_at is None or off < code_at:
        if off + 32 > len(prog):
            raise SystemExit("no code section in a compiled program")
        t, size, o, _hi, w4, w5 = struct.unpack_from("<6I", prog, off)
        table.append((t, size, o, w4, w5))
        if t == SEC_CODE and size > 0:
            code_at = o
        off += 32
    return table


def program_code(prog: bytes) -> bytes:
    """The program's instructions (16 bytes each): its one code section,
    which opens with 8 zero bytes."""
    code = [(size, o) for t, size, o, _, _ in sections(prog)
            if t == SEC_CODE and size > 0]
    if len(code) != 1:
        raise SystemExit(f"expected one code section, found {len(code)}")
    size, o = code[0]
    if prog[o:o + 8] != bytes(8) or size % 16:
        raise SystemExit("the code section is not laid out as expected")
    return prog[o + 8:o + 8 + 16 * ((size - 8) // 16)]


def opcodes(code: bytes) -> collections.Counter:
    return collections.Counter(struct.unpack_from("<Q", code, k)[0] & 0xfff
                               for k in range(0, len(code), 16))


def program_stats(prog: bytes) -> dict:
    """A fragment program's register count, static spill instructions,
    size, and the warps its registers allow. The count is the table entry
    with no data (size and offset 0) and 0xff in its sixth word, of type 1
    or 2, its fifth word (checked against the driver's own statistics,
    VK_KHR_pipeline_executable_properties, at 95, 158, 166 and 168)."""
    code = program_code(prog)
    ops = opcodes(code)
    regs = [w4 for t, size, o, w4, w5 in sections(prog)
            if t in (1, 2) and size == 0 and o == 0 and w5 == 0xff]
    if len(regs) != 1 or not 0 < regs[0] <= 255:
        raise SystemExit(f"no register count where one was expected ({regs})")
    granted = -(-regs[0] // REG_GRANULE) * REG_GRANULE
    return {"registers": regs[0], "ldl": ops[OP_LDL], "stl": ops[OP_STL],
            "instructions": len(code) // 16, "binary": len(code),
            "warps": min(SM_MAX_WARPS, SM_REGISTERS // (granted * 32)),
            "code": code}


def measure(spec: str, args, work: Path) -> dict:
    """The entry's fragment program for `spec`: what a renderer with the
    entry built cached, less what the renderer alone did (its eager
    pipelines, which may come from the same module)."""
    k = len(list(work.glob("src*.wgsl")))
    src = work / f"src{k}.wgsl"
    src.write_text(source_for(spec))
    job = {"source": str(src), "entry": None, "surface": args.surface,
           "tier": args.tier, "periodic": args.periodic == "on",
           "nested": args.nested}
    baseline = build_programs(job, work / f"cache{k}-renderer")
    progs = build_programs({**job, "entry": args.entry},
                           work / f"cache{k}-entry")
    new = [p for h, p in progs.items() if h not in baseline]
    frag = [p for p in new if opcodes(program_code(p))[OP_IPA] > 0]
    if len(frag) != 1:
        raise SystemExit(f"expected one new fragment program for "
                         f"{args.entry}, found {len(frag)} (of {len(new)} new)")
    return program_stats(frag[0])


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--child":
        return child(json.loads(sys.argv[2]))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shader", default="tree",
                    help="tree, git:REV, or a .wgsl path (default: tree)")
    ap.add_argument("--a", default=None, help="compare: baseline revision")
    ap.add_argument("--b", default=None, help="compare: candidate revision")
    ap.add_argument("--entry", default="fs_main", choices=ENTRIES)
    ap.add_argument("--surface", default="ocean",
                    choices=("ocean", "city"))
    ap.add_argument("--tier", default="low", choices=sorted(TIERS))
    ap.add_argument("--periodic", default="on", choices=("on", "off"))
    ap.add_argument("--nested", action="store_true")
    ap.add_argument("--sass", type=Path, default=None,
                    help="also disassemble each program to this path (one per "
                         "spec, suffixed); needs nvdisasm on PATH or $NVDISASM")
    # It opens a GPU device: refuse outside tools/gpulock (gpulock_guard),
    # before the parse: -h/--help still answers unlocked.
    from gpulock_guard import require_gpulock
    require_gpulock("tools/soar_regcount.py")
    args = ap.parse_args()
    nvdisasm = None
    if args.sass:
        nvdisasm = os.environ.get("NVDISASM") or shutil.which("nvdisasm")
        if not nvdisasm:
            raise SystemExit("--sass needs nvdisasm (CUDA 12.8+, which knows "
                             "SM120) on PATH or as $NVDISASM")

    specs = [args.a, args.b] if args.a and args.b else [args.shader]
    print(f"{args.entry}, surface {args.surface}, tier {args.tier}, periodic "
          f"{args.periodic}{', nested' if args.nested else ''}: the binary "
          f"wgpu built")
    with tempfile.TemporaryDirectory(prefix="soar-regcount-") as tmp:
        work = Path(tmp)
        for k, spec in enumerate(specs):
            s = measure(spec, args, work)
            print(f"  {spec:28s} registers {s['registers']:4d} "
                  f"({s['warps']:2d} warps/SM)   spills LDL {s['ldl']:4d} "
                  f"STL {s['stl']:4d}   {s['instructions']:6d} instructions "
                  f"({s['binary'] / 1024:5.0f} KB)")
            if nvdisasm:
                code = work / f"code{k}.bin"
                code.write_bytes(s["code"])
                out = args.sass.with_name(f"{args.sass.stem}.{k}{args.sass.suffix}"
                                          if len(specs) > 1 else args.sass.name)
                with open(out, "w") as f:
                    subprocess.run([nvdisasm, "-b", "SM120", str(code)],
                                   stdout=f, check=True)
                print(f"    SASS: {out}")


if __name__ == "__main__":
    main()
