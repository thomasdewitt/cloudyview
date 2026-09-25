"""raymarch.wgsl's surface sections strip cleanly, in both hosts alike.

Each surface's code (the sea, the city, the desert) is fenced in `// @section`
blocks with one `// @stub` twin that re-declares, with neutral bodies, the
symbols shared code calls into it; a module keeps its own surface's sections
and the other surfaces' stubs (soar_host.strip_sections, renderer.js
stripSections). What would go wrong silently, or only on a GPU, is pinned
here without one:

- stripping for any surface leaves no reference to a symbol only a blanked
  block defined (the Tint gate compiles every variant too; this names the
  symbol);
- every stub function has its section's signature, token for token;
- the two hosts strip to the same bytes;
- the marker grammar's failures are loud, and an unmarked (older) shader
  passes through untouched.

The node half skips without node.
"""

import json
import re
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

from cloudyview import soar_host
from cloudyview.soar_host import SECTION_OF_SURFACE, strip_sections

REPO = Path(__file__).resolve().parents[1]
RENDERER_JS = REPO / "web" / "soar" / "renderer.js"
SOURCE = soar_host.read_shader()

DEF = re.compile(r"^(?:fn|struct|const|override|alias|var(?:<[^>]*>)?)\s+"
                 r"([A-Za-z_]\w*)")
# An identifier in use: not a member access (u.desert_tile is the uniform
# field, not the desert section's desert_tile()) and not the name in a
# `name: type` declaration (a struct field, a parameter).
IDENT = re.compile(r"(?<![.\w])[A-Za-z_]\w*(?!\w)(?!\s*:(?!:))")


def module_defs(text):
    """Names defined at module scope (unindented declarations)."""
    return {m.group(1) for l in text.split("\n") for m in [DEF.match(l)] if m}


def referenced(text):
    return {w for l in text.split("\n")
            for w in IDENT.findall(l.split("//")[0])}


def blocks(text, kind, name):
    """The bodies of every `kind` (section|stub) block of `name`."""
    end = "end" if kind == "section" else "end-stub"
    pat = re.compile(rf"^// @{kind} {name}\s*$(.*?)^// @{end} {name}\s*$",
                     re.S | re.M)
    return [m.group(1) for m in pat.finditer(text)]


def signatures(text):
    """{fn name: its signature, whitespace-normalized}."""
    out = {}
    for m in re.finditer(r"^fn\s+(\w+)\s*\((.*?)\)\s*(->\s*[^{]+)?\{",
                         text, re.S | re.M):
        out[m.group(1)] = " ".join(m.group(0).split())
    return out


@pytest.mark.parametrize("surface", soar_host.SURFACES)
def test_stripping_leaves_no_dangling_symbol(surface):
    kept = strip_sections(SOURCE, surface)
    only_blanked = module_defs(SOURCE) - module_defs(kept)
    dangling = referenced(kept) & only_blanked
    assert not dangling, (
        f"the {surface} module references symbols only a stripped block "
        f"defines — give the owning section's stub a twin of each: "
        f"{sorted(dangling)}")


@pytest.mark.parametrize("name", sorted(set(SECTION_OF_SURFACE.values())))
def test_stubs_have_their_sections_signatures(name):
    real = {}
    for body in blocks(SOURCE, "section", name):
        real.update(signatures(body))
    (stub,) = blocks(SOURCE, "stub", name)
    stubs = signatures(stub)
    assert stubs, f"{name}'s stub declares no function"
    for fn, sig in stubs.items():
        assert fn in real, f"{name}'s stub declares {fn}, which no {name} " \
                           f"section defines"
        assert sig == real[fn], f"{fn}: stub {sig!r} != section {real[fn]!r}"
    # A stub declares only functions (a constant would be a second copy of
    # a number that must then be kept equal by hand).
    assert module_defs(stub) == set(stubs), (
        f"{name}'s stub declares more than functions: "
        f"{sorted(module_defs(stub) - set(stubs))}")


def test_each_module_keeps_one_definition_of_each_stubbed_name():
    for surface in soar_host.SURFACES:
        kept = strip_sections(SOURCE, surface)
        names = [m.group(1) for l in kept.split("\n")
                 for m in [DEF.match(l)] if m]
        dupes = sorted({n for n in names if names.count(n) > 1})
        assert not dupes, f"{surface}: defined twice: {dupes}"


def test_line_numbers_survive_stripping():
    for surface in soar_host.SURFACES:
        assert (strip_sections(SOURCE, surface).count("\n")
                == SOURCE.count("\n"))


# --- the grammar --------------------------------------------------------------

def _marked(extra=""):
    parts = []
    for n in ("sea", "city", "desert"):
        parts += [f"// @section {n}", f"fn {n}_real() {{}}", f"// @end {n}",
                  f"// @stub {n}", f"fn {n}_stub() {{}}", f"// @end-stub {n}"]
    return "\n".join(parts) + extra


def test_unmarked_source_passes_through():
    src = "fn a() {}\nconst B: f32 = 1.0;\n"
    for s in soar_host.SURFACES:
        assert strip_sections(src, s) == src


def test_marked_source_keeps_its_own_section_and_the_others_stubs():
    out = strip_sections(_marked(), "city")
    assert "city_real" in out and "city_stub" not in out
    for other in ("sea", "desert"):
        assert f"{other}_stub" in out and f"{other}_real" not in out
    assert "sea_real" in strip_sections(_marked(), "ocean")


@pytest.mark.parametrize("broken", [
    _marked().replace("// @end-stub desert", ""),              # never closed
    _marked().replace("// @stub sea", "// @stub sea\n// @section sea"),  # nest
    _marked().replace("// @end city", "// @end desert"),        # mismatched
    _marked("\n// @stub city\n// @end-stub city"),              # two stubs
    _marked().replace("// @section sea\nfn sea_real() {}\n// @end sea", ""),
    _marked("\n// @section moon\n// @end moon"),                # unknown
])
def test_malformed_markers_are_refused(broken):
    with pytest.raises(RuntimeError):
        strip_sections(broken, "ocean")


# --- the two hosts ---------------------------------------------------------------

@pytest.fixture(scope="module")
def js_stripped():
    if shutil.which("node") is None:
        pytest.skip("needs node")
    script = textwrap.dedent("""
        import { stripSections } from "%s";
        import fs from "fs";
        const src = fs.readFileSync(%s, "utf8");
        const out = {};
        for (const s of ["sea", "city", "desert"]) out[s] = stripSections(src, s);
        console.log(JSON.stringify(out));
    """) % (RENDERER_JS.as_uri(), json.dumps(str(soar_host.SHADER_PATH)))
    out = subprocess.run(["node", "--input-type=module", "-e", script],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def test_hosts_strip_to_the_same_bytes(js_stripped):
    for surface, section in SECTION_OF_SURFACE.items():
        assert js_stripped[section] == strip_sections(SOURCE, surface), surface
