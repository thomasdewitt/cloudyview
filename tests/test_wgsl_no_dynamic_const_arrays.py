"""No array in the shader may be indexed by anything but a literal.

A `const` array indexed at run time is lowered by naga into an array the
driver has to hold in registers or local memory, and the cost is not
confined to the code that does it. The desert's twelve such tables
(DESERT_STONE_PALETTE and friends) raised the CITY's surface pass from 128
to 168 registers although the city compiles DESERT to false and runs none
of that code (perf, 2026-09-24, bisected to e688dca); a function-local
`var c: array<f32, 4>` in CITY-only code moved day frames by 2^-11 the day
before (city-layout). Rewriting the tables as switch accessors returned the
city to 128. Fleet rule since then: an array is indexed by literals only;
anything indexed by a run-time value is a function with a `switch` (or a
`select` chain) that returns the constants.

This scans the shader and every component for arrays declared with
`const` / `let` / `var` / `var<private>` and fails on any `name[expr]`
whose expr is not an integer literal. Name-based and scope-blind on
purpose — a false positive costs one accessor, a miss costs a register
cliff in a scene nobody was looking at.

Needs no GPU: it is a tokenizer.
"""

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
# The renderer core and every city component. The overlay shaders (hud,
# bird, dart) are separate
# small pipelines whose full-screen-triangle tables are the one idiom this
# rule has no quarrel with, so they are not scanned.
SOURCES = sorted((REPO / "cloudyview" / "soar" / "shader").rglob("*.wgsl"))

# `const NAME = array<`, `const NAME: array<...> =`, `var NAME: array<`,
# `var<private> NAME: array<`, `let NAME = array<`, `var NAME = array<`.
DECL = re.compile(
    r"\b(?:const|let|var(?:<[^>]*>)?)\s+(\w+)\s*"
    r"(?::\s*array\s*<|=\s*array\s*<)")
LITERAL = re.compile(r"^\s*\d+[iu]?\s*$")


def dynamic_array_indexing(source):
    """(line, name, index) for each array indexed by a non-literal."""
    src = re.sub(r"//[^\n]*", "", source)
    names = set(DECL.findall(src))
    found = []
    for name in names:
        for m in re.finditer(r"\b" + re.escape(name) + r"\s*\[", src):
            # The bracket's contents, nesting-aware.
            i, depth = m.end(), 1
            while depth and i < len(src):
                depth += {"[": 1, "]": -1}.get(src[i], 0)
                i += 1
            index = src[m.end():i - 1]
            if not LITERAL.match(index):
                line = src[:m.start()].count("\n") + 1
                found.append((line, name, " ".join(index.split())[:60]))
    return sorted(found)


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_no_array_is_indexed_at_run_time(path):
    hits = dynamic_array_indexing(path.read_text())
    assert not hits, (
        f"{path.name}: arrays indexed by a run-time value — write a switch "
        f"accessor that returns the constants instead (see this test's "
        f"docstring for why it matters beyond the code that does it): "
        f"{hits}")


def test_the_scanner_catches_what_moved_the_city():
    src = """
        const DESERT_STONE_PALETTE = array<vec3<f32>, 2>(vec3<f32>(0.1),
                                                         vec3<f32>(0.2));
        fn f(i: i32) -> vec3<f32> { return DESERT_STONE_PALETTE[i]; }
    """
    assert dynamic_array_indexing(src)
    assert dynamic_array_indexing(
        "fn g(k: i32) -> f32 { var c: array<f32, 4>; return c[k + 1]; }")
    assert dynamic_array_indexing(
        "var<private> t: array<u32, 3>; fn h(k: u32) -> u32 { return t[k]; }")


def test_the_scanner_allows_literals_and_accessors():
    assert not dynamic_array_indexing("""
        const CELL = array<f32, 2>(0.03125, 0.125);
        fn a() -> f32 { return CELL[0] + CELL[1u]; }
        fn b(i: i32) -> f32 { switch i { case 0: { return 0.1; }
                                         default: { return 0.2; } } }
        fn c(v: vec4<f32>, i: i32) -> f32 { return v[i]; }
    """)
