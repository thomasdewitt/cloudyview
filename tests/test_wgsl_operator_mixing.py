"""WGSL gives bitwise operators no precedence against anything else.

`a ^ b * c` is a parse error in the WGSL grammar: a bitwise expression
(^ | & << >>) may not share a parenthesis level with a different binary
operator. Tint (Chrome, Safari) enforces it and refuses the whole module;
naga (wgpu, so every Python host and test here) accepts it. So a mix passes
the whole suite's other checks and then breaks every mode in
the browser — which is how 7affb9a reached master (city_ground_mottle's
`bitcast<u32>(jw.y) ^ u32(o) * 0x297a2d39u`). This scans the shader and every
city component the way Tint's grammar reads them.

Needs no GPU: it is a tokenizer.
"""

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SOURCES = sorted(
    list((REPO / "cloudyview" / "soar" / "shader").rglob("*.wgsl"))
    + list((REPO / "web" / "soar").glob("*.wgsl")))

BITWISE = {"^", "|", "&", "<<", ">>"}
ARITH = {"*", "+", "-", "/", "%"}
TOKEN = re.compile(r"<<|>>|&&|\|\||->|[(),\[\]^|&*+\-/%]")


def operator_mixes(source):
    """(line, statement) for each bitwise op sharing a level with another op.

    Statements are split at ; { }, commas reset a level (they separate
    arguments), and brackets open a new one. `&&`, `||` and `->` are
    their own tokens so they never read as a bitwise & or |, and a unary
    minus is counted like a binary one — conservative, never missing a mix.
    """
    src = re.sub(r"//[^\n]*", "", source)
    found = []
    for m in re.finditer(r"[^;{}]+", src):
        levels = [set()]
        for t in TOKEN.finditer(m.group(0)):
            tok = t.group(0)
            if tok in "([":
                levels.append(set())
            elif tok in ")]":
                if len(levels) > 1:
                    levels.pop()
            elif tok == ",":
                levels[-1] = set()
            elif tok in BITWISE or tok in ARITH:
                cur = levels[-1]
                cur.add(tok)
                bits = cur & BITWISE
                if bits and (cur & ARITH or len(bits) > 1):
                    line = src[:m.start() + t.start()].count("\n") + 1
                    found.append((line, " ".join(m.group(0).split())[:120]))
                    levels[-1] = set()
    return found


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_no_bitwise_operator_shares_a_level_with_another(path):
    mixes = operator_mixes(path.read_text())
    assert not mixes, (
        f"{path.name}: WGSL needs parentheses around these (Tint refuses "
        f"the module; naga does not notice): {mixes}")


def test_the_scanner_catches_the_bug_that_shipped():
    assert operator_mixes("let a = b ^ u32(o) * 0x297a2d39u;")
    assert operator_mixes("let a = x & 3u << 2u;")
    assert not operator_mixes("let a = b ^ (u32(o) * 0x297a2d39u);")
    assert not operator_mixes("let a = f(b ^ c, d * e);")
    assert not operator_mixes("if (a && b || c) { let d = e ^ f; }")
