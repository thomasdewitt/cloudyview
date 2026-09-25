"""No WGSL math builtin is called outside its domain unless someone said why.

pow(x, y) with x < 0, sqrt / log / inverseSqrt of a negative (or zero, for
the logs), acos / asin past [-1, 1] and normalize() of a zero vector are NaN
or undefined, and a NaN pixel in soar survives the accumulation: in the
browser it sticks for the whole hold. The one that shipped (2026-09-25) was
adscreens' `pow(1.0 - sv, 1.3)`, evaluated over the spill ring above a panel
where sv > 1 and then multiplied by 0 — which a NaN survives.

So every such call must be VISIBLY in its domain:
  - its argument is guarded where it is written: a clamp / max / abs /
    saturate (per builtin, below), or a form that cannot leave the domain
    (a sum of squares, 1 - clamp(x, 0, 1), a vector constructor with a
    nonzero literal component, ...);
  - or it is a plain name whose `let` in the same function is so guarded;
  - or the line, or the one above, carries `// @domain: <why it is safe>`.
The last is the escape hatch, and the reason is the point: it is what a
reader checks when the surrounding code changes.

Scans raymarch.wgsl's own code (the GENERATED city block is the component
files, which are scanned at their source) and the soar web shaders. Needs no
GPU.
"""

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CORE = REPO / "cloudyview" / "soar" / "raymarch.wgsl"
SOURCES = sorted(
    [CORE]
    + list((REPO / "cloudyview" / "soar" / "city" / "components").glob(
        "*.wgsl"))
    + [p for p in (REPO / "web" / "soar").glob("*.wgsl")
       if p.name != "raymarch.wgsl"])     # web/soar/raymarch.wgsl is CORE

# The builtins, and the outermost calls that visibly keep each argument in
# its domain. log / inverseSqrt want > 0, which abs cannot promise.
GUARDS = {
    "pow": ("max", "clamp", "abs", "saturate", "exp", "exp2"),
    "sqrt": ("max", "clamp", "abs", "saturate", "exp", "exp2"),
    "log": ("max", "clamp"),
    "log2": ("max", "clamp"),
    "inverseSqrt": ("max", "clamp"),
    "acos": ("clamp",),
    "asin": ("clamp",),
    "normalize": (),
}
# Calls whose result is never negative (for pow's base and sqrt).
NON_NEGATIVE = ("length", "exp", "exp2", "fract", "smoothstep", "max",
                "clamp", "abs", "saturate")
MARK_OPEN = "// >>> GENERATED CITY COMPONENTS"
MARK_CLOSE = "// <<< GENERATED CITY COMPONENTS"
NUMBER = r"-?[0-9]*\.?[0-9]+(e[-+]?[0-9]+)?"


def _args(src, i):
    """The text between the parenthesis at src[i] and its match."""
    depth = 0
    for j in range(i, len(src)):
        if src[j] == "(":
            depth += 1
        elif src[j] == ")":
            depth -= 1
            if depth == 0:
                return src[i + 1:j]
    return ""


def _split_top(a, sep):
    """Split at `sep` outside any bracket."""
    depth, out, cur = 0, [], ""
    for c in a:
        if c in "([":
            depth += 1
        elif c in ")]":
            depth -= 1
        if c == sep and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += c
    out.append(cur)
    return [t.strip() for t in out]


def _is_square(t):
    m = re.fullmatch(r"dot\((.*)\)", t)
    if m:
        parts = _split_top(m.group(1), ",")
        return len(parts) == 2 and parts[0] == parts[1]
    f = _split_top(t, "*")
    return len(f) == 2 and f[0] == f[1]


def _safe_form(fn, a):
    """Forms that cannot leave the domain, whatever their inputs."""
    if fn in ("pow", "sqrt"):
        if any(re.fullmatch(g + r"\(.*\)", a) for g in NON_NEGATIVE):
            return True
        # A sum of squares and non-negative literals.
        terms = _split_top(a, "+")
        if "-" not in a.replace("e-", "") and all(
                _is_square(t) or re.fullmatch(r"[0-9]*\.?[0-9]+", t)
                for t in terms):
            return True
        # 1 - (something in [0, 1]).
        m = re.fullmatch(r"1\.0\s*-\s*(clamp\((.*),\s*0\.0,\s*1\.0\)"
                         r"|min\((.*),\s*1\.0\))", a)
        if m and (m.group(2) is not None or re.fullmatch(
                r"(abs|max|clamp|length|saturate)\(.*\)",
                m.group(3).strip())):
            return True
    if fn == "normalize":
        # A vector constructor with a nonzero literal component.
        m = re.fullmatch(r"vec[234](<f32>)?\((.*)\)", a)
        if m and any(re.fullmatch(NUMBER, t) and float(t) != 0.0
                     for t in _split_top(m.group(2), ",")):
            return True
    return False


def _guarded(fn, a):
    return (any(re.match(g + r"\s*\(", a) for g in GUARDS[fn])
            or re.fullmatch(r"[0-9.]+(e[-+]?[0-9]+)?", a) is not None
            or _safe_form(fn, a))


def _last_let(code, pos, name):
    """The expression of the last `let name = ...;` in the function
    enclosing pos, or None."""
    starts = [m.start() for m in re.finditer(r"(?m)^fn\s", code[:pos])]
    body = code[starts[-1] if starts else 0:pos]
    best = None
    for m in re.finditer(r"\blet\s+" + re.escape(name)
                         + r"\s*(:[^=]+)?=\s*([^;]*);", body):
        best = m.group(2).strip()
    return best


def unguarded_calls(text):
    """(line, builtin, argument) for every call not visibly in its domain."""
    lines = text.split("\n")
    code = re.sub(r"//[^\n]*", lambda m: " " * len(m.group(0)), text)
    found = []
    for fn in GUARDS:
        for m in re.finditer(r"(?<![\w.])" + fn + r"\s*\(", code):
            line = code[:m.start()].count("\n")
            a = _split_top(_args(code, m.end() - 1), ",")[0]
            ok = _guarded(fn, a)
            if not ok and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", a):
                d = _last_let(code, m.start(), a)
                ok = d is not None and _guarded(fn, d)
            noted = ("@domain" in lines[line]
                     or (line > 0 and "@domain" in lines[line - 1]))
            if not ok and not noted:
                found.append((line + 1, fn, " ".join(a.split())[:80]))
    return sorted(found)


def _source_text(path):
    text = path.read_text()
    if path == CORE:
        # The components are scanned at their source; keep the line numbers.
        a, b = text.index(MARK_OPEN), text.index(MARK_CLOSE)
        text = text[:a] + "\n" * text[a:b].count("\n") + text[b:]
    return text


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_math_builtins_stay_in_their_domain(path):
    found = unguarded_calls(_source_text(path))
    assert not found, (
        f"{path.name}: calls not visibly in their domain — guard the "
        f"argument (clamp / max / ...) or say why it is safe with a "
        f"`// @domain: ...` line above: {found}")


def test_the_scanner_catches_the_bug_that_shipped():
    assert unguarded_calls(
        "fn f() { let bg = (0.18 + 0.82 * pow(1.0 - sv, 1.3)) * 0.55; }")
    assert unguarded_calls("fn f() { let a = acos(dot(u, v)); }")
    assert unguarded_calls("fn f() { let n = normalize(a - b); }")
    assert unguarded_calls("fn f() { let s = sqrt(disc); }")
    assert not unguarded_calls(
        "fn f() { let bg = pow(clamp(1.0 - sv, 0.0, 1.0), 1.3); }")
    assert not unguarded_calls("fn f() { let a = acos(clamp(c, -1.0, 1.0)); }")
    assert not unguarded_calls("fn f() { let r = sqrt(dot(v, v) + 1.0); }")
    assert not unguarded_calls("fn f() { let n = normalize(vec3<f32>(-s, 1.0)); }")
    assert not unguarded_calls(
        "fn f() { let d = max(x, 0.0);\n let s = sqrt(d); }")
    # A guarded `let` of the same name in ANOTHER function does not count.
    assert unguarded_calls(
        "fn g() { let d = max(x, 0.0); }\nfn f() { let s = sqrt(d); }")
    assert not unguarded_calls(
        "fn f() {\n    // @domain: disc > 0 on this branch\n    let s = sqrt(disc);\n}")
