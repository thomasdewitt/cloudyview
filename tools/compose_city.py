#!/usr/bin/env python
"""Splice the city component files into raymarch.wgsl.

The night city grows by components: WGSL files under
cloudyview/soar/city/components/, each owning one kind of thing and listed in
registry.json there. This tool concatenates the enabled ones, generates the
five hook dispatchers the core calls (see the hook comment block in
raymarch.wgsl), replaces the GENERATED block between the markers, and
compile-validates the result under wgpu for both CITY specializations before
writing a byte.

Run it after editing any component or the registry:

    uv run python tools/compose_city.py            # splice + validate + write
    uv run python tools/compose_city.py --check    # validate only, no write

Rules it enforces (the same ones components/SPEC.md states):
  * a component may only define module-scope symbols named cc_<name>_* —
    collisions with the core or with other components are refused;
  * every hook named in the registry must exist in the component's file with
    the exact dispatcher-compatible signature (checked by compiling a probe);
  * kind_base values are >= 100, multiples of 100 apart, and unique.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SHADER = REPO / "cloudyview" / "soar" / "raymarch.wgsl"
COMPONENTS = REPO / "cloudyview" / "soar" / "city" / "components"
MARK_OPEN = "// >>> GENERATED CITY COMPONENTS"
MARK_CLOSE = "// <<< GENERATED CITY COMPONENTS"

HOOKS = {
    "extra_trace": (
        "fn cc_extra_trace(o: vec3<f32>, dir: vec3<f32>, inv_dir: vec3<f32>)\n"
        "        -> CityHit"),
    "cell_props": (
        "fn cc_cell_props_trace(o: vec3<f32>, dir: vec3<f32>, "
        "inv_dir: vec3<f32>,\n"
        "                       t0: f32, t1: f32, ci: vec2<i32>, "
        "cc: CityCell)\n"
        "        -> CityHit"),
    "shade": (
        "fn cc_component_shade(h: CityHit, cc: CityCell, dir: vec3<f32>, "
        "fp: f32)\n"
        "        -> vec3<f32>"),
    "facade": (
        "fn cc_facade_detail(cc: CityCell, h: CityHit, uc: f32, vc: f32, "
        "fp: f32)\n"
        "        -> vec3<f32>"),
    "window_glyph": (
        "fn cc_window_glyph(cc: CityCell, wh: vec4<f32>, "
        "pane_uv: vec2<f32>, fp: f32)\n"
        "        -> vec3<f32>"),
}

# Module-scope declarations only (column 0): a function-local `var` or
# `let` is the component's own business (windowlife found the indented
# variant flagged its locals and had to write let-only code).
SYMBOL_RE = re.compile(
    r"^(?:fn|const|var|struct)\s+([A-Za-z_][A-Za-z0-9_]*)", re.M)


def load_registry():
    reg = json.loads((COMPONENTS / "registry.json").read_text())
    if reg.get("schema") != "cloudyview.city.components.v1":
        raise SystemExit(f"unknown registry schema: {reg.get('schema')!r}")
    return [c for c in reg["components"] if c.get("enabled", True)]


def validate_component(comp, source):
    name = comp["name"]
    prefix = f"cc_{name}_"
    bad = [s for s in SYMBOL_RE.findall(source)
           if not s.startswith(prefix)]
    if bad:
        raise SystemExit(
            f"component '{name}' defines symbols outside its namespace "
            f"({prefix}*): {sorted(set(bad))}")
    for hook, fn in comp.get("hooks", {}).items():
        if hook not in HOOKS:
            raise SystemExit(f"component '{name}' names unknown hook "
                             f"'{hook}' (have {sorted(HOOKS)})")
        if not fn.startswith(prefix):
            raise SystemExit(f"component '{name}' hook '{hook}' must point "
                             f"at a {prefix}* function, got '{fn}'")
        if f"fn {fn}(" not in source:
            raise SystemExit(f"component '{name}' registers {fn} for hook "
                             f"'{hook}' but does not define it")
    kb = comp.get("kind_base")
    needs_kinds = any(h in comp.get("hooks", {})
                      for h in ("extra_trace", "cell_props"))
    if needs_kinds and (kb is None or kb < 100 or kb % 100 != 0):
        raise SystemExit(f"component '{name}' traces geometry and needs a "
                         f"kind_base >= 100 in multiples of 100; got {kb!r}")


def dispatcher(comps):
    """The five hook dispatchers over whatever is registered."""
    def hooked(h):
        return [c for c in comps if h in c.get("hooks", {})]

    out = []
    nearest_body = []
    for c in hooked("extra_trace"):
        nearest_body.append(
            f"    let h_{c['name']} = {c['hooks']['extra_trace']}"
            "(o, dir, inv_dir);\n"
            f"    if (h_{c['name']}.hit && h_{c['name']}.t < res.t) "
            f"{{ res = h_{c['name']}; }}")
    out.append(
        f"{HOOKS['extra_trace']} {{\n"
        "    var res: CityHit;\n    res.hit = false;\n    res.t = 1e30;\n"
        + ("\n".join(nearest_body) + "\n" if nearest_body else "")
        + "    return res;\n}")

    props_body = []
    for c in hooked("cell_props"):
        props_body.append(
            f"    let h_{c['name']} = {c['hooks']['cell_props']}"
            "(o, dir, inv_dir, t0, t1, ci, cc);\n"
            f"    if (h_{c['name']}.hit && h_{c['name']}.t < res.t) "
            f"{{ res = h_{c['name']}; }}")
    out.append(
        f"{HOOKS['cell_props']} {{\n"
        "    var res: CityHit;\n    res.hit = false;\n    res.t = 1e30;\n"
        + ("\n".join(props_body) + "\n" if props_body else "")
        + "    return res;\n}")

    shade_body = []
    for c in hooked("shade"):
        kb = c["kind_base"]
        shade_body.append(
            f"    if (h.kind >= {kb} && h.kind < {kb + 100}) {{\n"
            f"        return {c['hooks']['shade']}(h, cc, dir, fp);\n"
            "    }")
    out.append(
        f"{HOOKS['shade']} {{\n"
        + ("\n".join(shade_body) + "\n" if shade_body else "")
        + "    // An unclaimed component kind is a bug, and this is its "
          "color.\n"
        "    return vec3<f32>(1.0, 0.0, 1.0);\n}")

    facade_body = [
        f"    e = e + {c['hooks']['facade']}(cc, h, uc, vc, fp);"
        for c in hooked("facade")]
    out.append(
        f"{HOOKS['facade']} {{\n"
        "    var e = vec3<f32>(0.0);\n"
        + ("\n".join(facade_body) + "\n" if facade_body else "")
        + "    return e;\n}")

    glyph_body = [
        f"    t = t * {c['hooks']['window_glyph']}(cc, wh, pane_uv, fp);"
        for c in hooked("window_glyph")]
    out.append(
        f"{HOOKS['window_glyph']} {{\n"
        "    var t = vec3<f32>(1.0);\n"
        + ("\n".join(glyph_body) + "\n" if glyph_body else "")
        + "    return t;\n}")
    return "\n\n".join(out)


def compose(check_only: bool) -> int:
    comps = load_registry()
    kbs = [c["kind_base"] for c in comps if c.get("kind_base") is not None]
    if len(kbs) != len(set(kbs)):
        raise SystemExit(f"duplicate kind_base values: {sorted(kbs)}")

    pieces = []
    for comp in comps:
        source = (COMPONENTS / comp["file"]).read_text()
        validate_component(comp, source)
        pieces.append(f"// --- component: {comp['name']} "
                      f"({comp['file']}) ---\n" + source.strip())
    generated = (
        f"{MARK_OPEN} — written by tools/compose_city.py from\n"
        "// >>> cloudyview/soar/city/components/; edit the component files "
        "and re-run\n"
        "// >>> the composer, never this block.\n"
        + ("\n\n".join(pieces) + "\n\n" if pieces else "")
        + dispatcher(comps)
        + f"\n{MARK_CLOSE}")

    shader = SHADER.read_text()
    start = shader.index(MARK_OPEN)
    end = shader.index(MARK_CLOSE) + len(MARK_CLOSE)
    out = shader[:start] + generated + shader[end:]

    import wgpu
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = adapter.request_device_sync()
    for city in ("false", "true"):
        probe = out.replace("const CITY: bool = false;",
                            f"const CITY: bool = {city};", 1)
        device.create_shader_module(code=probe)

    if check_only:
        print(f"OK: {len(comps)} component(s) validate and compile.")
        return 0
    SHADER.write_text(out)
    print(f"Spliced {len(comps)} component(s) into {SHADER.name}; "
          "both CITY specializations compile.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="validate and compile only; write nothing")
    args = ap.parse_args()
    return compose(args.check)


if __name__ == "__main__":
    sys.exit(main())
