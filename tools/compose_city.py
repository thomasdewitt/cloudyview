#!/usr/bin/env python
"""Splice the city component files into raymarch.wgsl.

The night city grows by components: WGSL files under
cloudyview/soar/city/components/, each owning one kind of thing and listed in
registry.json there. This tool concatenates the enabled ones, generates the
ten hook dispatchers the core calls (see the hook comment block in
raymarch.wgsl), replaces the GENERATED block between the markers, and
compile-validates the result under wgpu for both CITY specializations before
writing a byte.

Run it after editing any component or the registry:

    tools/gpulock uv run python tools/compose_city.py            # splice + validate + write
    tools/gpulock uv run python tools/compose_city.py --check    # validate only, no write

Rules it enforces (the same ones components/SPEC.md states):
  * a component may only define module-scope symbols named cc_<name>_* —
    collisions with the core or with other components are refused;
  * every hook named in the registry must exist in the component's file with
    the exact dispatcher-compatible signature (checked by compiling a probe);
  * kind_base values are >= 100, multiples of 100 apart, and unique, and
    each component's kinds (kind_base + 0..15) end below the facade shell's,
    which start at the core's CITY_KIND_SHELL.
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
        "fn cc_facade_detail(cc: CityCell, h: CityHit, dir: vec3<f32>,\n"
        "                    uc: f32, vc: f32, fp: f32)\n"
        "        -> vec3<f32>"),
    "facade_cover": (
        "fn cc_facade_cover(cc: CityCell, h: CityHit, uc: f32, vc: f32, "
        "fp: f32)\n"
        "        -> f32"),
    "window_glyph": (
        "fn cc_window_glyph(cc: CityCell, wh: vec4<f32>, "
        "pane_uv: vec2<f32>,\n"
        "                   deep_uv: vec2<f32>, fp: f32)\n"
        "        -> vec3<f32>"),
    "window_glyph_mean": (
        "fn cc_window_glyph_mean()\n"
        "        -> vec3<f32>"),
    "media": (
        "fn cc_media(o: vec3<f32>, dir: vec3<f32>, inv_dir: vec3<f32>, "
        "t_max: f32,\n"
        "            pixel_angle: f32, rnd: f32) -> vec4<f32>"),
    "roof": (
        "fn cc_roof_detail(cc: CityCell, h: CityHit, fp: f32)\n"
        "        -> vec3<f32>"),
    # Street-level props traced only by the street stage (fs_surface_street,
    # city_trace's CITY_TRACE_NEAR mode), never by the view ray's first
    # stage: detail that would cost every city pixel occupancy there.
    "near_props": (
        "fn cc_near_props_trace(o: vec3<f32>, dir: vec3<f32>, "
        "inv_dir: vec3<f32>,\n"
        "                       t0: f32, t1: f32, ci: vec2<i32>, "
        "cc: CityCell)\n"
        "        -> CityHit"),
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
    hooks = comp.get("hooks", {})
    if ("window_glyph" in hooks) != ("window_glyph_mean" in hooks):
        raise SystemExit(f"component '{name}' must register window_glyph "
                         f"and window_glyph_mean together: the core's far "
                         f"facade reads the glyph's population mean")
    kb = comp.get("kind_base")
    needs_kinds = any(h in comp.get("hooks", {})
                      for h in ("extra_trace", "cell_props", "near_props"))
    if needs_kinds and (kb is None or kb < 100 or kb % 100 != 0):
        raise SystemExit(f"component '{name}' traces geometry and needs a "
                         f"kind_base >= 100 in multiples of 100; got {kb!r}")


def dispatcher(comps):
    """The ten hook dispatchers over whatever is registered."""
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
        f"    e = e + {c['hooks']['facade']}(cc, h, dir, uc, vc, fp);"
        for c in hooked("facade")]
    out.append(
        f"{HOOKS['facade']} {{\n"
        "    var e = vec3<f32>(0.0);\n"
        + ("\n".join(facade_body) + "\n" if facade_body else "")
        + "    return e;\n}")

    cover_body = [
        f"    m = max(m, {c['hooks']['facade_cover']}(cc, h, uc, vc, fp));"
        for c in hooked("facade_cover")]
    out.append(
        f"{HOOKS['facade_cover']} {{\n"
        "    var m = 0.0;\n"
        + ("\n".join(cover_body) + "\n" if cover_body else "")
        + "    return m;\n}")

    glyph_body = [
        f"    t = t * {c['hooks']['window_glyph']}(cc, wh, pane_uv, deep_uv, "
        "fp);"
        for c in hooked("window_glyph")]
    out.append(
        f"{HOOKS['window_glyph']} {{\n"
        "    var t = vec3<f32>(1.0);\n"
        + ("\n".join(glyph_body) + "\n" if glyph_body else "")
        + "    return t;\n}")

    glyph_mean_body = [
        f"    t = t * {c['hooks']['window_glyph_mean']}();"
        for c in hooked("window_glyph_mean")]
    out.append(
        f"{HOOKS['window_glyph_mean']} {{\n"
        "    var t = vec3<f32>(1.0);\n"
        + ("\n".join(glyph_mean_body) + "\n" if glyph_mean_body else "")
        + "    return t;\n}")

    # Participating media in front of the city (steam): each component
    # returns (in-scattered radiance, transmittance) over [0, t_max]; they
    # compose front to back in registry order. With none registered the
    # dispatcher is the identity (0, 0, 0, 1), which the call site applies
    # as `rgb + a * city` — exact, so an empty registry changes no bit.
    media_body = []
    for c in hooked("media"):
        media_body.append(
            f"    let m_{c['name']} = {c['hooks']['media']}"
            "(o, dir, inv_dir, t_max, pixel_angle, rnd);\n"
            f"    acc = vec4<f32>(acc.rgb + acc.a * m_{c['name']}.rgb, "
            f"acc.a * m_{c['name']}.a);")
    out.append(
        f"{HOOKS['media']} {{\n"
        "    var acc = vec4<f32>(0.0, 0.0, 0.0, 1.0);\n"
        + ("\n".join(media_body) + "\n" if media_body else "")
        + "    return acc;\n}")

    near_body = []
    for c in hooked("near_props"):
        near_body.append(
            f"    let h_{c['name']} = {c['hooks']['near_props']}"
            "(o, dir, inv_dir, t0, t1, ci, cc);\n"
            f"    if (h_{c['name']}.hit && h_{c['name']}.t < res.t) "
            f"{{ res = h_{c['name']}; }}")
    out.append(
        f"{HOOKS['near_props']} {{\n"
        "    var res: CityHit;\n    res.hit = false;\n    res.t = 1e30;\n"
        + ("\n".join(near_body) + "\n" if near_body else "")
        + "    return res;\n}")

    roof_body = [
        f"    e = e + {c['hooks']['roof']}(cc, h, fp);"
        for c in hooked("roof")]
    out.append(
        f"{HOOKS['roof']} {{\n"
        "    var e = vec3<f32>(0.0);\n"
        + ("\n".join(roof_body) + "\n" if roof_body else "")
        + "    return e;\n}")
    return "\n\n".join(out)


def composed_text() -> tuple[list, str]:
    """The registry's components, and raymarch.wgsl as the composer would
    write it from them (nothing compiled, nothing written)."""
    comps = load_registry()
    kbs = [c["kind_base"] for c in comps if c.get("kind_base") is not None]
    if len(kbs) != len(set(kbs)):
        raise SystemExit(f"duplicate kind_base values: {sorted(kbs)}")
    # The facade shell's hit kinds (raymarch.wgsl, CITY_KIND_SHELL + part +
    # 16 * (side + 4 * tier)) sit above every component's.
    m = re.search(r"^const CITY_KIND_SHELL: i32 = (\d+);", SHADER.read_text(),
                  re.MULTILINE)
    if m is None:
        raise SystemExit("raymarch.wgsl defines no CITY_KIND_SHELL")
    shell = int(m.group(1))
    for c in comps:
        kb = c.get("kind_base")
        if kb is not None and kb + 15 >= shell:
            raise SystemExit(f"component '{c['name']}' kind_base {kb} runs into "
                             f"the facade shell's kinds (from {shell})")

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
    return comps, shader[:start] + generated + shader[end:]


def compose(check_only: bool) -> int:
    comps, out = composed_text()
    # This opens a GPU device to compile-validate (--check too), so run it
    # under tools/gpulock: a held context makes locked timing refuse.
    import wgpu
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = adapter.request_device_sync()
    # Compiled as the hosts compile it: stripped to one surface's sections
    # (soar_host.strip_sections), since the raw file holds each surface's
    # stub twin beside its real code.
    from cloudyview.soar_host import strip_sections
    for city in ("false", "true"):
        probe = strip_sections(out, "city" if city == "true" else "ocean")
        probe = probe.replace("const CITY: bool = false;",
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
    # It opens a GPU device: refuse outside tools/gpulock (gpulock_guard),
    # before the parse: -h/--help still answers unlocked.
    from gpulock_guard import require_gpulock
    require_gpulock("tools/compose_city.py")
    args = ap.parse_args()
    return compose(args.check)


if __name__ == "__main__":
    sys.exit(main())
