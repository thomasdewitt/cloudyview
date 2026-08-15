"""Every host must bind exactly what raymarch.wgsl declares.

There is one renderer core and three places that wire resources into it: the
browser's Renderer (its layout, its main bind group, and the separate one the
exposure meter builds) and the Python host's SoarRenderer. A binding the
shader declares and a host omits is a validation error at pipeline creation —
loud in Python, where the golden suite compiles the shader every run, and
invisible here until someone opens the app on a machine with a GPU.

That asymmetry is the reason for this file. The Python host is exercised by
tests/test_soar_witness_renders.py on every run; the browser's wiring is
exercised by nothing that CI can run, because the tab needs a real WebGPU
device. So the binding table is compared as text instead: the numbers the
shader declares, against the numbers each host lists.

It is a shallow check on purpose — it knows binding numbers, not what is
plugged into them. It exists to catch the specific mistake of adding a
resource to the shader and to one host, which is how binding 6 (the periodic
volume sampler, added when the ghost ring was retired) could have shipped
broken in the browser while every Python test passed.

Needs neither node nor a GPU.
"""

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SHADER = REPO / "web" / "soar" / "raymarch.wgsl"
RENDERER_JS = REPO / "web" / "soar" / "renderer.js"
SOAR_HOST = REPO / "cloudyview" / "soar_host.py"

pytestmark = pytest.mark.skipif(
    not SHADER.exists(), reason="needs web/soar/raymarch.wgsl")


def shader_bindings():
    """@group(0) @binding(N) declarations, N -> declared name."""
    source = SHADER.read_text()
    found = dict(
        (int(n), name) for n, name in
        re.findall(r"@group\(0\)\s*@binding\((\d+)\)\s*var(?:<[^>]*>)?\s*"
                   r"(\w+)\s*:", source))
    assert found, "no @group(0) bindings found — the regex has gone stale"
    return found


def _call_block(source, start):
    """The text of the call whose opening paren follows `start`."""
    i = source.index("(", start)
    depth = 0
    for j in range(i, len(source)):
        if source[j] in "([{":
            depth += 1
        elif source[j] in ")]}":
            depth -= 1
            if depth == 0:
                return source[start:j + 1]
    raise AssertionError("unbalanced brackets in renderer.js")


def js_binding_lists():
    """Every binding list in renderer.js built against the raymarch layout.

    Found by walking the calls rather than by line anchors: there is the
    layout itself, the main bind group, and the exposure meter's own — and
    the meter's is the one a person editing refreshBindGroup forgets.
    """
    source = RENDERER_JS.read_text()
    lists = {}
    start = source.index("this.rayLayout = ")
    lists["rayLayout"] = {
        int(n) for n in
        re.findall(r"binding:\s*(\d+)", _call_block(source, start))}
    seen = 0
    for match in re.finditer(r"createBindGroup", source):
        block = _call_block(source, match.start())
        if "this.rayLayout" not in block:
            continue
        seen += 1
        lists[f"bindGroup{seen}"] = {
            int(n) for n in re.findall(r"binding:\s*(\d+)", block)}
    assert seen == 2, (
        f"expected 2 bind groups against rayLayout in renderer.js, found "
        f"{seen}; the parser or the renderer has moved")
    for label, numbers in lists.items():
        assert numbers, f"no bindings parsed out of {label}"
    return lists


def python_binding_lists():
    """The layout and the bind group in soar_host.SoarRenderer."""
    source = SOAR_HOST.read_text()
    lists = {}
    for label, anchor, stop in (
            ("_ray_layout", "self._ray_layout = d.create_bind_group_layout(",
             "        ])"),
            ("_bind_group", "self._bind_group = self.device.create_bind_group(",
             "            ])")):
        start = source.index(anchor)
        end = source.index(stop, start)
        numbers = {int(n) for n in
                   re.findall(r'"binding":\s*(\d+)', source[start:end])}
        assert numbers, f"no bindings parsed out of {label}"
        lists[label] = numbers
    return lists


def test_every_host_binds_what_the_shader_declares():
    declared = set(shader_bindings())
    for label, numbers in {**js_binding_lists(), **python_binding_lists()}.items():
        missing = sorted(declared - numbers)
        extra = sorted(numbers - declared)
        assert not missing and not extra, (
            f"{label} does not match raymarch.wgsl's @group(0): "
            f"missing {missing}, unexpected {extra}. Adding a resource to the "
            "shader means adding it to the layout AND to every bind group "
            "built against that layout.")


def test_the_periodic_sampler_is_declared():
    """The binding the de-padding added, named rather than merely counted, so
    that renumbering cannot quietly satisfy the test above."""
    assert shader_bindings().get(6) == "vol_wrap_samp"
    assert "volWrapSampler" in RENDERER_JS.read_text()
    assert "_vol_wrap_sampler" in SOAR_HOST.read_text()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
