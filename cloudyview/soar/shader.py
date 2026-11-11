"""The soar shader's source: WGSL files under shader/, joined per surface.

manifest.json lists the files every module shares and, for each surface the
world can stand on, the files that draw it and the `absent` stand-ins that
take their place in a module compiled for another surface. A module therefore
holds the code of exactly one surface. web/soar/shader.js is the browser's
twin, and tests pin the two to the same text.
"""

import json
from functools import lru_cache
from pathlib import Path

SHADER_DIR = Path(__file__).resolve().parent / "shader"


@lru_cache(maxsize=None)
def manifest() -> dict:
    return json.loads((SHADER_DIR / "manifest.json").read_text())


def surfaces() -> tuple:
    """The surfaces a module can be compiled for, in manifest order."""
    return tuple(manifest()["surfaces"])


def shader_files(surface: str) -> list:
    """The files of `surface`'s module, in the order they are joined."""
    m = manifest()
    if surface not in m["surfaces"]:
        raise ValueError(f"unknown surface {surface!r}; the shader has "
                         f"{sorted(m['surfaces'])}.")
    files = list(m["core"]) + list(m["surfaces"][surface]["files"])
    for other, entry in m["surfaces"].items():
        if other != surface:
            files += entry["absent"]
    return files


@lru_cache(maxsize=None)
def assemble(surface: str) -> str:
    """`surface`'s module, without the switch header (see specialize)."""
    return "\n".join((SHADER_DIR / f).read_text()
                     for f in shader_files(surface))


def switches(*, periodic: bool, nested: bool, tone_map: bool = True,
             light_cache: bool = False) -> str:
    """The compile-time switch header uniforms.wgsl describes."""
    values = {"PERIODIC_DOMAIN": periodic, "NESTED": nested,
              "TONE_MAP": tone_map, "LIGHT_CACHE": light_cache}
    return "".join(f"const {name}: bool = {'true' if on else 'false'};\n"
                   for name, on in values.items())


def specialize(source: str, **switch_values) -> str:
    """A compilable module: the switch header, then `source`."""
    return switches(**switch_values) + source
