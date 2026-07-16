# -*- mode: python ; coding: utf-8 -*-
"""Cross-platform PyInstaller one-directory build for CloudyView Soar."""

from pathlib import Path
import sys

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
    copy_metadata,
)


ROOT = Path(SPECPATH).resolve().parent
PACKAGING = ROOT / "packaging"

datas = [
    (str(ROOT / "data" / "TWPICE_subvolume_256x256_5km.nc"), "data"),
    (str(PACKAGING / "icon_512.png"), "packaging"),
]
datas += collect_data_files("wgpu")
datas += collect_data_files("imgui_bundle", includes=["assets/fonts/**"])
datas += collect_data_files("netCDF4")
datas += collect_data_files("cloudyview", includes=["soar/*.wgsl"])

binaries = []
for package in ("wgpu", "glfw", "imgui_bundle", "netCDF4"):
    binaries += collect_dynamic_libs(package)

hiddenimports = []
for package in ("wgpu", "rendercanvas", "glfw"):
    hiddenimports += collect_submodules(package)
hiddenimports += collect_submodules("xarray.backends")
hiddenimports += [
    "imgui_bundle",
    "imgui_bundle._imgui_bundle",
    "imgui_bundle.imgui",
    "netCDF4",
    "xarray",
    "cloudyview.ocean_fif",  # lazy import in soar engine (_default_fif_normals)
]
hiddenimports += collect_submodules("scaleinvariance")

# scaleinvariance is installed editable (uv .pth import hook, invisible to
# PyInstaller's Analysis) — resolve its real source tree onto pathex.
import scaleinvariance as _si
from pathlib import Path as _Path
_editable_paths = [str(_Path(_si.__file__).resolve().parents[1])]

for distribution in (
    "wgpu",
    "rendercanvas",
    "glfw",
    "imgui-bundle",
    "numpy",
    "xarray",
    "netCDF4",
):
    try:
        datas += copy_metadata(distribution)
    except Exception:
        # Some platforms do not expose metadata for binary-only wheels. The
        # imports and native libraries above remain the authoritative inputs.
        pass

excludes = [
    "mitsuba",
    "drjit",
    "numba",
    "llvmlite",
    "matplotlib",
]

icon = PACKAGING / "icon_512.png"
if sys.platform == "win32":
    icon = PACKAGING / "icons" / "cloudyview-soar.ico"
elif sys.platform == "darwin":
    icon = PACKAGING / "icons" / "cloudyview-soar.icns"

a = Analysis(
    [str(PACKAGING / "smoke_test.py")],
    pathex=[str(ROOT)] + _editable_paths,
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="cloudyview-soar",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    icon=str(icon),
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="cloudyview-soar",
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="CloudyView Soar.app",
        icon=str(icon),
        bundle_identifier="org.cloudyview.soar",
        info_plist={
            "CFBundleDisplayName": "CloudyView Soar",
            "NSHighResolutionCapable": True,
        },
    )
