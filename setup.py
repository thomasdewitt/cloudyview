"""Build hook: snapshot the soar render assets into the wheel.

witness renders from web/soar/raymarch.wgsl and web/soar/ocean/, which live
beside the package so the browser and the Python host share one set of files.
A wheel cannot reach outside its package, so building copies them into
cloudyview/_soar_snapshot/ in the build output (never into the source tree —
in a checkout the live web/soar/ files remain the only copy, and
soar_host.py prefers them). Missing sources fail the build loudly.
"""
import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py


class build_py_with_soar_snapshot(build_py):
    def run(self):
        super().run()
        src = Path(__file__).resolve().parent / "web" / "soar"
        dest = Path(self.build_lib) / "cloudyview" / "_soar_snapshot"
        (dest / "ocean").mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / "raymarch.wgsl", dest / "raymarch.wgsl")
        for f in sorted((src / "ocean").iterdir()):
            shutil.copy2(f, dest / "ocean" / f.name)


setup(cmdclass={"build_py": build_py_with_soar_snapshot})
