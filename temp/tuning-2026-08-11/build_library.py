"""Build the visual-tuning iteration library: a self-contained HTML contact
sheet with the reference photo and every iteration render.

Usage: uv run python temp/tuning-2026-08-11/build_library.py
Writes temp/tuning-2026-08-11/library/ (index.html + img/*.webp).
"""
import os
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).parent
RENDERS = HERE / "renders"
OUT = HERE / "library"
IMG = OUT / "img"

REF = ("/tmp/claude-1000/-home-thomas-code-and-data-cloudyview/"
       "831ac381-87c7-448e-bcc9-b31c7a35ce13/scratchpad/ref_IMG_7017.png")

# (image source, id, title, notes) — order = presentation order.
ENTRIES = [
    (REF, "ref", "REFERENCE — IMG_7017 (real photo)",
     "Fair-weather cumulus, midday, sun off-frame left. The authority. "
     "Key measurements: mid-sky S 0.81; lit faces cool (B−R +20); bases "
     "blue-tinted luminous grey (S 0.22, V 0.76, linear B/R ≈ 2.4)."),
    (RENDERS / "iter00_baseline.png", "iter00",
     "iter00 — baseline (master @ 35453d3)",
     "The exact screenshot view re-rendered via soar_host: gamma 1.66, "
     "sun az 20 / el 55. Steel-grey mid-sky (S 0.70), warm dull whites, "
     "abrupt white→dark-grey bases (S 0.05, V 0.58)."),
    (RENDERS / "iter01_sky_gauss.png", "iter01",
     "iter01 — deep blue: Gaussian cutoff on horizon whitening",
     "Horizon whitening = legacy cubic × exp(−(z/0.33)²). Mid-sky "
     "saturation 0.70 → 0.77; horizon band and zenith within ~1/255 of "
     "before. The Gaussian width becomes the haze knob later."),
    (RENDERS / "iter02_cool_sun.png", "iter02",
     "iter02 — cooler direct beam",
     "SUN_COLOR (22,21,17) → (21.6,21.2,19.2). Whites B−R −5 → −2; "
     "small alone, groundwork for iter03."),
    (RENDERS / "iter03_wb_whitepoint.png", "iter03a",
     "iter03a — extended Reinhard white point (WB 0.972/1.0/1.045)",
     "White point 8.0 frees lit faces from the 227/255 Reinhard ceiling; "
     "whites reach the reference brightness (dV −0.003)."),
    (RENDERS / "iter03b_wb2.png", "iter03",
     "iter03 — committed: white balance widened (0.94/1.0/1.08)",
     "Whites bright and cool-neutral; sky_deep B matches the photo. "
     "Codex would instead use neutral beam + blue ambient — see iter08."),
    (RENDERS / "iter04_circumsolar.png", "iter04",
     "iter04 — broad circumsolar aerosol lobe (main view)",
     "Peak-normalized HG, g 0.68 (~17° half-width), amplitude 0.045, "
     "cool tint, warms with the spectral bloom at low sun. No cutoff, no "
     "elevation gate. Subtle at this view's 78°+ sun distance."),
    (RENDERS / "iter04_sunward_demo.png", "iter04s",
     "iter04 — sunward demo view (az 45, el 40)",
     "The same lobe seen toward the sun: warm core from the tight bloom, "
     "cool broad halo, sky desaturating over ~20°. The faint hue change "
     "the reference shows."),
    (RENDERS / "iter05c_diffuse_beam.png", "iter05",
     "iter05 — cloud bases: shallow/storm split + diffused-beam glow",
     "t_sky splits open shallow shadow from buried storm; high-sun "
     "skylight fill on a moderate-shadow gate; diffused-beam glow = "
     "isotropic Eddington tail of the beam (the out-and-back MS "
     "approximation at zero cost — the out-march is the existing "
     "tau_sun). Bases luminous, cliff softened; storms keep their dark."),
    (RENDERS / "iter06_beam_strong.png", "iter06a",
     "iter06a — beam strength 1.35, early gate",
     "Brightness matched (dV −0.016/−0.048) but bases too warm — the "
     "beam is sun-colored, reference bases are skylight-blue."),
    (RENDERS / "iter06c_blue_beam.png", "iter06b",
     "iter06b — strong blue beam tint (0.72,0.95,1.18)",
     "Blue right at the patches, but reads as sky-colored holes in cloud "
     "bodies — the tau_sun window amplifies light-march LOD quantization "
     "into visible patches. Rejected."),
    (RENDERS / "iter06d_softer.png", "iter06",
     "iter06 — committed: balanced (beam 0.95 @ mild blue, skylight 0.45)",
     "Blue from the smooth t_sky skylight fill, mild spectral blue on the "
     "beam (liquid water absorbs red over diffusion paths). Bases "
     "luminous blue-grey, no windows."),
    (RENDERS / "iter06_sunward_demo.png", "iter06s",
     "iter06 — sunward demo view (az 45, el 40) at committed state",
     "The circumsolar lobe with the final base lighting: warm core, cool "
     "broad halo, luminous cloud edges near the sun."),
    (RENDERS / "iter08_codex_whites.png", "iter08",
     "iter08 — ALTERNATIVE whites (codex's route; render-only, uncommitted)",
     "Neutral-cool beam (20.2, 21.0, 22.4), NO display white balance, "
     "luminance-preserving highlight shoulder (k = 0.35), bluer ambient "
     "(0.18, 0.225, 0.33) at strength 0.15. Whites genuinely blue-leaning "
     "(B−R +9) vs iter03's cool-neutral; sky slightly less matched. Pick "
     "this or iter03 — they are the same complaint solved two ways."),
    (RENDERS / "before_after.png", "beforeafter",
     "BEFORE / AFTER — baseline (top) vs committed branch (bottom)",
     "The whole pass in one image: deep blue mid-sky, vibrant cool "
     "whites, luminous blue-grey bases with a soft transition."),
    (RENDERS / "iter07_haze_0.0.png", "haze0",
     "iter07 — haze slider at 0.0 (crystalline)",
     "One 0–1 slider (settings menu, under gamma; default 0.35 = the "
     "committed look, verified to one pixel by 1/255). Drives aerial "
     "beta, ocean haze, the horizon-whitening wedge width+gain, and the "
     "circumsolar amplitude together. At 0: crisp far field, thin "
     "horizon band, deep sky everywhere."),
    (RENDERS / "iter07_haze_0.7.png", "haze07",
     "iter07 — haze slider at 0.7",
     "Distant clouds soften into the veil; horizon band widens."),
    (RENDERS / "iter07_haze_1.0.png", "haze1",
     "iter07 — haze slider at 1.0 (milky)",
     "Full haze: wide bright horizon band, strong aerial perspective, "
     "sunward glow broadened. Near clouds keep definition."),
    (RENDERS / "ref_anvil_IMG_7053.png", "ref2",
     "REFERENCE 2 — IMG_7053 (real photo, live from Thomas's phone)",
     "Thin cumulus under a convective anvil. Every buried-shadow patch "
     "holds B−R +15..+18 (cool blue-grey, V ≥ 0.45 even in the darkest "
     "cores) and the underside keeps soft billow definition deep into "
     "shadow. Drove iter09."),
    (RENDERS / "iter09_shadow_cool_def.png", "iter09v",
     "iter09 — shaded-cloud coolness + billow definition (main view)",
     "deep_fill_tint keeps the ambient's blue chroma (the old neutral "
     "anchor came from AI storm refs — linear B/R 0.97 displayed as only "
     "+4; the photo needs ~1.5); all diffuse fills now take the measured "
     "surface gradient at 0.6 weight so shadow keeps structure."),
    (RENDERS / "twpice_iter09.png", "twpice_9",
     "DEEP CONVECTION — TWPICE with iter09",
     "Cooler shadow tone and visibly more internal billow definition in "
     "the shaded mass vs the plain branch render below."),
    (RENDERS / "thomas_banding_still.png", "banding",
     "BUG REPORT — Thomas's still: non-monotonic base brightness",
     "White -> grey -> white -> grey moving into the mass. Cause: the "
     "diffuse beam / skylight gates opened over tau windows AFTER the "
     "beam and MS ladder died, so summed brightness dipped and re-rose."),
    (RENDERS / "edge_ab.png", "edge_ab",
     "iter11 — same view: old gates (top) vs exponential-onset gates (bottom)",
     "Gates now rise while the MS ladder is still alive; the wedge "
     "harness verifies brightness decays monotonically with tau (worst "
     "re-rise ~1/255). Storm floor raised 0.25 -> 0.55 to match the "
     "anvil photo's darkest cores (old floor was darker than the real "
     "reference). iter10 = the iter08 whites, landed. Default gamma is "
     "now 1.66."),
    (RENDERS / "iter12_default_haze1.png", "iter12",
     "iter12 — NEW DEFAULT: haze 1.0, white point 12",
     "Thomas's flight verdicts: haze slammed to 1 is the default now "
     "(slider runs to 2), and the white point moved 8 → 12 so the "
     "brightest faces stay just under clipping (blown-pixel fraction "
     "0.03% → 0.00%). This is the shipping look of the branch."),
    (RENDERS / "iter12_haze2.png", "iter12h2",
     "iter12 — haze slider at 2.0 (soupy)",
     "The extended range's far end."),
    (RENDERS / "twpice_master.png", "twpice_m",
     "DEEP CONVECTION — TWPICE 256², master",
     "The pre-tuning deep-convection look: heavy flat dark grey, abrupt "
     "white→grey — the same complaint as the shallow bases."),
    (RENDERS / "twpice_branch.png", "twpice_b",
     "DEEP CONVECTION — TWPICE 256², tuned branch",
     "The diffuse beam reaches moderate-tau samples in the towers too; "
     "the mass reads as luminous congestus. The true storm gate "
     "(tau_sun 38–80) engages less than the old look suggested."),
    (RENDERS / "twpice_loose.png", "twpice_l",
     "DEEP CONVECTION — storm-loosened variant (beam storm keep 0.25 → 0.55)",
     "Lifts the remaining dark under-band further, slightly flatter "
     "structure. Uncommitted — pick if the branch version is still too "
     "sharp for deep convection."),
    (RENDERS / "lowsun_master.png", "lowsun_m",
     "REGRESSION — golden hour, master (sun el 10°)",
     "Pre-tuning reference for the low-sun mood."),
    (RENDERS / "lowsun_tuned.png", "lowsun_t",
     "REGRESSION — golden hour, tuned branch (sun el 10°)",
     "Warm wedge preserved; upper sky deeper (wedge change); cloud bodies "
     "brighter/creamier from the diffuse beam — backlit translucency is "
     "real, but this is a look change to judge. Easy to elevation-gate "
     "the beam if the moodier master version is preferred."),
]


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    IMG.mkdir(parents=True)
    cards = []
    for src, eid, title, notes in ENTRIES:
        src = Path(src)
        if not src.exists():
            print(f"SKIP missing {src}")
            continue
        dst = IMG / f"{eid}.webp"
        subprocess.run(
            ["magick", str(src), "-resize", "1600x1600", "-quality", "88",
             str(dst)], check=True)
        cards.append(
            f'<figure id="{eid}">'
            f'<img src="img/{eid}.webp" loading="lazy" alt="{eid}">'
            f'<figcaption><b>{title}</b><br>{notes}</figcaption></figure>')
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>soar visual tuning — 2026-08-11</title>
<style>
 body {{ background:#111; color:#ddd; font:15px/1.45 system-ui, sans-serif;
        max-width: 1660px; margin: 0 auto; padding: 16px; }}
 h1 {{ font-size: 20px; }} a {{ color:#8cf; }}
 figure {{ margin: 0 0 28px 0; }}
 img {{ max-width: 100%; border-radius: 4px; display:block; }}
 figcaption {{ padding: 8px 2px; color:#bbb; }}
 figcaption b {{ color:#fff; }}
 .hint {{ color:#888; }}
</style></head><body>
<h1>soar visual tuning — 2026-08-11 — branch <code>visual-tuning</code></h1>
<p class="hint">Reference view: small_c002_s0030.nc (parent), camera from the
screenshot's metadata, sun az 20 / el 55, gamma 1.66. One commit per kept
iteration; variants shown were measured and superseded. Open images in new
tabs and flip between them to compare.</p>
{''.join(cards)}
</body></html>"""
    (OUT / "index.html").write_text(html)
    n = len(cards)
    size = sum(f.stat().st_size for f in IMG.iterdir()) / 1e6
    print(f"library: {n} entries, {size:.1f} MB → {OUT}/index.html")


if __name__ == "__main__":
    main()
