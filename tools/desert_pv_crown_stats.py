"""Palo verde crown statistics from a numpy mirror of raymarch.wgsl's lobes
(df_pv_limb, df_pv_lobe): the crown envelope's centre and half-height over
the height (DF_PV_CROWN_ZC / _RZ, DF_FAR_PV_ZC / _B, DF_FAR_BPV_ZC / _B), and
the far form's mean cover from above over its crown's disc (DF_FAR_PV_KAPPA,
DF_FAR_BPV_KAPPA, from which the decals' tau0 are solved). Draws are uniform
like the shader's hashes — not the same numbers: statistics only — so a
change to df_pv_limb, df_pv_lobe or DF_PV_LOBE_TAU re-runs this.

    python3 tools/desert_pv_crown_stats.py [DF_PV_LOBE_TAU]
"""
import sys
import numpy as np

LOBE_SIGMA = float(sys.argv[1]) if len(sys.argv) > 1 else 1.62  # DF_PV_LOBE_TAU
LPL = 3            # DF_PV_LOBES_PER_LIMB
rng = np.random.default_rng(1)


def tree(bank=False):
    U = rng.random
    if bank:
        # A creosote's reach, less 0.3, as df_paloverde_bank_item takes it:
        # df_creosote_dims (not bursage) and df_creosote_reach.
        hc = 0.7 + 1.3 * U() ** 1.3
        wc = hc * (0.40 + 0.20 * U())
        crown = wc + 0.30 * max(hc, 0.525) + 0.022 + 2 * 0.0075 - 0.3
        h = min(1.4 + 1.2 * crown, 3.0)
        cr = crown
        nt = int(2.0 + 1.99 * U())
    else:
        rx, ry, rz, rw = U(4)
        h = 3.0 + 3.0 * rx
        cr = (0.40 + 0.20 * ry) * h
        nt = int(3.0 + 2.99 * rz)
    yaw = U() * 2 * np.pi
    lobes = []
    for i in range(nt):
        hh = U(4)
        az = yaw + 2 * np.pi * (i + 0.3 * hh[0]) / nt
        d = np.array([np.cos(az), np.sin(az)])
        fork = np.array([*(d * (0.10 + 0.10 * hh[1]) * cr), (0.12 + 0.12 * hh[2]) * h])
        for k in (1, 2):
            hk = U(4)
            az2 = az + (0.55 if k == 1 else -0.55) + 0.5 * (hk[0] - 0.5)
            tip = np.array([*(np.array([np.cos(az2), np.sin(az2)]) * (0.40 + 0.14 * hk[1]) * cr),
                            (0.52 + 0.14 * hk[2]) * h])
            for m in range(LPL):
                hl = U(4)
                last = m == LPL - 1
                s = 0.92 + 0.08 * hl[0] if last else 0.20 + 0.35 * m + 0.15 * hl[0]
                rl = (0.30 + 0.10 * hl[1] if last else 0.225 + 0.125 * hl[1]) * cr
                rzz = (0.60 + 0.25 * hl[2]) * rl
                el = 0.85 + 0.30 * hl[3]
                c = fork + (tip - fork) * s + np.array([0, 0, 0.45 * rzz])
                r = np.array([rl * el, rl * (2 - el), rzz])
                sig = LOBE_SIGMA / cr * (0.55 + 0.90 * ((hl[3] * 7.31) % 1.0))
                lobes.append((c, r, sig))
    return h, cr, lobes


def stats(bank, n=400, grid=48):
    zc, zh, kap, ext = [], [], [], []
    for _ in range(n):
        h, cr, lobes = tree(bank)
        zlo = min(c[2] - r[2] for c, r, _ in lobes)
        zhi = max(c[2] + r[2] for c, r, _ in lobes)
        zc.append(0.5 * (zlo + zhi) / h)
        zh.append(0.5 * (zhi - zlo) / h)
        ext.append(max(np.hypot(*c[:2]) + max(r[0], r[1]) for c, r, _ in lobes) / cr)
        # Vertical optical depth over the crown's disc.
        g = np.linspace(-cr, cr, grid)
        X, Y = np.meshgrid(g, g)
        inside = X ** 2 + Y ** 2 <= cr ** 2
        tau = np.zeros_like(X)
        for c, r, sig in lobes:
            q = 1 - ((X - c[0]) / r[0]) ** 2 - ((Y - c[1]) / r[1]) ** 2
            tau += sig * 2 * r[2] * np.sqrt(np.clip(q, 0, None))
        kap.append(np.mean(1 - np.exp(-tau[inside])))
    return np.mean(zc), np.mean(zh), np.mean(kap), np.max(ext)


for bank in (False, True):
    zc, zh, kap, ext = stats(bank)
    print(f"{'bank' if bank else 'tree'}: centre {zc:.3f} h, half-height {zh:.3f} h, "
          f"kappa {kap:.3f} (lobe sigma {LOBE_SIGMA}), widest lobe extent {ext:.3f} crown")
