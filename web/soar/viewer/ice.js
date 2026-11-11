// Viewer methods (ice). Ice-detection mode and the dart-tip condensate
// readout.

"use strict";

import * as K from "../constants.js";
import { loadDemoIceVolume } from "../scene.js";
import { retireAfterSubmittedWork } from "../gpu.js";
import { rhoAirAt } from "../optical.js";
import { fromHalf } from "../half.js";

export const ice = {
  /**
   * I: ice-detection mode — false-color ice fraction (liquid red to ice cyan,
   * alien green sky) plus the dart-tip condensate readout.
   *
   * The first activation for a field reads its ice fraction, which is a
   * second pass over the file and a second volume the size of the first. That
   * is why nothing happens until somebody asks: a flight that never presses I
   * pays none of it. The mode comes on when the volume is there and not
   * before — a half-uploaded fraction would paint the unread part liquid-red,
   * which is a picture rather than a wait. Every toggle after that is
   * immediate.
   */
  async toggleIceMode() {
    if (this.iceModeOn) {                 // off is always free
      this.disableIceMode();
      this.ui.say("Ice detection off.", 4);
      return;
    }
    // The read in flight is for this scene: the toast is already saying so.
    // Before the token is taken, so a second press does not cancel the
    // activation the first one is waiting for.
    if (this._iceLoadingFor === this.scene) return;
    // Taken before the await so that anything which cancels the activation
    // while the volume is being read — disableIceMode — is seen below.
    const request = ++this._iceRequest;
    // No scene at all — the gap while a load is in flight. Nothing to say
    // about ice that does not exist yet; this used to read iceNote off null.
    if (!this.scene) return;
    if (!this.scene.iceAvailable) {
      this.ui.say(this.scene.iceNote, 6);
      return;
    }
    if (!this.scene.iceTexture) {
      // The read takes seconds and the menu is right there, so the field can
      // change under it. The fraction belongs to the scene it was read from
      // and to no other — attaching it to whatever is loaded when it lands
      // would tint one field by another's ice — so the scene is held here and
      // the texture is given back if it is no longer the one being flown.
      const scene = this.scene;
      this._iceLoadingFor = scene;
      try {
        const options = { progress: (message, done) => this.ui.say(
          `${message}${done > 0 && done < 1
            ? ` (${(done * 100).toFixed(0)}%)` : ""}`, 6) };
        // Two sources, one shape of load: a demo's fraction is a prebaked
        // file beside its volume, a file's is a second pass over the NetCDF.
        // The demo path deliberately does not import ./ingest — a baked
        // field needs no HDF5 reader, and pulling one in would be megabytes
        // of wasm to tint a volume that is already quantized.
        const texture = scene.iceSource.kind === "demo"
          ? await loadDemoIceVolume(this.device, scene.iceSource, options)
          : await (await import("../ingest/index.js"))
              .loadIceVolume(this.device, scene.iceSource, options);
        // Fenced: the upload's writes for this texture may still be queued,
        // and destroying under them crashes GPU processes rather than
        // raising (see gpu.retireAfterSubmittedWork).
        if (this.scene !== scene) {
          retireAfterSubmittedWork(this.device, texture);
          return;
        }
        scene.attachIce(texture);
        this.renderer.refreshBindGroup();
      } catch (err) {
        this.ui.say(
          `The ice fraction could not be read: ${err.message}`, 8);
        return;
      } finally {
        // Only this request's own claim is released. A read that outlived its
        // field must not clear the flag a later scene's read set.
        if (this._iceLoadingFor === scene) this._iceLoadingFor = null;
      }
    }
    // The activation can be cancelled while the volume is read — leaving
    // research mode does that. The volume is kept (it was paid for and
    // belongs to this scene); the mode does not come on behind a menu that
    // has no row to turn it off again.
    if (request !== this._iceRequest) return;
    this.iceModeOn = true;
    this.renderer.iceMode = true;
    this._syncIceReadout();
    this.ui.say("Ice detection on — liquid burns red, ice glows cyan.", 4);
    this._wake("iceMode");
  },

  /**
   * Turn ice detection off and cancel any activation still waiting on a read.
   *
   * Called by the toggle and by ui.setMode when the mode being entered has no
   * ice control. Silent: the caller says whatever needs saying.
   */
  disableIceMode() {
    this._iceRequest += 1;         // discards a pending activation
    if (!this.iceModeOn) return;
    this.iceModeOn = false;
    this.renderer.iceMode = false;
    this._syncIceReadout();
    this._wake("iceMode");
  },

  _iceReadoutEl() {
    return document.getElementById("ice-readout");
  },

  _syncIceReadout() {
    // Hidden until a probe finds condensate: the readout only shows while
    // the flyer is actually inside cloud (see _setIceReadout).
    const el = this._iceReadoutEl();
    if (el) el.hidden = true;
  },

  /**
   * Rate-limited dart-tip probe: two texels — extinction and ice fraction —
   * read back from the resident volumes at the flyer's position and inverted
   * through the extinction prefactors to mixing ratios. Nearest voxel, on
   * the shader's own linear world-to-grid map (a stretched vertical grid is
   * probed as the march renders it, not as the file stores it).
   */
  _iceTick(now) {
    if (!this.iceModeOn || this._iceProbePending || this._capturing) return;
    if (now - this._iceProbeAt < 200) return;
    const scene = this.scene;
    if (!scene?.iceTexture) return;
    const [nx, ny, nz] = scene.shape;
    if (nz < 2) return;              // the 2-texel copy needs the room

    const p = this.flyer && this.flyerEnabled
      ? this.flyer.position : this.camera.position;
    const { bmin, bmax } = scene;
    const frac = (v, i) => (v - bmin[i]) / (bmax[i] - bmin[i]);
    let gx = frac(p[0], 0), gy = frac(p[1], 1);
    const gz = frac(p[2], 2);
    if (this.renderer.periodic) {
      gx -= Math.floor(gx); gy -= Math.floor(gy);
    }
    const outside = gx < 0 || gx >= 1 || gy < 0 || gy >= 1
                 || gz < 0 || gz >= 1;
    this._iceProbeAt = now;
    if (outside) { this._setIceReadout(0.0, 0.0); return; }

    const ix = Math.min(nx - 1, Math.floor(gx * nx));
    const iy = Math.min(ny - 1, Math.floor(gy * ny));
    const iz = Math.min(nz - 1, Math.floor(gz * nz));
    const iz0 = Math.min(iz, nz - 2);   // 4-byte copy alignment wants 2 texels
    const zCentre = bmin[2] + (iz + 0.5) * (bmax[2] - bmin[2]) / nz;

    this._iceProbeBuf ??= this.device.createBuffer({
      label: "soar-ice-probe",
      size: 16,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const origin = { x: iz0, y: iy, z: ix };   // texture is (w=z, h=y, d=x)
    const enc = this.device.createCommandEncoder({ label: "ice-probe" });
    enc.copyTextureToBuffer(
      { texture: scene.volumeTexture, origin },
      { buffer: this._iceProbeBuf, offset: 0 }, [2, 1, 1]);
    enc.copyTextureToBuffer(
      { texture: scene.iceTexture, origin },
      { buffer: this._iceProbeBuf, offset: 8 }, [2, 1, 1]);
    this.device.queue.submit([enc.finish()]);

    this._iceProbePending = true;
    this._iceProbeBuf.mapAsync(GPUMapMode.READ).then(() => {
      const raw = this._iceProbeBuf.getMappedRange().slice(0);
      this._iceProbeBuf.unmap();
      this._iceProbePending = false;
      const k = iz - iz0;
      // Two texels of each texture, in the two formats they are stored in:
      // fp16 extinction from byte 0, uint8 ice fraction from byte 8. The
      // fraction comes back through the same 1/255 the write quantized with,
      // so the readout is what the shader is sampling rather than what the
      // file said.
      const sigma = fromHalf(new Uint16Array(raw, 0, 2)[k]);
      const f = new Uint8Array(raw, 8, 2)[k] / 255.0;
      // sigma = (P_liq * lwc + P_ice * iwc) * rho_air, f = ice share of it.
      const rho = rhoAirAt(zCentre);
      const iwc = f * sigma / (K.SIGMA_ICE_PREFACTOR * rho);
      const lwc = (1.0 - f) * sigma / (K.SIGMA_LIQUID_PREFACTOR * rho);
      this._setIceReadout(iwc, lwc);
    }).catch(() => { this._iceProbePending = false; });
  },

  _setIceReadout(iwcGkg, lwcGkg) {
    const el = this._iceReadoutEl();
    if (!el) return;
    // Only while inside cloud: both values rounding to zero (clear air, or
    // outside the domain) hides the readout rather than reporting nothing.
    // A non-finite value still shows — NaN condensate is worth seeing.
    const seen = (v) => !Number.isFinite(v) || Math.abs(v) >= 5e-4;
    const show = this.iceModeOn && (seen(iwcGkg) || seen(lwcGkg));
    el.hidden = !show;
    if (!show) return;
    const where = this.flyer && this.flyerEnabled
      ? (this.flyerKind === "bird" ? "At the beak" : "At dart tip") : "Here";
    const fmt = (v) => (Number.isFinite(v)
      ? (Math.abs(v) < 5e-4 ? "0.000" : v.toFixed(3)) : "?");
    el.textContent =
      `${where}: ${fmt(iwcGkg)} g/kg ice, ${fmt(lwcGkg)} g/kg liquid`;
  },
};
