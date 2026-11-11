// Drive soar the way a user does, in a real Chrome, and assert on app state.
//
//   node tools/browser_smoke.mjs [--web web] [--cdp 9333] [--scenes day,city]
//
// For a day demo (?demo=twpice, research mode) and the cyberpunk city, this
// loads the scene and then: flies with key events, picks a tier in the
// Quality panel, backs out with Escape, parks until the picture converges,
// cycles the minimap and the flyer, lets the loop sleep and wakes it, moves
// the sun, flips the light cache (K), toggles the periodic
// domain, opens every pause-menu panel, reads the terminal commands, saves a
// still, records a track and renders it to video and saves it, toggles ice
// mode, and (day only) opens data/TWPICE_subvolume_128x128_5km.nc through the
// file picker path.
//
// Output: one line per step.
//   ok    the step passed; what follows is deterministic and should be the
//         same on every run of the same tree
//   ~     a measured value that varies run to run (timings, counts)
//   skip  the step could not run here, and why
//   FAIL  what was expected and what was found
// Two trees behave alike when their ok/skip/FAIL lines are the same.
// Exit 0 when nothing failed, 1 when a step failed, 2 when it cannot run.
//
// What this does NOT drive, because headless Chrome cannot:
//   - A real download. URL.createObjectURL and <a download>.click() are
//     intercepted in the page, so a still, a video and a track get as far as
//     their finished Blob and no file is written.
//   - The native file picker. window.showOpenFilePicker is replaced with one
//     that returns a File built from a fetch of the test file; everything
//     after the picker (Viewer.pickFile, the ingest worker, the question
//     panels) is the app's own.
//   - Pointer lock, if this Chrome refuses it without a real click. The step
//     that needs it (mouse look, Escape releasing the lock into the pause
//     menu) then says "skip". The keyboard paths do not need the lock.
//   - Fullscreen, and the clipboard behind "Copy the command".
// And what it leaves out by choice: the first-flight tutorial (marked done in
// localStorage, or it would own the keys), the mouse wheel, click-to-travel
// on the fullscreen minimap, the sliders (the setters behind them are
// called directly where a step needs one), and leaving the viewer.
// The mode is set through localStorage before the page loads, as
// pinned_still.mjs does; the landing page's own mode toggle is not clicked.
// The cyberpunk city is entered through its "fly" button.

import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { openTab, requireHardwareAdapter, serveTree } from "./soar_cdp.mjs";

const repo = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const argv = process.argv.slice(2);
const option = (name, fallback) => {
  const i = argv.indexOf(name);
  return i < 0 ? fallback : argv[i + 1];
};
const webDir = option("--web", join(repo, "web"));
const cdpPort = option("--cdp", "9333");
const scenes = option("--scenes", "day,city").split(",");

const SCENES = {
  day: { mode: "research", demo: "twpice" },
  city: { mode: "cyberpunk", demo: null },
};
const NC_NAME = "TWPICE_subvolume_128x128_5km.nc";

let failed = 0, passed = 0, skipped = 0;
let tab = null;

class Skip extends Error {}

/** Run one step in the page. `body` is the inside of an async function with
 *  `s` (window.soar), `t` (the helpers below) and `K` (constants.js) in
 *  scope; it returns the step's deterministic detail (a string), or an
 *  object { detail, vary } to also print a "~" line. It throws to fail and
 *  calls t.skip(why) to skip. */
async function step(scene, name, body, timeoutMs = 180000) {
  const label = `${scene}: ${name}`;
  let r;
  try {
    r = await tab.eval(`(async () => {
      const s = window.soar, t = window.__smoke, K = t.K;
      try {
        const out = await (async () => { ${body} })();
        return { ok: true, out: out ?? "" };
      } catch (err) {
        return { ok: false, skip: Boolean(err.smokeSkip),
                 message: String(err.message || err) };
      }
    })()`, timeoutMs);
  } catch (err) {
    r = { ok: false, message: err.message };
  }
  if (r.ok) {
    passed += 1;
    const detail = typeof r.out === "string" ? r.out : r.out.detail;
    console.log(`ok    ${label}${detail ? ` — ${detail}` : ""}`);
    if (r.out.vary) console.log(`~     ${label} — ${r.out.vary}`);
  } else if (r.skip) {
    skipped += 1;
    console.log(`skip  ${label} — ${r.message}`);
  } else {
    failed += 1;
    console.log(`FAIL  ${label} — ${r.message}`);
  }
  return r.ok;
}

/** The helpers every step uses, installed once per page. */
const HELPERS = (base) => `(async () => {
  const t = window.__smoke = {
    K: await import("${base}/soar/constants.js"),
    blobs: [], downloads: [], errors: [], trackContext: null,
    sleep: (ms) => new Promise((r) => setTimeout(r, ms)),
    async until(test, ms, what) {
      const t0 = performance.now();
      for (;;) {
        const v = test();
        if (v) return v;
        if (performance.now() - t0 > ms) {
          throw new Error("waited " + ms + " ms for " + what);
        }
        await t.sleep(50);
      }
    },
    expect(cond, what) { if (!cond) throw new Error("expected " + what); },
    skip(why) { throw Object.assign(new Error(why), { smokeSkip: true }); },
    key(type, key) {
      document.dispatchEvent(new KeyboardEvent(
        type, { key, bubbles: true, cancelable: true }));
    },
    press(key) { t.key("keydown", key); t.key("keyup", key); },
    button(text) {
      const all = [...document.querySelectorAll("#menu button")];
      const b = all.find((x) => x.textContent.trim().startsWith(text));
      if (!b) {
        throw new Error("a menu button starting '" + text + "'; the menu has: "
          + all.map((x) => x.textContent.trim()).join(" | "));
      }
      return b;
    },
    toast: () => document.getElementById("toast").textContent,
    // Everything loaded and compiled, the auto-tier probe finished.
    settled: () => {
      const s = window.soar;
      return s?.scene && s.renderer && !s._probe
        && (s.renderer._pendingPipelines?.size ?? 0) === 0
        && !s.renderer.lightBakePending;
    },
  };
  // Downloads stop at the Blob.
  const createObjectURL = URL.createObjectURL.bind(URL);
  URL.createObjectURL = (blob) => { t.blobs.push(blob); return createObjectURL(blob); };
  const click = HTMLAnchorElement.prototype.click;
  HTMLAnchorElement.prototype.click = function () {
    if (this.download) { t.downloads.push(this.download); return; }
    click.call(this);
  };
  // Anything the page complains about while the steps run.
  window.addEventListener("error", (e) => t.errors.push("error: " + e.message));
  window.addEventListener("unhandledrejection",
    (e) => t.errors.push("unhandled rejection: " + (e.reason?.message ?? e.reason)));
  const consoleError = console.error.bind(console);
  console.error = (...a) => { t.errors.push("console.error: " + a.join(" ")); consoleError(...a); };
  // The track panel's context (the recorded samples) is only ever handed to
  // ui.open; remember it so the panel can be opened a second time.
  const ui = window.soar.ui;
  const open = ui.open.bind(ui);
  ui.open = (name, context) => {
    if (name === "track") t.trackContext = context;
    return open(name, context);
  };
  return true;
})()`;

async function loadScene(name, server) {
  const sc = SCENES[name];
  await tab.goto(`${server.base}/soar/`);
  await tab.eval(`localStorage.clear();
    localStorage.setItem("soar.mode", ${JSON.stringify(sc.mode)});
    localStorage.setItem("soar.tutorialDone", "1");
    localStorage.setItem("soar.tutorialResearchDone", "1"); true`);
  if (sc.demo) {
    await tab.goto(`${server.base}/soar/?demo=${sc.demo}`);
  } else {
    await tab.goto(`${server.base}/soar/`);
    await tab.waitFor(`!document.getElementById("capability").innerText.startsWith("Checking")`);
    await tab.eval(`document.getElementById("fly-less").click(); true`);
  }
  await tab.waitFor(`(() => { const L = document.getElementById("loading");
    if (L.classList.contains("failed") && !L.hidden) throw new Error(L.innerText);
    return window.soar?.scene && L.hidden; })()`, 180000);
  await tab.eval(HELPERS(server.base));
}

async function runScene(name, server, dataBase) {
  try {
    await loadScene(name, server);
  } catch (err) {
    failed += 1;
    console.log(`FAIL  ${name}: load — ${err.message}`);
    return;
  }

  await step(name, "load", `
    await t.until(t.settled, 120000, "the probe, the pipelines and the light bake");
    t.expect(!s.paused && !s.stop, "an unpaused, running viewer");
    t.expect(s.frameIndex > 0, "frames to have been drawn");
    return { detail: "source '" + s.scene.sourceName + "', surface "
               + (s.scene.city ? "city" : "ocean") + ", mode " + s.ui.mode
               + ", flyers " + s.availableFlyers.join("+")
               + ", minimap " + Boolean(s.minimap),
             vary: "the probe chose '" + s.qualityTier + "' after "
               + s.frameIndex + " frames" };`);

  await step(name, "fly with W", `
    const p0 = [...s.camera.position];
    t.key("keydown", "w");
    t.expect(s.camera.keys.has("w"), "the camera to be holding W");
    await t.until(() => Math.hypot(...s.camera.position.map((v, i) => v - p0[i])) > 1.0,
                  15000, "the camera to move 1 m");
    t.expect(s.renderer.cameraMoving, "the renderer to know the camera is moving");
    t.key("keyup", "w");
    t.expect(!s.camera.keys.has("w"), "keyup to release W");
    return "camera moved";`);

  await step(name, "pick Low in the Quality panel", `
    s.pause("quality");
    t.expect(s.paused && s.ui.panel === "quality", "the Quality panel, paused");
    t.button("Low").click();
    t.expect(s.qualityTier === "low", "viewer tier low, got " + s.qualityTier);
    t.expect(s.renderer.qualityTier === "low", "renderer tier low, got " + s.renderer.qualityTier);
    t.expect(s.autoTier === false, "auto tier off after a hand pick");
    t.expect(s._probe === null && s._governor === null, "no probe or governor");
    return "tier low, auto off";`);

  await step(name, "Escape backs out of the menu", `
    t.expect(s.ui.panel === "quality", "to start on the Quality panel");
    t.key("keydown", "Escape");
    t.expect(s.ui.panel === "main", "Escape to go back to the main panel, got " + s.ui.panel);
    await t.sleep(450);
    t.key("keydown", "Escape");
    t.expect(!s.ui.isOpen && !s.paused, "the second Escape to resume");
    await t.sleep(450);
    t.key("keydown", "Escape");
    t.expect(s.paused && s.ui.panel === "main", "Escape in flight to pause on the main panel");
    s.resume({ capture: false });
    t.expect(!s.paused && !s.ui.isOpen, "resume to close the menu");
    return "quality -> main -> flying -> main -> flying";`);

  await step(name, "park and converge", `
    await t.until(() => s.renderer.converged, 120000, "a parked view to converge");
    const need = K.PARKED_ACCUM_FRAMES_BY_TIER.low;
    t.expect(s.renderer.accumCount >= need,
             "at least " + need + " accumulated samples, got " + s.renderer.accumCount);
    t.expect(!s.renderer.cameraMoving, "a still camera");
    return { detail: "converged at >= " + need + " spp",
             vary: s.renderer.accumCount + " spp, hold rung "
               + (s.renderer.holdRung + 1) + "/" + s.renderer.holdRungCount };`);

  await step(name, "M cycles the minimap", `
    if (!s.minimap) t.skip("no minimap was built for this field");
    const seen = [s.minimapMode];
    for (let i = 0; i < 3; i++) { t.press("m"); seen.push(s.minimapMode); }
    t.expect(seen.join(",") === "corner,full,off,corner",
             "corner,full,off,corner, got " + seen.join(","));
    return seen.join(" -> ");`);

  await step(name, "B cycles the flyer; with none the loop sleeps, a key wakes it", `
    const seen = [s.flyerLabel];
    for (let i = 0; i < 2; i++) { t.press("b"); seen.push(s.flyerLabel); }
    t.expect(seen.join(",") === "paper dart,swift,off",
             "paper dart,swift,off, got " + seen.join(","));
    t.expect(!s.flyerEnabled, "the flyer disabled");
    await t.until(() => s._sleeping, 120000, "the loop to sleep on a converged view");
    const frame = s.frameIndex;
    await t.sleep(300);
    t.expect(s.frameIndex === frame, "no frames while asleep");
    t.press("b");
    seen.push(s.flyerLabel);
    t.expect(!s._sleeping, "B to wake the loop");
    t.expect(s.flyerLabel === "paper dart", "the dart back, got " + s.flyerLabel);
    await t.until(() => s.frameIndex > frame, 15000, "frames after the wake");
    return seen.join(" -> ");`);

  await step(name, "move the sun", `
    const before = s.sunElevation;
    s.setSun({ elevation: 35, azimuth: 200 });
    t.expect(s.sunElevation === 35 && s.sunAzimuth === 200, "sun at 35/200");
    t.expect(s.renderer.lightBakePending, "a new light bake to be pending");
    await t.until(() => !s.renderer.lightBakePending, 120000, "the light bake");
    await t.until(() => s.renderer.converged, 120000, "the view to converge again");
    return "elevation " + before + " -> 35, bake finished, converged";`);

  await step(name, "K flips the light cache", `
    if (!s.ui.allows("quality")) t.skip("mode '" + s.ui.mode + "' has no quality controls, so K does nothing");
    const cache = s.lightCacheOn;
    t.press("k");
    t.expect(s.lightCacheOn === !cache, "K to flip the light cache");
    t.expect(s.qualityCustom, "the quality marked Custom");
    await t.until(() => (s.renderer._pendingPipelines?.size ?? 0) === 0, 120000, "the new shader variant");
    await t.until(() => s.renderer.converged, 120000, "the view to converge on it");
    s.setQualityTier("low");
    t.expect(!s.qualityCustom && s.lightCacheOn === cache,
             "picking a tier to put it back");
    return "cache " + cache + " -> " + !cache + ", reset by tier";`);

  await step(name, "toggle the periodic domain", `
    const was = s.renderer.periodic;
    s.togglePeriodic();
    t.expect(s.renderer.periodic === !was && s.camera.periodic === !was, "periodic flipped on renderer and camera");
    await t.until(t.settled, 120000, "the pipelines and the bake");
    s.togglePeriodic();
    t.expect(s.renderer.periodic === was, "periodic flipped back");
    await t.until(t.settled, 120000, "the pipelines and the bake");
    return "periodic " + was + " -> " + !was + " -> " + was;`);

  await step(name, "open every pause-menu panel", `
    s.pause();
    const keys = [...document.querySelectorAll("#menu [data-menu-key]")].map((n) => n.dataset.menuKey);
    const out = ["main rows: " + keys.join(",")];
    for (const panel of ["quality", "sun", "controls", "terminal", "capture", "main"]) {
      s.ui.open(panel);
      t.expect(s.ui.panel === panel && s.ui.isOpen, "panel '" + panel + "' open");
      const text = document.getElementById("menu").textContent;
      t.expect(!text.includes("This panel failed to build"), "panel '" + panel + "' to build: " + text);
      out.push(panel + " " + document.querySelectorAll("#menu button").length + " buttons, "
               + document.querySelectorAll("#menu input").length + " sliders");
    }
    s.ui.open("quality");
    t.button("Advanced").click();
    out.push("quality advanced " + document.querySelectorAll("#menu button").length + " buttons, "
             + document.querySelectorAll("#menu input").length + " sliders");
    t.button("Advanced").click();
    s.ui.close();
    t.expect(!s.ui.isOpen && s.paused, "close to leave the viewer paused with no menu");
    s.resume({ capture: false });
    return out.join("; ");`);

  // A fixed view for what gets printed: the scene's own start pose, a hand
  // exposure, and the scene clock at zero.
  await step(name, "terminal commands", `
    s.camera.reset();
    s.setAutoExposure(false);
    s.setExposure(2.0);
    s.pause();
    s.sceneTime = 0;
    s.ui.open("terminal");
    const box = () => [...document.querySelectorAll("#menu .row")].at(-1).textContent;
    const witness = box();
    t.expect(witness === s.witnessCommand(), "the panel to show witnessCommand()");
    t.expect(witness.includes("witness"), "a witness command, got " + witness);
    t.button("behold").click();
    const behold = box();
    t.expect(s.terminalRenderer === "behold" && behold === s.beholdCommand(), "the panel to show beholdCommand()");
    t.button("witness").click();
    t.expect(s.terminalRenderer === "witness", "witness selected again");
    return "\\n        " + witness + "\\n        " + behold;`);

  await step(name, "save a still", `
    s.captureSize = [320, 180];
    s.ui.open("capture");
    t.button("High").click();
    t.expect(s.captureStillTier === "high", "the still tier high");
    const n = t.blobs.length;
    t.button("Save a still").click();
    t.expect(s._capturing, "a capture to be running");
    await t.until(() => !s._capturing, 240000, "the still to finish");
    t.expect(t.blobs.length === n + 1, "one download; the toast says: " + t.toast());
    const blob = t.blobs.at(-1);
    t.expect(blob.type === "image/png", "a PNG blob, got " + blob.type);
    t.expect(/^cloudyview_soar_.*\\.png$/.test(t.downloads.at(-1)), "a .png filename, got " + t.downloads.at(-1));
    const bytes = new Uint8Array(await blob.arrayBuffer());
    t.expect(bytes[0] === 0x89 && bytes[1] === 0x50 && bytes[2] === 0x4e && bytes[3] === 0x47, "the PNG signature");
    const bitmap = await createImageBitmap(blob);
    t.expect(bitmap.width === 320 && bitmap.height === 180, "320x180 pixels, got " + bitmap.width + "x" + bitmap.height);
    const canvas = new OffscreenCanvas(320, 180);
    const ctx = canvas.getContext("2d");
    ctx.drawImage(bitmap, 0, 0);
    const px = ctx.getImageData(0, 0, 320, 180).data;
    let sum = 0;
    for (let i = 0; i < px.length; i += 4) sum += px[i] + px[i + 1] + px[i + 2];
    t.expect(sum > 0, "a picture that is not black");
    // The tEXt chunk sits right after IHDR (capture.js embedPngText).
    const dv = new DataView(bytes.buffer);
    const length = dv.getUint32(33);
    const text = new TextDecoder().decode(bytes.subarray(41, 41 + length));
    const key = "cloudyview.render_metadata";
    t.expect(text.startsWith(key + "\\0"), "the render metadata chunk");
    const meta = JSON.parse(text.slice(key.length + 1));
    delete meta.timestamp;
    t.expect(t.toast() === "Saved a 320x180 still.", "the saved toast, got: " + t.toast());
    t.expect(!s.ui.isOpen && s.paused, "the menu closed, still paused");
    return { detail: "PNG 320x180 with metadata\\n        " + JSON.stringify(meta),
             vary: bytes.length + " bytes, mean " + (sum / (px.length / 4) / 3).toFixed(2) };`,
    300000);

  await step(name, "record a track with R", `
    s.resume({ capture: false });
    t.press("r");
    t.expect(s.recorder.recording, "R to start recording");
    t.key("keydown", "w");
    await t.until(() => s.recorder.duration >= 0.5, 30000, "half a second of track");
    t.key("keyup", "w");
    t.press("r");
    t.expect(!s.recorder.recording, "R to stop recording");
    t.expect(s.paused && s.ui.panel === "track", "the track panel, paused; got panel " + s.ui.panel);
    const { samples } = t.trackContext;
    t.expect(samples.length >= 2, "at least two samples");
    t.expect(samples.every((row) => row.length === 10), "ten columns per sample");
    return { detail: "track panel open, 10 columns per sample",
             vary: samples.length + " samples over " + samples.at(-1)[0].toFixed(2) + " s" };`);

  await step(name, "render the track to video", `
    t.expect(s.ui.panel === "track", "the track panel");
    const { videoCapabilities } = await import(new URL("video.js", location.href));
    const caps = videoCapabilities();
    if (!caps.available) t.skip("no WebCodecs here: " + caps.why);
    s.captureSize = [320, 180];
    s.captureVideoTier = "minimal";
    s.videoFps = K.VIDEO_FPS_LIMITS[0];
    const frames = s.trackFrameCount(t.trackContext.samples);
    const n = t.blobs.length;
    t.button("Render it to video").click();
    await t.until(() => s._capturing, 15000, "the video render to start");
    await t.until(() => !s._capturing, 600000, "the video render to finish");
    t.expect(t.blobs.length === n + 1, "one download; the toast says: " + t.toast());
    const blob = t.blobs.at(-1);
    t.expect(blob.type.startsWith("video/"), "a video blob, got '" + blob.type + "'");
    t.expect(blob.size > 0, "a video with bytes in it");
    t.expect(t.toast().startsWith("Saved " + frames + " frames"), "the saved toast, got: " + t.toast());
    t.expect(s.renderer.qualityTier === "low", "the live tier restored, got " + s.renderer.qualityTier);
    t.expect(s.renderer.boundedQueue === false, "the renderer out of offline mode");
    return { detail: "video rendered, live tier restored",
             vary: frames + " frames at " + s.videoFps + " fps, " + blob.size + " bytes, " + blob.type
               + ", " + t.downloads.at(-1).split(".").pop() };`,
    660000);

  await step(name, "save the track", `
    const context = t.trackContext;
    s.ui.open("track", context);
    const n = t.blobs.length;
    t.button("Save the track").click();
    t.expect(s.ui.panel === "main", "the main panel after saving");
    await t.until(() => t.blobs.length > n, 120000, "the track download; the toast says: " + t.toast());
    const blob = t.blobs.at(-1);
    t.expect(blob.type === "application/json", "a JSON blob, got " + blob.type);
    const track = JSON.parse(await blob.text());
    t.expect(track.samples.length === context.samples.length, "every sample in the file");
    t.expect(track.header.render.tier === "minimal", "the header to describe the video tier");
    t.expect(JSON.stringify(track.header.render.size) === "[320,180]", "the header to carry the video size");
    return "schema " + track.schema + ", keys " + Object.keys(track).join(",")
      + ", header " + Object.keys(track.header).join(",")
      + ", render " + Object.keys(track.header.render).join(",");`);

  await step(name, "I toggles ice mode", `
    s.ui.close();
    s.resume({ capture: false });
    if (!s.ui.allows("ice")) t.skip("mode '" + s.ui.mode + "' has no ice control, so I does nothing");
    if (!s.scene.iceAvailable) t.skip("this field has no ice: " + s.scene.iceNote);
    t.press("i");
    await t.until(() => s.iceModeOn, 120000, "ice mode to come on; the toast says: " + t.toast());
    t.expect(s.renderer.iceMode === true, "the renderer in ice mode");
    t.expect(Boolean(s.scene.iceTexture), "the ice volume attached");
    await t.until(t.settled, 120000, "the pipelines");
    t.press("i");
    t.expect(!s.iceModeOn && s.renderer.iceMode === false, "I to turn it off again");
    return "on (volume read), then off";`);

  // A click the browser itself delivers, which is what the lock asks for.
  for (const type of ["mousePressed", "mouseReleased"]) {
    await tab.send("Input.dispatchMouseEvent",
                   { type, x: 200, y: 200, button: "left", clickCount: 1 });
  }
  await step(name, "pointer lock, mouse look, Escape", `
    await t.sleep(1000);
    if (!s.captured) t.skip("this Chrome did not grant the pointer lock to a DevTools click");
    const az = s.camera.azimuth;
    document.dispatchEvent(new MouseEvent("mousemove", { movementX: 0, movementY: 0 }));
    document.dispatchEvent(new MouseEvent("mousemove", { movementX: 40, movementY: 0 }));
    t.expect(s.camera.azimuth !== az, "mouse movement to turn the camera");
    document.exitPointerLock();
    await t.until(() => !s.captured, 5000, "the lock to drop");
    t.expect(s.paused && s.ui.panel === "main", "losing the lock to pause");
    s.resume({ capture: false });
    return "locked, looked, lock loss paused";`);

  if (name === "day") {
    await step(name, `open ${NC_NAME} through the file picker`, `
      const response = await fetch("${dataBase}/${NC_NAME}");
      t.expect(response.ok, "the test file to be served");
      const file = new File([await response.blob()], "${NC_NAME}");
      window.showOpenFilePicker = async () => [{ getFile: async () => file }];
      const old = s.scene;
      const questions = [];
      const picked = s.pickFile();
      // This file has liquid water (QC) and no ice variable, so the loader
      // asks one question, "Which variable is ice?", answered "No ice". Any
      // other question would wait for ever, so it is reported instead.
      const asking = new Set(["groups", "units", "coordUnits", "gridSpacing", "axes", "variable", "timestep"]);
      let done = false, failure = null;
      picked.then(() => { done = true; }, (err) => { failure = err; done = true; });
      await t.until(() => {
        const panel = s.ui.panel;
        const text = document.getElementById("menu").textContent;
        if (panel === "variable" && text.includes("Which variable is ice?")) {
          questions.push("variable (ice): No ice");
          t.button("No ice").click();
        } else if (asking.has(panel)) {
          throw new Error("only the ice question, but the load opened the '" + panel + "' panel: "
            + text.slice(0, 300));
        }
        return done;
      }, 240000, "the file to load");
      t.expect(questions.length === 1, "one question (ice), got " + questions.length);
      if (failure) throw failure;
      const L = document.getElementById("loading");
      t.expect(!(L.classList.contains("failed") && !L.hidden), "no failure panel: " + L.innerText);
      t.expect(s.scene && s.scene !== old, "a new scene");
      t.expect(s.scene.sourceName === "${NC_NAME}", "the file as the source, got " + s.scene.sourceName);
      t.expect(!s.paused, "flying again after the load");
      await t.until(t.settled, 120000, "the pipelines and the light bake");
      const frame = s.frameIndex;
      await t.until(() => s.frameIndex > frame, 15000, "frames of the new field");
      t.expect(s.iceModeOn === false, "ice mode off on a new field");
      return "asked " + questions.join(", ") + "; loaded; volume " + (s.scene.dims ?? s.scene.shape ?? s.scene.volumeShape ?? []).join("x")
        + ", ice available " + s.scene.iceAvailable + ", nested " + Boolean(s.scene.nested)
        + "\\n        " + s.witnessCommand({ size: [320, 180] }).replace(/--time \\S+/, "--time T");`,
      300000);
  }

  await step(name, "nothing went wrong along the way", `
    const L = document.getElementById("loading");
    t.expect(!s.stop, "the renderer still running");
    t.expect(!(L.classList.contains("failed") && !L.hidden), "no failure panel: " + L.innerText);
    t.expect(t.errors.length === 0, "no page errors, got: " + t.errors.join(" || "));
    return "no page errors, no failure panel";`);
}

// The test NetCDF sits outside web/, so it is served from its own port, with
// the header a cross-origin fetch needs.
const dataServer = createServer(async (req, res) => {
  try {
    if (req.url !== `/${NC_NAME}`) throw new Error("not served");
    const body = await readFile(join(repo, "data", NC_NAME));
    res.writeHead(200, { "Content-Type": "application/x-netcdf",
                         "Access-Control-Allow-Origin": "*" });
    res.end(body);
  } catch {
    res.writeHead(404, { "Access-Control-Allow-Origin": "*" });
    res.end();
  }
});
await new Promise((ok) => dataServer.listen(0, "127.0.0.1", ok));
const dataBase = `http://127.0.0.1:${dataServer.address().port}`;

const server = await serveTree(webDir);
const done = async (code) => {
  await tab?.close();
  server.close();
  dataServer.close();
  process.exit(code);
};
try {
  tab = await openTab(cdpPort);
  await tab.goto(`${server.base}/`);
  const adapter = await requireHardwareAdapter(tab);
  console.log(`~     tree ${server.root} (${adapter})`);
  for (const name of scenes) {
    if (!SCENES[name]) throw new Error(`unknown scene '${name}' (day, city)`);
    await runScene(name, server, dataBase);
  }
  console.log(`browser_smoke: ${passed} ok, ${skipped} skipped, ${failed} failed`);
  await done(failed ? 1 : 0);
} catch (err) {
  console.error(`browser_smoke: ${err.message}`);
  await done(err.code ?? 2);
}
