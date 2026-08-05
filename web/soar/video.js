// Frame-exact video, encoded in the tab.
//
// The offline re-render is the whole reason this file exists. A recorded
// track is replayed at an exact frame rate with many accumulated passes per
// frame, so one frame can take seconds of wall clock; screen capture and
// MediaRecorder both timestamp from the clock on the wall and would turn a
// 30-second flight into a twenty-minute slideshow. WebCodecs' `VideoEncoder`
// takes the timestamp the caller hands it, so frame n is at exactly n/fps
// seconds no matter how long it took to converge. Mediabunny muxes the
// resulting chunks into a real container in memory.
//
// A frame ZIP is not a fallback. It is not a video (Thomas, 2026-08-04). If
// nothing here can encode, this module throws and says why.
//
// Public API
// ----------
//   videoCapabilities()
//       -> { available, why, hasVideoEncoder, hasVideoFrame, candidates }
//       Synchronous, no side effects, no encoder is constructed. `why` is
//       null when available.
//
//   await chooseCodec({ width, height, fps, bitrate })
//       -> { codec, container, fileExtension, encoderConfig,
//            decoderConfig, colorSpace, colorSpaceWarning, trialBytes,
//            frameSource }
//       Trial-encodes one real frame at the real size and only returns a
//       config that actually produced a chunk. Throws if none did.
//
//   new VideoWriter({ width, height, fps, bitrate, chosen, keyFrameSeconds })
//       .init()                     -> the chosen config (await)
//       .addFrame(source, index)    -> await; source is ImageData, a
//                                      VideoFrame, or any CanvasImageSource
//       .finish()                   -> await; a Blob
//       .abort()                    -> await; discards everything
//       Read-only after init: .codec .container .fileExtension .mimeType
//       .width .height .fps .frameCount .chunkCount .encodedBytes .warnings
//
//   evenSize([w, h]) -> [w, h] rounded down to even
//       H.264 has no odd dimensions, anywhere. VideoWriter refuses odd sizes
//       rather than quietly cropping a pixel off the user's capture; call
//       this first if you want the snap to happen.

"use strict";

import {
  BufferTarget,
  EncodedPacket,
  EncodedVideoPacketSource,
  Mp4OutputFormat,
  Output,
  WebMOutputFormat,
} from "./vendor/mediabunny/mediabunny.min.mjs";

// Tried in order. H.264 first because it is what Chrome and Safari have and
// what every editor, phone and web page will play back without argument.
// Firefox has no H.264 VideoEncoder at all (bugzilla 2049470 — the decoder is
// there, the encoder is not), so it falls through to VP9. VP8 is last: it is
// a worse picture, but it is a video, and a video is the deliverable.
const CANDIDATES = [
  { codec: "avc1.640034", family: "avc", label: "H.264 High 5.2" },
  { codec: "avc1.640028", family: "avc", label: "H.264 High 4.0" },
  { codec: "avc1.4d0034", family: "avc", label: "H.264 Main 5.2" },
  { codec: "avc1.42003c", family: "avc", label: "H.264 Baseline 6.0" },
  { codec: "vp09.00.10.08", family: "vp9", label: "VP9 profile 0" },
  { codec: "av01.0.08M.08", family: "av1", label: "AV1 Main 8-bit" },
  { codec: "vp8", family: "vp8", label: "VP8" },
];

// Mediabunny writes the MP4 `colr` box and the Matroska colour elements from
// enumerated names and has no entry for anything else; an unrecognised name
// would be written as a NaN-shaped hole in the container. These are its
// tables (mediabunny 1.52.3), duplicated here so we can check before handing
// a colour space over rather than after.
const KNOWN_PRIMARIES = ["bt709", "bt470bg", "smpte170m", "bt2020", "smpte432"];
const KNOWN_TRANSFER =
  ["bt709", "smpte170m", "linear", "iec61966-2-1", "pq", "hlg"];
const KNOWN_MATRIX = ["rgb", "bt709", "bt470bg", "smpte170m", "bt2020-ncl"];

// How far the encoder is allowed to fall behind before addFrame waits. The
// caller renders one slow frame at a time so this should never bind, but an
// unbounded queue on a 4K track is a way to run a tab out of memory without
// ever being told why.
const MAX_QUEUE_DEPTH = 8;

/** Round a capture size down to even, which H.264 requires everywhere. */
export function evenSize([w, h]) {
  return [w & ~1, h & ~1];
}

/**
 * What can this browser encode, asked without touching anything.
 *
 * Deliberately not a verdict: `available: true` means the APIs exist, not
 * that a codec works. Only chooseCodec knows that, and it knows it because
 * it encoded something.
 */
export function videoCapabilities() {
  const hasVideoEncoder = typeof VideoEncoder !== "undefined";
  const hasVideoFrame = typeof VideoFrame !== "undefined";
  let why = null;
  if (!hasVideoEncoder) {
    why = "This browser has no WebCodecs VideoEncoder, so it cannot write a " +
      "video with the exact frame timing an offline render needs. Chrome 94+, " +
      "Edge 94+, Safari 16.4+ and Firefox 130+ can.";
  } else if (!hasVideoFrame) {
    why = "This browser has VideoEncoder but no VideoFrame, so there is no " +
      "way to hand it a picture.";
  }
  return {
    available: why === null,
    why,
    hasVideoEncoder,
    hasVideoFrame,
    candidates: CANDIDATES.map((c) => ({ codec: c.codec, label: c.label })),
  };
}

/** A default bitrate: generous, because clouds are gradients and dither. */
function defaultBitrate(width, height, fps) {
  const bits = width * height * fps * 0.15;
  return Math.round(Math.min(80_000_000, Math.max(4_000_000, bits)));
}

function checkSize(width, height) {
  if (!Number.isInteger(width) || !Number.isInteger(height) ||
      width < 2 || height < 2) {
    throw new Error(
      `Video size must be two integers of at least 2; got ${width}x${height}.`);
  }
  if ((width & 1) || (height & 1)) {
    throw new Error(
      `H.264 cannot encode ${width}x${height} — both dimensions must be even, ` +
      `and ${(width & 1) ? `width ${width}` : `height ${height}`} is odd. ` +
      `Use ${width & ~1}x${height & ~1} instead.`);
  }
}

function checkFps(fps) {
  if (!Number.isFinite(fps) || fps <= 0 || fps > 240) {
    throw new Error(`Frame rate must be between 0 and 240; got ${fps}.`);
  }
}

function encoderConfigFor(candidate, width, height, fps, bitrate) {
  const config = {
    codec: candidate.codec,
    width, height,
    bitrate,
    framerate: fps,
    bitrateMode: "variable",
    // Every frame here cost seconds to render; spending milliseconds to
    // encode it well is free by comparison.
    latencyMode: "quality",
  };
  if (candidate.family === "avc") {
    // AVCC with an out-of-band description, which is what the MP4 muxer
    // wants. The Annex B path exists but makes the muxer re-derive the
    // parameter sets from the bitstream, and that is a second thing to be
    // wrong about.
    config.avc = { format: "avc" };
  }
  return config;
}

// --- frame sources --------------------------------------------------------

/**
 * ImageData is not a CanvasImageSource, so it cannot be passed to the
 * VideoFrame(image, init) constructor. The buffer constructor takes it
 * directly and skips a copy through a canvas; where that is not implemented
 * we stage through a 2D canvas instead. Same pixels either way — this is a
 * plumbing choice, not a quality one — but which path was taken is reported
 * so it is never a mystery.
 */
function makeImageDataConverter(width, height) {
  if (typeof VideoFrame === "undefined") {
    throw new Error("This browser has no VideoFrame.");
  }
  let probe = null;
  try {
    probe = new VideoFrame(new Uint8Array(2 * 2 * 4), {
      format: "RGBA", codedWidth: 2, codedHeight: 2, timestamp: 0,
    });
  } catch {
    probe = null;
  }
  if (probe) {
    probe.close();
    return {
      path: "buffer-rgba",
      convert(image, timestamp, duration) {
        return new VideoFrame(image.data, {
          format: "RGBA",
          codedWidth: image.width,
          codedHeight: image.height,
          timestamp, duration,
        });
      },
      dispose() {},
    };
  }
  const canvas = typeof OffscreenCanvas !== "undefined"
    ? new OffscreenCanvas(width, height)
    : Object.assign(document.createElement("canvas"), { width, height });
  const ctx = canvas.getContext("2d", { willReadFrequently: false });
  if (!ctx) {
    throw new Error(
      "This browser accepts neither an RGBA buffer nor a 2D canvas as a " +
      "VideoFrame source, so there is no way to hand it a rendered frame.");
  }
  return {
    path: "canvas-2d",
    convert(image, timestamp, duration) {
      ctx.putImageData(image, 0, 0);
      return new VideoFrame(canvas, { timestamp, duration });
    },
    dispose() {},
  };
}

/** A frame the trial encoder can compress but not trivially: a gradient. */
function syntheticImage(width, height) {
  const image = new ImageData(width, height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4;
      image.data[i] = (x * 255 / width) | 0;
      image.data[i + 1] = (y * 255 / height) | 0;
      image.data[i + 2] = ((x ^ y) & 0xff);
      image.data[i + 3] = 255;
    }
  }
  return image;
}

// --- colour ---------------------------------------------------------------

/**
 * Whatever the encoder says it produced is what goes in the container.
 *
 * Chrome's H.264 encoder matrixes RGB to YUV as BT.601 limited range no
 * matter what the source frame was tagged, and reports that honestly in
 * `decoderConfig.colorSpace`. Writing BT.709 into the container anyway is
 * how you get a video that is visibly washed out on one player and
 * over-saturated on the next. So: propagate, never assume.
 */
function vetColorSpace(colorSpace) {
  if (!colorSpace) {
    return { colorSpace: null, warning:
      "The encoder reported no colour space, so no colour tag is written " +
      "and players will guess (usually BT.709 above SD, BT.601 below)." };
  }
  const { primaries, transfer, matrix, fullRange } = colorSpace;
  const unknown = [];
  if (primaries && !KNOWN_PRIMARIES.includes(primaries)) {
    unknown.push(`primaries "${primaries}"`);
  }
  if (transfer && !KNOWN_TRANSFER.includes(transfer)) {
    unknown.push(`transfer "${transfer}"`);
  }
  if (matrix && !KNOWN_MATRIX.includes(matrix)) {
    unknown.push(`matrix "${matrix}"`);
  }
  if (unknown.length) {
    return { colorSpace: null, warning:
      `The encoder reported ${unknown.join(", ")}, which mediabunny 1.52.3 ` +
      "cannot write into a container. The colour tag is omitted rather than " +
      "written wrong; the picture is correct, players will guess the matrix." };
  }
  if (!primaries || !transfer || !matrix || fullRange === undefined) {
    const missing = [
      primaries ? null : "primaries", transfer ? null : "transfer",
      matrix ? null : "matrix", fullRange === undefined ? "fullRange" : null,
    ].filter(Boolean);
    return { colorSpace: null, warning:
      `The encoder left ${missing.join(", ")} unspecified, and a colour tag ` +
      "is all-or-nothing in both MP4 and Matroska, so none is written." };
  }
  return { colorSpace: { primaries, transfer, matrix, fullRange }, warning: null };
}

/** Replace meta.decoderConfig.colorSpace with the vetted one (or drop it). */
function vetMeta(meta) {
  const decoderConfig = meta?.decoderConfig;
  if (!decoderConfig) {
    throw new Error(
      "The encoder produced a chunk with no decoder configuration, so there " +
      "is nothing to describe the track with and the file would not play.");
  }
  const { colorSpace, warning } = vetColorSpace(decoderConfig.colorSpace);
  const out = { ...meta, decoderConfig: { ...decoderConfig } };
  if (colorSpace) out.decoderConfig.colorSpace = colorSpace;
  else delete out.decoderConfig.colorSpace;
  return { meta: out, colorSpace, warning };
}

// --- trial encode ---------------------------------------------------------

/**
 * Encode one real frame with each candidate and return the first that gave
 * back a chunk.
 *
 * Not `isConfigSupported`, which lies in both directions — it has reported
 * H.264 supported on builds with no encoder, and reported nothing at all on
 * builds that encode fine. Not the user agent, which is a string anyone can
 * set. The only question worth asking a codec is whether it just produced
 * bytes, so ask it that.
 */
export async function chooseCodec({ width, height, fps, bitrate } = {}) {
  checkSize(width, height);
  checkFps(fps);
  const caps = videoCapabilities();
  if (!caps.available) throw new Error(caps.why);

  const rate = bitrate ?? defaultBitrate(width, height, fps);
  const converter = makeImageDataConverter(width, height);
  const image = syntheticImage(width, height);
  const rejected = [];

  try {
    for (const candidate of CANDIDATES) {
      const config = encoderConfigFor(candidate, width, height, fps, rate);
      const chunks = [];
      let meta = null;
      let failure = null;
      let encoder = null;
      try {
        encoder = new VideoEncoder({
          output: (chunk, chunkMeta) => {
            chunks.push(chunk.byteLength);
            if (chunkMeta?.decoderConfig && !meta) meta = chunkMeta;
          },
          error: (e) => { failure = e; },
        });
        encoder.configure(config);
        const frame = converter.convert(image, 0, Math.round(1e6 / fps));
        try {
          encoder.encode(frame, { keyFrame: true });
        } finally {
          frame.close();
        }
        await encoder.flush();
      } catch (e) {
        failure = failure ?? e;
      } finally {
        try { if (encoder && encoder.state !== "closed") encoder.close(); }
        catch { /* already gone */ }
      }

      if (failure) {
        rejected.push(`${candidate.label} (${candidate.codec}): ` +
                      `${failure.message ?? failure}`);
        continue;
      }
      if (chunks.length === 0 || chunks[0] === 0) {
        rejected.push(`${candidate.label} (${candidate.codec}): configured ` +
                      "without complaint but produced no bytes");
        continue;
      }
      if (!meta?.decoderConfig) {
        rejected.push(`${candidate.label} (${candidate.codec}): produced ` +
                      `${chunks[0]} bytes but no decoder configuration`);
        continue;
      }
      if (candidate.family === "avc" && !meta.decoderConfig.description) {
        // Without the AVCDecoderConfigurationRecord the MP4 muxer has to dig
        // the parameter sets out of an Annex B bitstream we did not ask for.
        rejected.push(`${candidate.label} (${candidate.codec}): gave no AVCC ` +
                      "description despite avc.format = \"avc\"");
        continue;
      }

      const vetted = vetMeta(meta);
      const container = candidate.family === "avc" ? "mp4" : "webm";
      return {
        codec: candidate.codec,
        codecFamily: candidate.family,
        label: candidate.label,
        container,
        fileExtension: container === "mp4" ? ".mp4" : ".webm",
        encoderConfig: config,
        decoderConfig: vetted.meta.decoderConfig,
        colorSpace: vetted.colorSpace,
        colorSpaceWarning: vetted.warning,
        trialBytes: chunks[0],
        frameSource: converter.path,
        rejected,
      };
    }
  } finally {
    converter.dispose();
  }

  throw new Error(
    `No codec in this browser could encode a ${width}x${height} frame at ` +
    `${fps} fps. Tried: ${rejected.join("; ")}. Firefox has no H.264 ` +
    "VideoEncoder (bugzilla 2049470) and needs VP9; if VP9 was refused too, " +
    "record in Chrome, Edge or Safari 16.4+, or try a smaller capture size.");
}

// --- the writer -----------------------------------------------------------

export class VideoWriter {
  /**
   * @param {object} options
   * @param {number} options.width   even, pixels
   * @param {number} options.height  even, pixels
   * @param {number} options.fps     the *output* frame rate, not the render rate
   * @param {number} [options.bitrate]
   * @param {object} [options.chosen]  a chooseCodec result, to skip the trial
   * @param {number} [options.keyFrameSeconds=2]  keyframe spacing, for seeking
   */
  constructor({ width, height, fps, bitrate, chosen = null,
                keyFrameSeconds = 2 } = {}) {
    checkSize(width, height);
    checkFps(fps);
    this.width = width;
    this.height = height;
    this.fps = fps;
    this.bitrate = bitrate ?? defaultBitrate(width, height, fps);
    this.keyFrameInterval = Math.max(1, Math.round(fps * keyFrameSeconds));

    this.codec = null;
    this.container = null;
    this.fileExtension = null;
    this.mimeType = null;
    this.colorSpace = null;
    this.warnings = [];
    this.frameCount = 0;
    this.chunkCount = 0;
    this.encodedBytes = 0;

    this._chosen = chosen;
    this._state = "new";
    this._encoder = null;
    this._output = null;
    this._source = null;
    this._converter = null;
    this._encoderError = null;
    this._nextIndex = 0;
    this._pending = Promise.resolve();
    this._sawFirstMeta = false;
  }

  /** Pick a codec if one was not handed in, then open encoder and container. */
  async init() {
    if (this._state !== "new") {
      throw new Error(`VideoWriter.init called in state "${this._state}".`);
    }
    const chosen = this._chosen ?? await chooseCodec({
      width: this.width, height: this.height, fps: this.fps,
      bitrate: this.bitrate,
    });
    this._chosen = chosen;
    this.codec = chosen.codec;
    this.container = chosen.container;
    this.fileExtension = chosen.fileExtension;
    if (chosen.colorSpaceWarning) this.warnings.push(chosen.colorSpaceWarning);

    this._converter = makeImageDataConverter(this.width, this.height);
    this._source = new EncodedVideoPacketSource(chosen.codecFamily);
    this._output = new Output({
      // fastStart puts the index at the front of the file, which is what a
      // downloaded MP4 needs to be scrubbable without reading to the end.
      // The whole file is in an ArrayBuffer anyway, so it costs nothing.
      format: chosen.container === "mp4"
        ? new Mp4OutputFormat({ fastStart: "in-memory" })
        : new WebMOutputFormat(),
      target: new BufferTarget(),
    });
    // frameRate is not decoration: the MP4 muxer derives the track timescale
    // from it, so 30 fps gives a timescale of 30 and every frame lands on an
    // integer tick instead of being rounded off a microsecond clock.
    this._output.addVideoTrack(this._source, { frameRate: this.fps });

    this._encoder = new VideoEncoder({
      output: (chunk, meta) => this._onChunk(chunk, meta),
      error: (e) => { this._encoderError = e; },
    });
    this._encoder.configure(chosen.encoderConfig);
    await this._output.start();
    this._state = "writing";
    return chosen;
  }

  /**
   * Hand over one rendered frame.
   *
   * `index` is the frame's position in the output, and it is the only thing
   * that decides when the frame is shown. Pass 0, 1, 2, ... and a 900-frame
   * track is 30 seconds at 30 fps whether it rendered in a minute or an hour.
   *
   * @param {ImageData|VideoFrame|CanvasImageSource} source
   * @param {number} index
   */
  async addFrame(source, index) {
    if (this._state !== "writing") {
      throw new Error(
        `VideoWriter.addFrame called in state "${this._state}"; ` +
        "call init() first and do not add frames after finish().");
    }
    this._throwIfEncoderFailed();
    if (!Number.isInteger(index) || index < 0) {
      throw new Error(`Frame index must be a non-negative integer; got ${index}.`);
    }
    if (index !== this._nextIndex) {
      throw new Error(
        `Frames must arrive in order: expected index ${this._nextIndex}, ` +
        `got ${index}. Out-of-order frames would silently reorder the video.`);
    }

    // Computed from the index, never accumulated, so 10,000 frames at
    // 29.97 fps do not drift. The durations tile exactly because each is the
    // difference of two rounded start times.
    const timestamp = Math.round((index * 1e6) / this.fps);
    const duration = Math.round(((index + 1) * 1e6) / this.fps) - timestamp;

    const frame = this._toVideoFrame(source, timestamp, duration);
    try {
      await this._waitForQueue();
      this._encoder.encode(frame, {
        keyFrame: index % this.keyFrameInterval === 0,
      });
    } finally {
      // Always ours to close: every branch of _toVideoFrame constructs a new
      // VideoFrame, so a caller who passed one still owns theirs.
      frame.close();
    }
    this._nextIndex = index + 1;
    this.frameCount++;
    // Chunks are muxed from the encoder callback, which is synchronous and
    // cannot await; _pending is the chain of those writes. Joining it here
    // keeps back-pressure honest and surfaces a muxer error at the frame
    // that caused it rather than at finish().
    await this._pending;
    this._throwIfEncoderFailed();
  }

  /** Flush, close the container, and hand back the file. */
  async finish() {
    if (this._state !== "writing") {
      throw new Error(`VideoWriter.finish called in state "${this._state}".`);
    }
    this._state = "finishing";
    await this._encoder.flush();
    this._throwIfEncoderFailed();
    this._encoder.close();
    await this._pending;
    if (this.frameCount === 0) {
      await this._output.cancel();
      this._state = "aborted";
      throw new Error("No frames were added, so there is no video to write.");
    }
    this.mimeType = await this._output.getMimeType();
    await this._output.finalize();
    const buffer = this._output.target.buffer;
    if (!buffer || buffer.byteLength === 0) {
      throw new Error(
        `The muxer accepted ${this.chunkCount} chunks totalling ` +
        `${this.encodedBytes} bytes but produced an empty file.`);
    }
    this._converter.dispose();
    this._state = "done";
    return new Blob([buffer], { type: this.mimeType });
  }

  /** Throw everything away — a cancelled render, or one that failed. */
  async abort() {
    if (this._state === "done" || this._state === "aborted") return;
    this._state = "aborted";
    try {
      if (this._encoder && this._encoder.state !== "closed") {
        this._encoder.close();
      }
    } catch { /* the encoder is already gone; that is what we wanted */ }
    try { this._converter?.dispose(); } catch { /* nothing to release */ }
    try { await this._output?.cancel(); } catch { /* nothing to close */ }
  }

  // --- internals ---

  _throwIfEncoderFailed() {
    if (!this._encoderError) return;
    const e = this._encoderError;
    this._encoderError = null;
    throw new Error(
      `The ${this.codec ?? "video"} encoder failed after ${this.frameCount} ` +
      `frames: ${e.message ?? e}`);
  }

  _toVideoFrame(source, timestamp, duration) {
    if (typeof VideoFrame !== "undefined" && source instanceof VideoFrame) {
      // Cloned rather than used directly, so the caller's frame survives and
      // so its own timestamp — whatever the renderer put there — is replaced
      // by the index-derived one. The index is the only clock that counts.
      return new VideoFrame(source, { timestamp, duration });
    }
    if (typeof ImageData !== "undefined" && source instanceof ImageData) {
      if (source.width !== this.width || source.height !== this.height) {
        throw new Error(
          `Frame ${this._nextIndex} is ${source.width}x${source.height} but ` +
          `the video is ${this.width}x${this.height}. Every frame of a track ` +
          "must be the same size.");
      }
      return this._converter.convert(source, timestamp, duration);
    }
    // A canvas, an ImageBitmap, a <video> — anything the platform calls a
    // CanvasImageSource. If it is none of those, VideoFrame says so better
    // than a guess here would.
    return new VideoFrame(source, { timestamp, duration });
  }

  /**
   * Do not let the encoder's queue grow without bound.
   *
   * With a caller that spends seconds per frame this will essentially never
   * wait, which is exactly why it is worth having: the one track that does
   * outrun the encoder is the 4K one, and the failure mode without this is a
   * tab that dies with no message at frame 400.
   */
  async _waitForQueue() {
    while (this._encoder.encodeQueueSize > MAX_QUEUE_DEPTH) {
      this._throwIfEncoderFailed();
      await new Promise((resolve) => {
        let done = false;
        const finish = () => {
          if (done) return;
          done = true;
          this._encoder.removeEventListener?.("dequeue", finish);
          resolve();
        };
        // "dequeue" is the right signal and Chrome fires it; the timer is
        // there so a browser that does not fire it still makes progress.
        this._encoder.addEventListener?.("dequeue", finish);
        setTimeout(finish, 20);
      });
    }
  }

  _onChunk(chunk, meta) {
    this.chunkCount++;
    this.encodedBytes += chunk.byteLength;
    let packetMeta;
    if (!this._sawFirstMeta) {
      const vetted = vetMeta(meta);
      if (vetted.warning && !this.warnings.includes(vetted.warning)) {
        this.warnings.push(vetted.warning);
        console.warn(`soar video: ${vetted.warning}`);
      }
      this.colorSpace = vetted.colorSpace;
      packetMeta = vetted.meta;
      this._sawFirstMeta = true;
    } else if (meta?.decoderConfig) {
      // Mid-stream reconfiguration. Rare, but the muxer wants to know.
      packetMeta = vetMeta(meta).meta;
    }
    const packet = EncodedPacket.fromEncodedChunk(chunk);
    this._pending = this._pending.then(() => {
      if (this._state === "aborted") return;
      return this._source.add(packet, packetMeta);
    });
  }
}
