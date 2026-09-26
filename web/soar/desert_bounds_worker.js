// The desert's bounds pyramid, built off the main thread: a module Worker
// running desert_bounds.js's own buildDesertBounds on the height levels the
// loader posts it (desert_bounds.js buildDesertBoundsInWorker), and posting
// the pyramid back. Nothing here but the transport, so the bytes are the
// builder's.

import { buildDesertBounds } from "./desert_bounds.js";

self.onmessage = (e) => {
  try {
    const out = buildDesertBounds(e.data.levels, e.data.cellM);
    self.postMessage({ out }, out.map((a) => a.buffer));
  } catch (err) {
    self.postMessage({ error: String(err?.message || err) });
  }
};
