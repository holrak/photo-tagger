/**
 * photo-tagger telemetry collector (Cloudflare Worker).
 *
 * Accepts a single anonymous JSON beacon per run and writes one Analytics Engine data point. The
 * exact, closed set of fields it expects is documented in the Telemetry section of the project
 * README and produced by src/photo_tagger/telemetry.py::build_payload.
 *
 * It is deliberately minimal and privacy-preserving:
 *   - Write-only: it never reads anything back and returns an empty 204.
 *   - It stores no IP address, sets no cookies, and writes only the fields below.
 *   - Every string is clamped so a malformed or hostile client cannot bloat a data point.
 *
 * Querying lives in queries.sql; deployment in README.md.
 */

const SCHEMA_VERSION = 1;
const MAX_STR = 200;
// A real beacon is ~400 bytes; anything bigger is not ours. Rejecting on Content-Length keeps a
// hostile client from making the Worker parse megabytes of JSON (the field clamps below already
// bound what gets stored).
const MAX_BODY_BYTES = 4096;

// Coerce to a bounded string; anything non-string (or oversized) becomes a safe value.
const str = (v) => (typeof v === "string" ? v.slice(0, MAX_STR) : "");
// Coerce to a finite number; NaN/Infinity/non-numbers become 0.
const num = (v) => (typeof v === "number" && Number.isFinite(v) ? v : 0);

export default {
  async fetch(request, env) {
    if (request.method !== "POST") {
      return new Response("Method Not Allowed", { status: 405 });
    }

    const length = Number(request.headers.get("content-length") ?? 0);
    if (!Number.isFinite(length) || length > MAX_BODY_BYTES) {
      return new Response("Payload Too Large", { status: 413 });
    }

    let body;
    try {
      body = await request.json();
    } catch {
      return new Response("Bad Request", { status: 400 });
    }

    // Reject anything that is not the beacon shape we know how to store.
    if (!body || typeof body !== "object" || body.schema_version !== SCHEMA_VERSION) {
      return new Response("Unsupported schema", { status: 400 });
    }

    env.TELEMETRY.writeDataPoint({
      // index1: the random install id, so COUNT(DISTINCT index1) approximates active installs.
      indexes: [str(body.install_id)],
      // blob1..blob11, in the order queries.sql reads them.
      blobs: [
        str(body.app_version), //     blob1
        str(body.interface), //       blob2  "cli" | "gui"
        str(body.provider), //        blob3
        str(body.model), //           blob4
        str(body.arch), //            blob5
        str(body.os), //              blob6
        str(body.os_release), //      blob7
        str(body.python_version), //  blob8
        str(body.output_language), // blob9  metadata language, e.g. "English"
        str(body.ui_language), //     blob10 resolved UI language code, e.g. "en"
        str(body.file_types), //      blob11 distinct extensions, e.g. "cr3,jpg"
      ],
      // double1..double3.
      doubles: [
        num(body.schema_version), //  double1
        num(body.batch_size), //      double2
        num(body.duration_seconds), //double3
      ],
    });

    // No body: the client is fire-and-forget and ignores the response.
    return new Response(null, { status: 204 });
  },
};
