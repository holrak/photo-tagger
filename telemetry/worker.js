/**
 * photo-tagger telemetry collector (Cloudflare Worker).
 *
 * Accepts a single anonymous JSON beacon per event and writes one Analytics Engine data point.
 * Two event kinds share one dataset and one column layout: "run" (end of a tagging run) and
 * "crash" (an unhandled exception; type and in-app code location only, never messages). The
 * exact, closed set of fields is documented in the Telemetry section of the project README and
 * produced by src/photo_tagger/telemetry.py (build_payload / build_crash_payload).
 *
 * It is deliberately minimal and privacy-preserving:
 *   - Write-only: it never reads anything back and returns an empty 204.
 *   - It stores no IP address, sets no cookies, and writes only the fields below.
 *   - Every string is clamped so a malformed or hostile client cannot bloat a data point.
 *
 * Schema compatibility: v1 clients (older installs) keep working. The v2 layout is a strict
 * superset of v1 (same blob/double positions, new fields appended), so old and new rows can be
 * queried uniformly; v1 rows simply have empty strings / zeros in the new columns, and their
 * event column is normalized to "run" (v1 predates crash reporting).
 *
 * Querying lives in queries.sql; deployment in README.md.
 */

const KNOWN_SCHEMAS = [1, 2];
const MAX_STR = 200;
// A real beacon is well under 1 KiB; anything bigger is not ours. Rejecting on Content-Length
// keeps a hostile client from making the Worker parse megabytes of JSON (the field clamps below
// already bound what gets stored).
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

    // Reject anything that is not a beacon shape we know how to store.
    if (!body || typeof body !== "object" || !KNOWN_SCHEMAS.includes(body.schema_version)) {
      return new Response("Unsupported schema", { status: 400 });
    }

    env.TELEMETRY.writeDataPoint({
      // index1: the random install id, so COUNT(DISTINCT index1) approximates active installs.
      indexes: [str(body.install_id)],
      // blob1..blob17, in the order queries.sql reads them. blob1..blob11 match schema v1
      // exactly; blob12+ are the v2 additions (empty in rows written by v1 clients).
      blobs: [
        str(body.app_version), //      blob1
        str(body.interface), //        blob2  "cli" | "gui"
        str(body.provider), //         blob3
        str(body.model), //            blob4
        str(body.arch), //             blob5
        str(body.os), //               blob6
        str(body.os_release), //       blob7
        str(body.python_version), //   blob8
        str(body.output_language), //  blob9  metadata language, e.g. "English"
        str(body.ui_language), //      blob10 resolved UI language code, e.g. "en"
        str(body.file_types), //       blob11 distinct extensions, e.g. "cr3,jpg"
        str(body.event) || "run", //   blob12 "run" | "crash" (v1 rows normalize to "run")
        str(body.cpu), //              blob13 CPU model, e.g. "Apple M3 Pro"
        str(body.gpu), //              blob14 GPU model, e.g. "NVIDIA GeForce RTX 4070"
        str(body.exception_type), //   blob15 crash only: exception class name
        str(body.crash_location), //   blob16 crash only: deepest in-app frame "module:func:line"
        str(body.crash_frames), //     blob17 crash only: last in-app frames, '>'-joined
      ],
      // double1..double13. double1..double3 match schema v1; the rest are v2 additions.
      doubles: [
        num(body.schema_version), //   double1
        num(body.batch_size), //       double2
        num(body.duration_seconds), // double3
        num(body.success_count), //    double4
        num(body.failure_count), //    double5
        num(body.cache_hits), //       double6
        num(body.retry_successes), //  double7
        num(body.workers), //          double8
        num(body.total_tokens), //     double9
        num(body.inference_seconds), //double10
        num(body.cpu_count), //        double11
        num(body.memory_gb), //        double12
        num(body.dry_run), //          double13 1 when the run previewed without writing
      ],
    });

    // No body: the client is fire-and-forget and ignores the response.
    return new Response(null, { status: 204 });
  },
};
