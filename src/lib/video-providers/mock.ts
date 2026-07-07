import {
  CreateJobInput,
  CreateJobResult,
  JobStatus,
  VideoProvider,
} from "./provider-interface";

/**
 * DEVELOPMENT-ONLY mock provider.
 *
 * Lets you exercise the full generation lifecycle (queued → processing →
 * completed/failed) without a paid API key. It is refused in production by
 * env validation (see src/lib/env.ts) and the UI displays a persistent
 * "mock mode" banner while it is active.
 *
 * Timing is encoded in the job ID so the mock is stateless and survives
 * server restarts. Include "[fail]" in a prompt to simulate a provider
 * failure and test error states.
 */
const QUEUE_MS = 4_000;
const PROCESSING_MS = 12_000;

// Locally bundled synthetic test clip (generated with ffmpeg testsrc2),
// served from /public so dev mock mode works fully offline.
const SAMPLE_VIDEO_URL = "/mock/sample-video.mp4";

export class MockProvider implements VideoProvider {
  readonly name = "mock";

  async createJob(input: CreateJobInput): Promise<CreateJobResult> {
    const shouldFail = input.prompt.toLowerCase().includes("[fail]");
    const id = `mock_${Date.now()}_${shouldFail ? "fail" : "ok"}_${Math.random()
      .toString(36)
      .slice(2, 8)}`;
    return {
      providerJobId: id,
      requestPayload: {
        mock: true,
        note: "Development mock mode — no real provider was called.",
        prompt: input.prompt,
        aspectRatio: input.aspectRatio,
        durationSec: input.durationSec,
      },
    };
  }

  async getJobStatus(providerJobId: string): Promise<JobStatus> {
    const parts = providerJobId.split("_");
    const createdAt = Number(parts[1]);
    const shouldFail = parts[2] === "fail";
    if (!Number.isFinite(createdAt)) {
      return { state: "failed", errorMessage: "Malformed mock job ID." };
    }
    const elapsed = Date.now() - createdAt;
    if (elapsed < QUEUE_MS) return { state: "queued" };
    if (elapsed < QUEUE_MS + PROCESSING_MS) return { state: "processing" };
    if (shouldFail) {
      return {
        state: "failed",
        errorMessage:
          "Simulated provider failure (prompt contained \"[fail]\"). This is mock mode.",
      };
    }
    return { state: "completed", videoUrl: SAMPLE_VIDEO_URL };
  }
}
