import { NextRequest, NextResponse } from "next/server";
import { timingSafeEqual } from "crypto";
import { db } from "@/lib/db";
import { getEnv } from "@/lib/env";
import { generationInclude, syncGenerationStatus } from "@/lib/generation-service";

/**
 * Provider webhook receiver. The webhook is only a signal — we never trust
 * its payload. On receipt we re-poll the provider over the authenticated
 * API and persist the verified status, which makes this endpoint safe even
 * if the payload is forged.
 *
 * URL shape (registered at job submission):
 *   POST /api/webhooks/video/<provider>?token=<WEBHOOK_SECRET>&generationId=<id>
 */
export async function POST(req: NextRequest) {
  const env = getEnv();
  if (!env.WEBHOOK_SECRET) {
    return NextResponse.json({ error: "Webhooks are not configured." }, { status: 404 });
  }

  const url = new URL(req.url);
  const token = url.searchParams.get("token") ?? "";
  const expected = Buffer.from(env.WEBHOOK_SECRET);
  const received = Buffer.from(token);
  if (received.length !== expected.length || !timingSafeEqual(received, expected)) {
    return NextResponse.json({ error: "Invalid webhook token." }, { status: 401 });
  }

  const generationId = url.searchParams.get("generationId");
  if (!generationId) {
    return NextResponse.json({ error: "Missing generationId." }, { status: 400 });
  }

  const generation = await db.videoGeneration.findUnique({
    where: { id: generationId },
    include: generationInclude,
  });
  if (!generation) {
    // Return 200 so providers don't endlessly retry for deleted generations.
    return NextResponse.json({ ok: true, note: "Unknown generation." });
  }

  await syncGenerationStatus(generation);
  return NextResponse.json({ ok: true });
}
