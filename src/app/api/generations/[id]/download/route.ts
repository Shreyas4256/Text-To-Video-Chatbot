import { NextRequest } from "next/server";
import { db } from "@/lib/db";
import { requireUser } from "@/lib/auth";
import { ApiError, handleApiError, NotFoundError } from "@/lib/api-utils";

/**
 * Authenticated download proxy. Streams the provider-hosted video through
 * the backend so ownership is enforced and the browser gets a proper
 * attachment filename.
 */
export async function GET(req: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  try {
    const user = await requireUser();
    const { id } = await params;
    const generation = await db.videoGeneration.findFirst({
      where: { id, userId: user.id },
      select: { videoUrl: true, status: true },
    });
    if (!generation) throw new NotFoundError("Generation not found.");
    if (generation.status !== "COMPLETED" || !generation.videoUrl) {
      throw new ApiError("This video is not ready to download yet.", 409);
    }

    // App-relative URLs (dev mock mode) resolve against this deployment.
    const videoUrl = generation.videoUrl.startsWith("/")
      ? new URL(generation.videoUrl, req.nextUrl.origin).toString()
      : generation.videoUrl;

    const upstream = await fetch(videoUrl);
    if (!upstream.ok || !upstream.body) {
      throw new ApiError(
        "The video file could not be fetched from storage. It may have expired — try regenerating.",
        502
      );
    }

    return new Response(upstream.body, {
      headers: {
        "Content-Type": upstream.headers.get("Content-Type") ?? "video/mp4",
        "Content-Disposition": `attachment; filename="video-${id}.mp4"`,
        ...(upstream.headers.get("Content-Length")
          ? { "Content-Length": upstream.headers.get("Content-Length")! }
          : {}),
      },
    });
  } catch (error) {
    return handleApiError(error);
  }
}
