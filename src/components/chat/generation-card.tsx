"use client";

import { useState } from "react";
import type { GenerationDto } from "@/lib/types";
import { Button, Spinner, StatusBadge } from "@/components/ui";
import { useToast } from "@/components/toast";

/**
 * Live status card for a generation inside the chat or library detail.
 * Status shown here always reflects the database/provider — never simulated.
 */
export function GenerationCard({
  generation,
  onRetry,
  onDelete,
  compact = false,
}: {
  generation: GenerationDto;
  onRetry?: (id: string) => Promise<void> | void;
  onDelete?: (id: string) => Promise<void> | void;
  compact?: boolean;
}) {
  const { toast } = useToast();
  const [retrying, setRetrying] = useState(false);

  const aspect =
    generation.settings?.aspectRatio === "9:16"
      ? "aspect-[9/16] max-h-96"
      : generation.settings?.aspectRatio === "1:1"
        ? "aspect-square max-h-80"
        : "aspect-video";

  async function copyPrompt() {
    try {
      await navigator.clipboard.writeText(
        generation.enhancedPrompt ?? generation.prompt
      );
      toast("Prompt copied to clipboard", "success");
    } catch {
      toast("Could not copy prompt", "error");
    }
  }

  async function handleRetry() {
    if (!onRetry) return;
    setRetrying(true);
    try {
      await onRetry(generation.id);
    } finally {
      setRetrying(false);
    }
  }

  return (
    <div className="w-full max-w-lg rounded-xl border border-surface-600 bg-surface-800 p-3.5">
      <div className="flex items-start justify-between gap-3">
        <p className="line-clamp-2 text-xs text-zinc-400" title={generation.prompt}>
          {generation.prompt}
        </p>
        <StatusBadge status={generation.status} />
      </div>

      {(generation.status === "QUEUED" || generation.status === "PROCESSING") && (
        <div className={`mt-3 flex items-center justify-center rounded-lg bg-surface-900 ${compact ? "py-8" : "py-12"}`}>
          <div className="flex flex-col items-center gap-2 text-zinc-400">
            <Spinner className="size-6 text-accent-400" />
            <p className="text-xs">
              {generation.status === "QUEUED"
                ? "Waiting in the provider queue…"
                : "The AI is rendering your video…"}
            </p>
          </div>
        </div>
      )}

      {generation.status === "COMPLETED" && generation.videoUrl && (
        <video
          controls
          playsInline
          preload="metadata"
          poster={generation.thumbnailUrl ?? undefined}
          src={generation.videoUrl}
          className={`mt-3 w-full rounded-lg bg-black object-contain ${aspect}`}
        />
      )}

      {generation.status === "FAILED" && (
        <div className="mt-3 rounded-lg border border-red-900/50 bg-red-950/40 px-3 py-2.5 text-xs text-red-200">
          {generation.errorMessage ?? "Generation failed for an unknown reason."}
        </div>
      )}

      <div className="mt-3 flex flex-wrap items-center gap-2">
        {generation.status === "COMPLETED" && generation.videoUrl && (
          <a
            href={`/api/generations/${generation.id}/download`}
            className="rounded-md bg-surface-700 px-2.5 py-1.5 text-xs font-medium text-zinc-200 transition-colors hover:bg-surface-600"
          >
            ⬇ Download
          </a>
        )}
        <button
          onClick={copyPrompt}
          className="rounded-md bg-surface-700 px-2.5 py-1.5 text-xs font-medium text-zinc-200 transition-colors hover:bg-surface-600"
        >
          ⧉ Copy prompt
        </button>
        {onRetry && (generation.status === "FAILED" || generation.status === "COMPLETED") && (
          <Button
            variant="secondary"
            size="sm"
            loading={retrying}
            onClick={handleRetry}
          >
            ↻ {generation.status === "FAILED" ? "Retry" : "Regenerate"}
          </Button>
        )}
        {onDelete && (
          <button
            onClick={() => onDelete(generation.id)}
            className="rounded-md px-2.5 py-1.5 text-xs font-medium text-red-300 transition-colors hover:bg-red-950/50"
          >
            Delete
          </button>
        )}
        {generation.settings && (
          <span className="ml-auto text-[11px] text-zinc-500">
            {generation.settings.aspectRatio} · {generation.settings.durationSec}s ·{" "}
            {generation.settings.quality}
          </span>
        )}
      </div>
    </div>
  );
}
