"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import type { GenerationDto } from "@/lib/types";
import { api } from "@/lib/client-api";
import { useToast } from "@/components/toast";
import { Button, EmptyState, Input, Select, Skeleton, StatusBadge } from "@/components/ui";
import { GenerationCard } from "@/components/chat/generation-card";
import { STYLE_PRESETS } from "@/lib/constants";

const POLL_MS = 5000;

type SortOption = "newest" | "oldest" | "duration" | "status";

export function LibraryView() {
  const { toast } = useToast();
  const [items, setItems] = useState<GenerationDto[]>([]);
  const [cursor, setCursor] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [status, setStatus] = useState<string>("");
  const [search, setSearch] = useState("");
  const [debouncedSearch, setDebouncedSearch] = useState("");
  const [sort, setSort] = useState<SortOption>("newest");
  const [selected, setSelected] = useState<GenerationDto | null>(null);

  useEffect(() => {
    const t = setTimeout(() => setDebouncedSearch(search), 350);
    return () => clearTimeout(t);
  }, [search]);

  const query = useCallback(
    (cursorArg?: string | null) => {
      const params = new URLSearchParams();
      if (status) params.set("status", status);
      if (debouncedSearch) params.set("search", debouncedSearch);
      params.set("sort", sort);
      params.set("limit", "12");
      if (cursorArg) params.set("cursor", cursorArg);
      return `/api/generations?${params.toString()}`;
    },
    [status, debouncedSearch, sort]
  );

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api<{ generations: GenerationDto[]; nextCursor: string | null }>(
        query()
      );
      setItems(data.generations);
      setCursor(data.nextCursor);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load your library.");
    } finally {
      setLoading(false);
    }
  }, [query]);

  useEffect(() => {
    void load();
  }, [load]);

  async function loadMore() {
    if (!cursor || loadingMore) return;
    setLoadingMore(true);
    try {
      const data = await api<{ generations: GenerationDto[]; nextCursor: string | null }>(
        query(cursor)
      );
      setItems((prev) => [...prev, ...data.generations]);
      setCursor(data.nextCursor);
    } catch (err) {
      toast(err instanceof Error ? err.message : "Could not load more", "error");
    } finally {
      setLoadingMore(false);
    }
  }

  /* Poll pending items so statuses stay live. */
  const pendingIds = useMemo(
    () => items.filter((g) => g.status === "QUEUED" || g.status === "PROCESSING").map((g) => g.id),
    [items]
  );
  const pendingRef = useRef(pendingIds);
  pendingRef.current = pendingIds;

  useEffect(() => {
    if (pendingIds.length === 0) return;
    const timer = setInterval(async () => {
      for (const id of pendingRef.current) {
        try {
          const { generation } = await api<{ generation: GenerationDto }>(
            `/api/generations/${id}`
          );
          setItems((prev) => prev.map((g) => (g.id === id ? generation : g)));
          setSelected((sel) => (sel?.id === id ? generation : sel));
        } catch {
          // retry next tick
        }
      }
    }, POLL_MS);
    return () => clearInterval(timer);
  }, [pendingIds.length]);

  async function deleteGeneration(id: string) {
    if (!confirm("Delete this generation permanently?")) return;
    try {
      await api(`/api/generations/${id}`, { method: "DELETE" });
      setItems((prev) => prev.filter((g) => g.id !== id));
      setSelected((sel) => (sel?.id === id ? null : sel));
      toast("Generation deleted", "success");
    } catch (err) {
      toast(err instanceof Error ? err.message : "Delete failed", "error");
    }
  }

  async function retryGeneration(id: string) {
    try {
      const { generation } = await api<{ generation: GenerationDto }>(
        `/api/generations/${id}/retry`,
        { method: "POST" }
      );
      setItems((prev) => [generation, ...prev]);
      toast("New attempt submitted", "success");
    } catch (err) {
      toast(err instanceof Error ? err.message : "Retry failed", "error");
    }
  }

  return (
    <div className="mx-auto max-w-6xl px-4 py-8 sm:px-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold text-white">Video library</h1>
          <p className="mt-1 text-sm text-zinc-400">
            Every generation you&apos;ve created, with live status.
          </p>
        </div>
      </div>

      {/* Filters */}
      <div className="mt-6 flex flex-wrap items-center gap-2">
        <Input
          type="search"
          placeholder="Search prompts…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="max-w-xs"
          aria-label="Search prompts"
        />
        <Select
          value={status}
          onChange={(e) => setStatus(e.target.value)}
          className="w-auto"
          aria-label="Filter by status"
        >
          <option value="">All statuses</option>
          <option value="COMPLETED">Completed</option>
          <option value="PROCESSING">Processing</option>
          <option value="QUEUED">Queued</option>
          <option value="FAILED">Failed</option>
        </Select>
        <Select
          value={sort}
          onChange={(e) => setSort(e.target.value as SortOption)}
          className="w-auto"
          aria-label="Sort"
        >
          <option value="newest">Newest first</option>
          <option value="oldest">Oldest first</option>
          <option value="duration">Longest first</option>
          <option value="status">By status</option>
        </Select>
      </div>

      {/* Grid */}
      {loading ? (
        <div className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <Skeleton key={i} className="h-56" />
          ))}
        </div>
      ) : error ? (
        <div className="mt-6">
          <EmptyState
            icon="⚠️"
            title="Couldn't load your library"
            description={error}
            action={
              <Button variant="secondary" onClick={() => void load()}>
                Try again
              </Button>
            }
          />
        </div>
      ) : items.length === 0 ? (
        <div className="mt-6">
          <EmptyState
            icon="🎬"
            title={debouncedSearch || status ? "No matches" : "No videos yet"}
            description={
              debouncedSearch || status
                ? "Try different filters or a different search term."
                : "Start a chat and generate your first video — it'll show up here."
            }
            action={
              !debouncedSearch && !status ? (
                <Link
                  href="/dashboard"
                  className="rounded-lg bg-accent-600 px-4 py-2 text-sm font-medium text-white hover:bg-accent-500"
                >
                  Go create one
                </Link>
              ) : undefined
            }
          />
        </div>
      ) : (
        <>
          <div className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {items.map((g) => (
              <VideoTile key={g.id} generation={g} onOpen={() => setSelected(g)} />
            ))}
          </div>
          {cursor && (
            <div className="mt-8 text-center">
              <Button variant="secondary" onClick={loadMore} loading={loadingMore}>
                Load more
              </Button>
            </div>
          )}
        </>
      )}

      {/* Detail modal */}
      {selected && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4"
          role="dialog"
          aria-modal="true"
          aria-label="Video details"
        >
          <div className="absolute inset-0 bg-black/70" onClick={() => setSelected(null)} aria-hidden />
          <div className="relative max-h-[90vh] w-full max-w-xl overflow-y-auto rounded-2xl border border-surface-600 bg-surface-900 p-5">
            <div className="mb-4 flex items-start justify-between gap-4">
              <h2 className="text-sm font-semibold text-zinc-200">Generation details</h2>
              <button
                onClick={() => setSelected(null)}
                aria-label="Close details"
                className="rounded-md p-1 text-zinc-400 hover:bg-surface-700"
              >
                ✕
              </button>
            </div>
            <GenerationCard
              generation={selected}
              onRetry={retryGeneration}
              onDelete={deleteGeneration}
            />
            <dl className="mt-4 grid grid-cols-2 gap-x-4 gap-y-2 text-xs">
              <DetailRow label="Created" value={new Date(selected.createdAt).toLocaleString()} />
              <DetailRow label="Provider" value={selected.provider ?? "—"} />
              <DetailRow
                label="Style"
                value={
                  STYLE_PRESETS.find((s) => s.id === selected.settings?.style)?.label ??
                  selected.settings?.style ??
                  "—"
                }
              />
              <DetailRow label="Camera" value={selected.settings?.cameraMovement ?? "—"} />
              <DetailRow
                label="Motion"
                value={selected.settings ? `${selected.settings.motionStrength}/10` : "—"}
              />
              <DetailRow
                label="Seed"
                value={selected.settings?.seed !== undefined ? String(selected.settings.seed) : "Random"}
              />
            </dl>
            {selected.enhancedPrompt && (
              <div className="mt-4">
                <p className="text-xs font-semibold uppercase tracking-wide text-zinc-500">
                  Enhanced prompt
                </p>
                <p className="mt-1 rounded-lg bg-surface-800 p-3 text-xs leading-relaxed text-zinc-300">
                  {selected.enhancedPrompt}
                </p>
              </div>
            )}
            {selected.conversationId && (
              <Link
                href={`/dashboard/chat/${selected.conversationId}`}
                className="mt-4 inline-block text-xs text-accent-400 hover:underline"
              >
                Open source chat →
              </Link>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function DetailRow({ label, value }: { label: string; value: string }) {
  return (
    <>
      <dt className="text-zinc-500">{label}</dt>
      <dd className="text-zinc-300">{value}</dd>
    </>
  );
}

function VideoTile({
  generation,
  onOpen,
}: {
  generation: GenerationDto;
  onOpen: () => void;
}) {
  return (
    <button
      onClick={onOpen}
      className="group overflow-hidden rounded-xl border border-surface-700 bg-surface-900 text-left transition-colors hover:border-accent-500/50 focus-visible:outline-2 focus-visible:outline-accent-400"
    >
      <div className="relative aspect-video bg-surface-800">
        {generation.status === "COMPLETED" && generation.videoUrl ? (
          <video
            src={generation.videoUrl}
            poster={generation.thumbnailUrl ?? undefined}
            muted
            playsInline
            preload="metadata"
            className="size-full object-cover"
          />
        ) : (
          <div className="flex size-full items-center justify-center text-3xl opacity-40">
            {generation.status === "FAILED" ? "⚠️" : "⏳"}
          </div>
        )}
        <span className="absolute left-2 top-2">
          <StatusBadge status={generation.status} />
        </span>
      </div>
      <div className="p-3">
        <p className="line-clamp-2 text-sm text-zinc-300">{generation.prompt}</p>
        <p className="mt-2 text-[11px] text-zinc-500">
          {new Date(generation.createdAt).toLocaleDateString()} ·{" "}
          {generation.settings
            ? `${generation.settings.aspectRatio} · ${generation.settings.durationSec}s`
            : ""}
        </p>
      </div>
    </button>
  );
}
