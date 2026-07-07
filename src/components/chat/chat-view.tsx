"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import type { ConversationDto, GenerationDto, MessageDto } from "@/lib/types";
import { DEFAULT_SETTINGS, STARTER_PROMPTS } from "@/lib/constants";
import { api, newIdempotencyKey } from "@/lib/client-api";
import { useToast } from "@/components/toast";
import { Button, Spinner, Textarea } from "@/components/ui";
import { GenerationCard } from "@/components/chat/generation-card";
import { SettingsPanel, type DraftState } from "@/components/chat/settings-panel";

const POLL_MS = 4000;

function draftStorageKey(conversationId: string) {
  return `t2v-draft-${conversationId}`;
}

export function ChatView({
  conversation,
  initialMessages,
  initialGenerations,
}: {
  conversation: ConversationDto;
  initialMessages: MessageDto[];
  initialGenerations: GenerationDto[];
}) {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { toast } = useToast();

  const [messages, setMessages] = useState<MessageDto[]>(initialMessages);
  const [generations, setGenerations] = useState<Record<string, GenerationDto>>(
    () => Object.fromEntries(initialGenerations.map((g) => [g.id, g]))
  );
  const [input, setInput] = useState(() => searchParams.get("prompt") ?? "");
  const [sending, setSending] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [genError, setGenError] = useState<string | null>(null);
  const [panelOpen, setPanelOpen] = useState(false);

  const [draft, setDraft] = useState<DraftState>(() => ({
    prompt: "",
    settings: { ...DEFAULT_SETTINGS },
  }));

  // Restore the per-conversation draft after mount (avoids SSR mismatch).
  useEffect(() => {
    try {
      const raw = localStorage.getItem(draftStorageKey(conversation.id));
      if (raw) setDraft(JSON.parse(raw));
    } catch {
      // Corrupt draft — fall back to defaults.
    }
  }, [conversation.id]);

  const updateDraft = useCallback(
    (next: DraftState) => {
      setDraft(next);
      try {
        localStorage.setItem(draftStorageKey(conversation.id), JSON.stringify(next));
      } catch {
        // Storage full/unavailable — draft just won't persist.
      }
    },
    [conversation.id]
  );

  const scrollRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [messages.length]);

  /* --------------------------- polling ---------------------------- */
  const generationsRef = useRef(generations);
  generationsRef.current = generations;

  const pendingIds = useMemo(
    () =>
      Object.values(generations)
        .filter((g) => g.status === "QUEUED" || g.status === "PROCESSING")
        .map((g) => g.id),
    [generations]
  );

  useEffect(() => {
    if (pendingIds.length === 0) return;
    const timer = setInterval(async () => {
      for (const id of pendingIds) {
        try {
          const { generation } = await api<{ generation: GenerationDto }>(
            `/api/generations/${id}`
          );
          const before = generationsRef.current[id]?.status;
          setGenerations((prev) => ({ ...prev, [id]: generation }));
          if (before && before !== generation.status) {
            if (generation.status === "COMPLETED") {
              toast("Your video is ready! 🎉", "success");
            } else if (generation.status === "FAILED") {
              toast("A generation failed — see the chat for details.", "error");
            }
          }
        } catch {
          // Transient polling error — next tick will retry.
        }
      }
    }, POLL_MS);
    return () => clearInterval(timer);
  }, [pendingIds, toast]);

  /* --------------------------- actions ---------------------------- */

  async function refreshConversation() {
    try {
      const data = await api<{
        messages: MessageDto[];
        generations: GenerationDto[];
      }>(`/api/conversations/${conversation.id}`);
      setMessages(data.messages);
      setGenerations(Object.fromEntries(data.generations.map((g) => [g.id, g])));
    } catch {
      // Non-fatal; UI keeps optimistic state.
    }
  }

  async function sendMessage(text?: string) {
    const content = (text ?? input).trim();
    if (!content || sending) return;
    setSending(true);
    setInput("");

    const optimistic: MessageDto = {
      id: `optimistic-${Date.now()}`,
      role: "USER",
      content,
      generationId: null,
      createdAt: new Date().toISOString(),
    };
    setMessages((m) => [...m, optimistic]);

    try {
      const res = await api<{
        userMessage: MessageDto;
        assistantMessage: MessageDto;
        draft: DraftState;
      }>(`/api/conversations/${conversation.id}/messages`, {
        method: "POST",
        json: { content, draft: draft.prompt ? draft : undefined },
      });
      setMessages((m) => [
        ...m.filter((x) => x.id !== optimistic.id),
        res.userMessage,
        res.assistantMessage,
      ]);
      updateDraft(res.draft);
      router.refresh(); // keeps the sidebar title in sync
    } catch (err) {
      setMessages((m) => m.filter((x) => x.id !== optimistic.id));
      setInput(content);
      toast(err instanceof Error ? err.message : "Message failed to send", "error");
    } finally {
      setSending(false);
    }
  }

  async function generate() {
    setGenError(null);
    if (draft.prompt.trim().length < 3) {
      setGenError("Write a prompt first — describe what the video should show.");
      return;
    }
    setGenerating(true);
    try {
      const { generation } = await api<{ generation: GenerationDto }>(
        "/api/generations",
        {
          method: "POST",
          json: {
            prompt: draft.prompt.trim(),
            settings: draft.settings,
            conversationId: conversation.id,
            idempotencyKey: newIdempotencyKey(),
            enhance: true,
          },
        }
      );
      setGenerations((prev) => ({ ...prev, [generation.id]: generation }));
      await refreshConversation();
      setPanelOpen(false);
      if (generation.status === "FAILED") {
        toast(generation.errorMessage ?? "Generation could not be submitted", "error");
      } else {
        toast("Generation submitted — tracking progress in the chat", "success");
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Generation failed";
      setGenError(msg);
      toast(msg, "error");
    } finally {
      setGenerating(false);
    }
  }

  async function retryGeneration(id: string) {
    try {
      const { generation } = await api<{ generation: GenerationDto }>(
        `/api/generations/${id}/retry`,
        { method: "POST" }
      );
      setGenerations((prev) => ({ ...prev, [generation.id]: generation }));
      await refreshConversation();
      toast("New attempt submitted", "success");
    } catch (err) {
      toast(err instanceof Error ? err.message : "Retry failed", "error");
    }
  }

  function onKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void sendMessage();
    }
  }

  /* --------------------------- render ------------------------------ */

  return (
    <div className="flex h-full min-h-0">
      {/* Chat column */}
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex items-center justify-between gap-3 border-b border-surface-700 px-4 py-3">
          <h1 className="min-w-0 truncate text-sm font-semibold text-zinc-200">
            {conversation.title}
          </h1>
          <Button
            variant="secondary"
            size="sm"
            className="lg:hidden"
            onClick={() => setPanelOpen(true)}
          >
            🎛 Settings
          </Button>
        </header>

        <div ref={scrollRef} className="min-h-0 flex-1 overflow-y-auto px-4 py-6">
          <div className="mx-auto max-w-2xl space-y-4">
            {messages.length === 0 && (
              <div className="py-10 text-center">
                <p className="text-lg font-semibold text-zinc-200">
                  What should we create today?
                </p>
                <p className="mt-1 text-sm text-zinc-500">
                  Describe a video in plain language — I&apos;ll shape it into a strong
                  generation prompt.
                </p>
                <div className="mx-auto mt-6 grid max-w-lg gap-2 text-left">
                  {STARTER_PROMPTS.slice(0, 3).map((p) => (
                    <button
                      key={p}
                      onClick={() => sendMessage(p)}
                      className="rounded-xl border border-surface-700 bg-surface-900 px-4 py-3 text-sm text-zinc-300 transition-colors hover:border-accent-500/50"
                    >
                      {p}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {messages.map((m) => (
              <div key={m.id}>
                {m.role === "USER" ? (
                  <div className="ml-auto w-fit max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-br-sm bg-accent-600 px-4 py-2.5 text-sm text-white">
                    {m.content}
                  </div>
                ) : (
                  <div className="space-y-2">
                    <div className="w-fit max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-bl-sm bg-surface-800 px-4 py-2.5 text-sm text-zinc-200">
                      {renderAssistantText(m.content)}
                    </div>
                    {m.generationId && generations[m.generationId] && (
                      <GenerationCard
                        generation={generations[m.generationId]}
                        onRetry={retryGeneration}
                      />
                    )}
                  </div>
                )}
              </div>
            ))}

            {sending && (
              <div className="flex w-fit items-center gap-2 rounded-2xl rounded-bl-sm bg-surface-800 px-4 py-3 text-sm text-zinc-400">
                <Spinner className="size-4" /> Thinking…
              </div>
            )}
          </div>
        </div>

        <div className="border-t border-surface-700 p-3 sm:p-4">
          <div className="mx-auto flex max-w-2xl items-end gap-2">
            <Textarea
              rows={1}
              placeholder='Describe a video, or say "make it more cinematic"…'
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onKeyDown}
              className="max-h-40 min-h-[44px] resize-y"
              aria-label="Chat message"
            />
            <Button
              onClick={() => sendMessage()}
              disabled={!input.trim()}
              loading={sending}
              aria-label="Send message"
              className="h-[44px] shrink-0"
            >
              Send
            </Button>
          </div>
          <p className="mx-auto mt-1.5 max-w-2xl text-[11px] text-zinc-600">
            Enter to send · Shift+Enter for a new line · Generate from the settings panel
          </p>
        </div>
      </div>

      {/* Settings panel — desktop */}
      <aside className="hidden w-80 shrink-0 border-l border-surface-700 bg-surface-900 lg:block">
        <SettingsPanel
          draft={draft}
          onChange={updateDraft}
          onGenerate={generate}
          generating={generating}
          validationError={genError}
        />
      </aside>

      {/* Settings panel — mobile sheet */}
      {panelOpen && (
        <div className="fixed inset-0 z-50 lg:hidden">
          <div className="absolute inset-0 bg-black/60" onClick={() => setPanelOpen(false)} aria-hidden />
          <aside className="absolute inset-y-0 right-0 flex w-[85%] max-w-sm flex-col bg-surface-900">
            <div className="flex items-center justify-between border-b border-surface-700 px-4 py-3">
              <h2 className="text-sm font-semibold text-zinc-200">Generation settings</h2>
              <button
                onClick={() => setPanelOpen(false)}
                aria-label="Close settings"
                className="rounded-md p-1.5 text-zinc-400 hover:bg-surface-700"
              >
                ✕
              </button>
            </div>
            <div className="min-h-0 flex-1">
              <SettingsPanel
                draft={draft}
                onChange={updateDraft}
                onGenerate={generate}
                generating={generating}
                validationError={genError}
              />
            </div>
          </aside>
        </div>
      )}
    </div>
  );
}

/** Render **bold** spans in assistant text without a markdown dependency. */
function renderAssistantText(text: string): React.ReactNode {
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  return parts.map((part, i) =>
    part.startsWith("**") && part.endsWith("**") ? (
      <strong key={i} className="font-semibold text-white">
        {part.slice(2, -2)}
      </strong>
    ) : (
      <span key={i}>{part}</span>
    )
  );
}
