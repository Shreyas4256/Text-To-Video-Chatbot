"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { api } from "@/lib/client-api";
import { useToast } from "@/components/toast";
import type { ConversationDto } from "@/lib/types";
import { Button } from "@/components/ui";

/**
 * Creates a conversation and navigates to it. An optional starter prompt is
 * passed via query param and pre-filled into the chat input (never
 * auto-submitted).
 */
export function NewChatButton({
  prompt,
  children,
  variant = "primary",
  className,
}: {
  prompt?: string;
  children: React.ReactNode;
  variant?: "primary" | "secondary" | "ghost";
  className?: string;
}) {
  const router = useRouter();
  const { toast } = useToast();
  const [loading, setLoading] = useState(false);

  async function start() {
    setLoading(true);
    try {
      const { conversation } = await api<{ conversation: ConversationDto }>(
        "/api/conversations",
        { method: "POST", json: {} }
      );
      const qs = prompt ? `?prompt=${encodeURIComponent(prompt)}` : "";
      router.push(`/dashboard/chat/${conversation.id}${qs}`);
      router.refresh();
    } catch (err) {
      toast(err instanceof Error ? err.message : "Could not start a chat", "error");
      setLoading(false);
    }
  }

  return (
    <Button onClick={start} loading={loading} variant={variant} className={className}>
      {children}
    </Button>
  );
}
