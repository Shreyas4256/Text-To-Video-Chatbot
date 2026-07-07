"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useState } from "react";
import type { ConversationDto } from "@/lib/types";
import { api } from "@/lib/client-api";
import { useToast } from "@/components/toast";
import { Spinner } from "@/components/ui";

export function Sidebar({
  user,
  conversations,
}: {
  user: { email: string; name: string | null };
  conversations: ConversationDto[];
}) {
  const pathname = usePathname();
  const router = useRouter();
  const { toast } = useToast();
  const [open, setOpen] = useState(false);
  const [creating, setCreating] = useState(false);
  const [loggingOut, setLoggingOut] = useState(false);

  async function newConversation() {
    setCreating(true);
    try {
      const { conversation } = await api<{ conversation: ConversationDto }>(
        "/api/conversations",
        { method: "POST", json: {} }
      );
      setOpen(false);
      router.push(`/dashboard/chat/${conversation.id}`);
      router.refresh();
    } catch (err) {
      toast(err instanceof Error ? err.message : "Could not create conversation", "error");
    } finally {
      setCreating(false);
    }
  }

  async function logout() {
    setLoggingOut(true);
    try {
      await api("/api/auth/logout", { method: "POST" });
      router.push("/");
      router.refresh();
    } catch {
      setLoggingOut(false);
      toast("Logout failed — try again", "error");
    }
  }

  async function deleteConversation(id: string) {
    if (!confirm("Delete this chat? Its videos stay in your library.")) return;
    try {
      await api(`/api/conversations/${id}`, { method: "DELETE" });
      toast("Chat deleted", "success");
      if (pathname === `/dashboard/chat/${id}`) router.push("/dashboard");
      router.refresh();
    } catch (err) {
      toast(err instanceof Error ? err.message : "Delete failed", "error");
    }
  }

  const nav = (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between p-4">
        <Link href="/dashboard" className="flex items-center gap-2 font-bold text-white">
          <span className="flex size-7 items-center justify-center rounded-lg bg-gradient-to-br from-accent-500 to-cyan-500 text-sm font-black">
            F
          </span>
          FrameFlow
        </Link>
        <button
          className="rounded-md p-1.5 text-zinc-400 hover:bg-surface-700 md:hidden"
          onClick={() => setOpen(false)}
          aria-label="Close menu"
        >
          ✕
        </button>
      </div>

      <div className="px-3">
        <button
          onClick={newConversation}
          disabled={creating}
          className="flex w-full items-center justify-center gap-2 rounded-lg bg-accent-600 px-4 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-accent-500 disabled:opacity-60"
        >
          {creating ? <Spinner className="size-4" /> : <span aria-hidden>＋</span>}
          New video
        </button>
      </div>

      <nav className="mt-4 space-y-0.5 px-3">
        <NavLink href="/dashboard" active={pathname === "/dashboard"}>
          🏠 Overview
        </NavLink>
        <NavLink href="/dashboard/library" active={pathname.startsWith("/dashboard/library")}>
          🎞️ Video library
        </NavLink>
        <NavLink href="/dashboard/settings" active={pathname.startsWith("/dashboard/settings")}>
          ⚙️ Account
        </NavLink>
      </nav>

      <div className="mt-5 min-h-0 flex-1 overflow-y-auto px-3 pb-2">
        <p className="mb-2 px-2 text-[11px] font-semibold uppercase tracking-wider text-zinc-500">
          Recent chats
        </p>
        {conversations.length === 0 && (
          <p className="px-2 text-xs text-zinc-600">No chats yet — start one!</p>
        )}
        <ul className="space-y-0.5">
          {conversations.map((c) => {
            const active = pathname === `/dashboard/chat/${c.id}`;
            return (
              <li key={c.id} className="group relative">
                <Link
                  href={`/dashboard/chat/${c.id}`}
                  onClick={() => setOpen(false)}
                  className={`block truncate rounded-lg px-2.5 py-2 pr-8 text-sm transition-colors ${
                    active
                      ? "bg-surface-700 text-white"
                      : "text-zinc-400 hover:bg-surface-800 hover:text-zinc-200"
                  }`}
                >
                  {c.title}
                </Link>
                <button
                  onClick={() => deleteConversation(c.id)}
                  aria-label={`Delete chat: ${c.title}`}
                  className="absolute right-1.5 top-1/2 -translate-y-1/2 rounded p-1 text-xs text-zinc-500 opacity-0 transition-opacity hover:text-red-400 focus-visible:opacity-100 group-hover:opacity-100"
                >
                  🗑
                </button>
              </li>
            );
          })}
        </ul>
      </div>

      <div className="border-t border-surface-700 p-3">
        <div className="flex items-center justify-between gap-2">
          <div className="min-w-0">
            <p className="truncate text-sm font-medium text-zinc-200">
              {user.name || user.email.split("@")[0]}
            </p>
            <p className="truncate text-xs text-zinc-500">{user.email}</p>
          </div>
          <button
            onClick={logout}
            disabled={loggingOut}
            className="shrink-0 rounded-lg border border-surface-600 px-3 py-1.5 text-xs text-zinc-300 transition-colors hover:bg-surface-700"
          >
            {loggingOut ? "…" : "Log out"}
          </button>
        </div>
      </div>
    </div>
  );

  return (
    <>
      {/* Mobile top bar */}
      <div className="fixed inset-x-0 top-0 z-30 flex h-14 items-center gap-3 border-b border-surface-700 bg-surface-950/90 px-4 backdrop-blur md:hidden">
        <button
          onClick={() => setOpen(true)}
          aria-label="Open menu"
          className="rounded-md p-2 text-zinc-300 hover:bg-surface-700"
        >
          ☰
        </button>
        <span className="font-semibold text-white">FrameFlow</span>
      </div>
      {/* Mobile drawer */}
      {open && (
        <div className="fixed inset-0 z-50 md:hidden">
          <div
            className="absolute inset-0 bg-black/60"
            onClick={() => setOpen(false)}
            aria-hidden
          />
          <aside className="absolute inset-y-0 left-0 w-72 border-r border-surface-700 bg-surface-900">
            {nav}
          </aside>
        </div>
      )}

      {/* Desktop sidebar */}
      <aside className="hidden w-64 shrink-0 border-r border-surface-700 bg-surface-900 md:block">
        {nav}
      </aside>
    </>
  );
}

function NavLink({
  href,
  active,
  children,
}: {
  href: string;
  active: boolean;
  children: React.ReactNode;
}) {
  return (
    <Link
      href={href}
      className={`block rounded-lg px-2.5 py-2 text-sm transition-colors ${
        active ? "bg-surface-700 text-white" : "text-zinc-400 hover:bg-surface-800 hover:text-zinc-200"
      }`}
    >
      {children}
    </Link>
  );
}
