import Link from "next/link";
import type { Metadata } from "next";
import { redirect } from "next/navigation";
import { db } from "@/lib/db";
import { getCurrentUser } from "@/lib/auth";
import { generationInclude } from "@/lib/generation-service";
import { toGenerationDto } from "@/lib/dto";
import { STARTER_PROMPTS } from "@/lib/constants";
import { NewChatButton } from "@/components/new-chat-button";
import { StatusBadge } from "@/components/ui";

export const metadata: Metadata = { title: "Dashboard" };

export default async function DashboardHome() {
  const user = await getCurrentUser();
  if (!user) redirect("/login");

  const [recent, counts] = await Promise.all([
    db.videoGeneration.findMany({
      where: { userId: user.id },
      include: generationInclude,
      orderBy: { createdAt: "desc" },
      take: 6,
    }),
    db.videoGeneration.groupBy({
      by: ["status"],
      where: { userId: user.id },
      _count: true,
    }),
  ]);

  const total = counts.reduce((sum, c) => sum + c._count, 0);
  const completed = counts.find((c) => c.status === "COMPLETED")?._count ?? 0;
  const pending = counts
    .filter((c) => c.status === "QUEUED" || c.status === "PROCESSING")
    .reduce((sum, c) => sum + c._count, 0);

  return (
    <div className="mx-auto max-w-5xl px-4 py-8 sm:px-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white">
            Hey {user.name || user.email.split("@")[0]} 👋
          </h1>
          <p className="mt-1 text-sm text-zinc-400">
            Describe a video and let the assistant bring it to life.
          </p>
        </div>
        <NewChatButton>＋ New video</NewChatButton>
      </div>

      {/* Stats */}
      <div className="mt-8 grid grid-cols-3 gap-4">
        <Stat label="Total generations" value={total} />
        <Stat label="Completed videos" value={completed} />
        <Stat label="In progress" value={pending} />
      </div>

      {/* Starter prompts */}
      <h2 className="mt-10 text-sm font-semibold uppercase tracking-wider text-zinc-500">
        Try a starter prompt
      </h2>
      <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {STARTER_PROMPTS.slice(0, 6).map((p) => (
          <StarterCard key={p} prompt={p} />
        ))}
      </div>

      {/* Recent generations */}
      <div className="mt-10 flex items-center justify-between">
        <h2 className="text-sm font-semibold uppercase tracking-wider text-zinc-500">
          Recent generations
        </h2>
        <Link href="/dashboard/library" className="text-sm text-accent-400 hover:underline">
          View library →
        </Link>
      </div>
      {recent.length === 0 ? (
        <p className="mt-4 rounded-xl border border-dashed border-surface-600 px-6 py-10 text-center text-sm text-zinc-500">
          Nothing generated yet. Start a chat and create your first video!
        </p>
      ) : (
        <ul className="mt-4 space-y-2">
          {recent.map((g) => {
            const dto = toGenerationDto(g);
            return (
              <li key={dto.id}>
                <Link
                  href={
                    dto.conversationId
                      ? `/dashboard/chat/${dto.conversationId}`
                      : "/dashboard/library"
                  }
                  className="flex items-center justify-between gap-4 rounded-xl border border-surface-700 bg-surface-900 px-4 py-3 transition-colors hover:border-accent-500/40"
                >
                  <span className="min-w-0 truncate text-sm text-zinc-300">{dto.prompt}</span>
                  <span className="flex shrink-0 items-center gap-3">
                    <span className="hidden text-xs text-zinc-500 sm:block">
                      {new Date(dto.createdAt).toLocaleDateString()}
                    </span>
                    <StatusBadge status={dto.status} />
                  </span>
                </Link>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-xl border border-surface-700 bg-surface-900 p-4 sm:p-5">
      <p className="text-2xl font-bold text-white sm:text-3xl">{value}</p>
      <p className="mt-1 text-xs text-zinc-500 sm:text-sm">{label}</p>
    </div>
  );
}

function StarterCard({ prompt }: { prompt: string }) {
  return (
    <div className="flex flex-col justify-between rounded-xl border border-surface-700 bg-surface-900 p-4">
      <p className="text-sm text-zinc-300">{prompt}</p>
      <div className="mt-3">
        <NewChatButton prompt={prompt} variant="secondary" className="!px-3 !py-1.5 !text-xs">
          Use prompt →
        </NewChatButton>
      </div>
    </div>
  );
}
