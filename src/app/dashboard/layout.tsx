import { redirect } from "next/navigation";
import { getCurrentUser } from "@/lib/auth";
import { db } from "@/lib/db";
import { isMockProvider } from "@/lib/env";
import { Sidebar } from "@/components/sidebar";
import { toConversationDto } from "@/lib/dto";

export default async function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const user = await getCurrentUser();
  if (!user) redirect("/login");

  const conversations = await db.conversation.findMany({
    where: { userId: user.id },
    orderBy: { updatedAt: "desc" },
    take: 30,
  });

  return (
    <div className="flex h-dvh overflow-hidden">
      <Sidebar
        user={{ email: user.email, name: user.name }}
        conversations={conversations.map(toConversationDto)}
      />
      <div className="flex min-w-0 flex-1 flex-col pt-14 md:pt-0">
        {isMockProvider() && (
          <div className="border-b border-amber-800/40 bg-amber-950/40 px-4 py-1.5 text-center text-xs text-amber-300">
            Development mock mode — videos are simulated with a sample clip, no real
            provider is called. Configure VIDEO_PROVIDER for real generations.
          </div>
        )}
        <main className="min-h-0 flex-1 overflow-y-auto">{children}</main>
      </div>
    </div>
  );
}
