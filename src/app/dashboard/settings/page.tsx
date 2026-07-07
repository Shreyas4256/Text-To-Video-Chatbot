import type { Metadata } from "next";
import { redirect } from "next/navigation";
import { getCurrentUser } from "@/lib/auth";
import { getEnv } from "@/lib/env";
import { AccountSettings } from "@/components/account-settings";

export const metadata: Metadata = { title: "Account" };

export default async function SettingsPage() {
  const user = await getCurrentUser();
  if (!user) redirect("/login");
  const env = getEnv();

  return (
    <AccountSettings
      user={{ email: user.email, name: user.name, createdAt: user.createdAt.toISOString() }}
      providerInfo={{
        provider: env.VIDEO_PROVIDER,
        llmEnhancement: Boolean(env.ANTHROPIC_API_KEY),
      }}
    />
  );
}
