"use client";

import { useState } from "react";
import { api } from "@/lib/client-api";
import { useToast } from "@/components/toast";
import { Button, Input, Label } from "@/components/ui";

export function AccountSettings({
  user,
  providerInfo,
}: {
  user: { email: string; name: string | null; createdAt: string };
  providerInfo: { provider: string; llmEnhancement: boolean };
}) {
  const { toast } = useToast();
  const [name, setName] = useState(user.name ?? "");
  const [savingName, setSavingName] = useState(false);

  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [savingPassword, setSavingPassword] = useState(false);
  const [passwordError, setPasswordError] = useState<string | null>(null);

  async function saveName(e: React.FormEvent) {
    e.preventDefault();
    setSavingName(true);
    try {
      await api("/api/account", { method: "PATCH", json: { name } });
      toast("Profile updated", "success");
    } catch (err) {
      toast(err instanceof Error ? err.message : "Update failed", "error");
    } finally {
      setSavingName(false);
    }
  }

  async function savePassword(e: React.FormEvent) {
    e.preventDefault();
    setPasswordError(null);
    if (newPassword !== confirmPassword) {
      setPasswordError("New passwords don't match.");
      return;
    }
    if (newPassword.length < 8 || !/[a-zA-Z]/.test(newPassword) || !/[0-9]/.test(newPassword)) {
      setPasswordError("New password needs 8+ characters with letters and numbers.");
      return;
    }
    setSavingPassword(true);
    try {
      await api("/api/account", {
        method: "PATCH",
        json: { currentPassword, newPassword },
      });
      setCurrentPassword("");
      setNewPassword("");
      setConfirmPassword("");
      toast("Password changed — other sessions were signed out", "success");
    } catch (err) {
      setPasswordError(err instanceof Error ? err.message : "Password change failed");
    } finally {
      setSavingPassword(false);
    }
  }

  return (
    <div className="mx-auto max-w-2xl px-4 py-8 sm:px-6">
      <h1 className="text-2xl font-bold text-white">Account</h1>
      <p className="mt-1 text-sm text-zinc-400">
        Member since {new Date(user.createdAt).toLocaleDateString()}
      </p>

      <section className="mt-8 rounded-2xl border border-surface-700 bg-surface-900 p-6">
        <h2 className="font-semibold text-white">Profile</h2>
        <form onSubmit={saveName} className="mt-4 space-y-4">
          <div>
            <Label htmlFor="acct-email">Email</Label>
            <Input id="acct-email" value={user.email} disabled className="opacity-60" />
          </div>
          <div>
            <Label htmlFor="acct-name">Display name</Label>
            <Input
              id="acct-name"
              value={name}
              maxLength={80}
              onChange={(e) => setName(e.target.value)}
              placeholder="Your name"
            />
          </div>
          <Button type="submit" loading={savingName}>
            Save profile
          </Button>
        </form>
      </section>

      <section className="mt-6 rounded-2xl border border-surface-700 bg-surface-900 p-6">
        <h2 className="font-semibold text-white">Change password</h2>
        <form onSubmit={savePassword} className="mt-4 space-y-4">
          <div>
            <Label htmlFor="acct-current">Current password</Label>
            <Input
              id="acct-current"
              type="password"
              autoComplete="current-password"
              required
              value={currentPassword}
              onChange={(e) => setCurrentPassword(e.target.value)}
            />
          </div>
          <div className="grid gap-4 sm:grid-cols-2">
            <div>
              <Label htmlFor="acct-new">New password</Label>
              <Input
                id="acct-new"
                type="password"
                autoComplete="new-password"
                required
                value={newPassword}
                onChange={(e) => setNewPassword(e.target.value)}
              />
            </div>
            <div>
              <Label htmlFor="acct-confirm">Confirm new password</Label>
              <Input
                id="acct-confirm"
                type="password"
                autoComplete="new-password"
                required
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
              />
            </div>
          </div>
          {passwordError && (
            <p role="alert" className="rounded-lg border border-red-800/60 bg-red-950/60 px-3 py-2 text-sm text-red-200">
              {passwordError}
            </p>
          )}
          <Button type="submit" loading={savingPassword}>
            Change password
          </Button>
        </form>
      </section>

      <section className="mt-6 rounded-2xl border border-surface-700 bg-surface-900 p-6">
        <h2 className="font-semibold text-white">Workspace status</h2>
        <dl className="mt-4 space-y-2 text-sm">
          <div className="flex justify-between">
            <dt className="text-zinc-400">Video provider</dt>
            <dd className="font-medium text-zinc-200">
              {providerInfo.provider === "mock"
                ? "Mock (development only)"
                : providerInfo.provider === "local"
                  ? "Self-hosted local model (no external AI service)"
                  : providerInfo.provider}
            </dd>
          </div>
          <div className="flex justify-between">
            <dt className="text-zinc-400">LLM prompt enhancement</dt>
            <dd className="font-medium text-zinc-200">
              {providerInfo.llmEnhancement ? "Anthropic (enabled)" : "Rule-based (no API key)"}
            </dd>
          </div>
          <div className="flex justify-between">
            <dt className="text-zinc-400">Plan</dt>
            <dd className="font-medium text-zinc-200">Free (billing not connected)</dd>
          </div>
        </dl>
        <p className="mt-4 text-xs text-zinc-500">
          Providers are configured by the server administrator via environment variables —
          API keys are never exposed to the browser.
        </p>
      </section>
    </div>
  );
}
