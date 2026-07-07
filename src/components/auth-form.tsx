"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useState } from "react";
import { api } from "@/lib/client-api";
import { Button, Input, Label } from "@/components/ui";

export function AuthForm({ mode }: { mode: "login" | "signup" }) {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);

    if (mode === "signup") {
      if (password.length < 8) {
        setError("Password must be at least 8 characters.");
        return;
      }
      if (!/[a-zA-Z]/.test(password) || !/[0-9]/.test(password)) {
        setError("Password must contain at least one letter and one number.");
        return;
      }
    }

    setLoading(true);
    try {
      await api(`/api/auth/${mode}`, {
        method: "POST",
        json: mode === "signup" ? { email, password, name: name || undefined } : { email, password },
      });
      const next = searchParams.get("next");
      router.push(next && next.startsWith("/") ? next : "/dashboard");
      router.refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
      setLoading(false);
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center px-4">
      <div className="w-full max-w-sm">
        <Link href="/" className="mb-8 flex items-center justify-center gap-2 text-xl font-bold text-white">
          <span className="flex size-8 items-center justify-center rounded-lg bg-gradient-to-br from-accent-500 to-cyan-500 text-sm font-black">
            F
          </span>
          FrameFlow
        </Link>
        <div className="rounded-2xl border border-surface-700 bg-surface-900 p-6 sm:p-8">
          <h1 className="text-xl font-semibold text-white">
            {mode === "login" ? "Welcome back" : "Create your account"}
          </h1>
          <p className="mt-1 text-sm text-zinc-500">
            {mode === "login"
              ? "Log in to continue creating videos."
              : "Start turning descriptions into videos."}
          </p>

          <form onSubmit={onSubmit} className="mt-6 space-y-4" noValidate>
            {mode === "signup" && (
              <div>
                <Label htmlFor="name">Name (optional)</Label>
                <Input
                  id="name"
                  autoComplete="name"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="Ada Lovelace"
                  maxLength={80}
                />
              </div>
            )}
            <div>
              <Label htmlFor="email">Email</Label>
              <Input
                id="email"
                type="email"
                autoComplete="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@example.com"
              />
            </div>
            <div>
              <Label htmlFor="password">Password</Label>
              <Input
                id="password"
                type="password"
                autoComplete={mode === "login" ? "current-password" : "new-password"}
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder={mode === "signup" ? "8+ characters, letters & numbers" : "••••••••"}
              />
            </div>

            {error && (
              <p role="alert" className="rounded-lg border border-red-800/60 bg-red-950/60 px-3 py-2 text-sm text-red-200">
                {error}
              </p>
            )}

            <Button type="submit" loading={loading} className="w-full" size="lg">
              {mode === "login" ? "Log in" : "Sign up"}
            </Button>
          </form>

          <p className="mt-6 text-center text-sm text-zinc-500">
            {mode === "login" ? (
              <>
                No account?{" "}
                <Link href="/signup" className="text-accent-400 hover:underline">
                  Sign up free
                </Link>
              </>
            ) : (
              <>
                Already have an account?{" "}
                <Link href="/login" className="text-accent-400 hover:underline">
                  Log in
                </Link>
              </>
            )}
          </p>
        </div>
      </div>
    </div>
  );
}
