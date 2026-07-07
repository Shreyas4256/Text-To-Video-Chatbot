import Link from "next/link";
import { STARTER_PROMPTS } from "@/lib/constants";
import { getCurrentUser } from "@/lib/auth";

export default async function LandingPage() {
  const user = await getCurrentUser().catch(() => null);

  return (
    <div className="min-h-screen">
      {/* Nav */}
      <header className="sticky top-0 z-40 border-b border-surface-700/60 bg-surface-950/80 backdrop-blur">
        <div className="mx-auto flex h-16 max-w-6xl items-center justify-between px-4 sm:px-6">
          <Link href="/" className="flex items-center gap-2 text-lg font-bold text-white">
            <Logo />
            FrameFlow
          </Link>
          <nav className="flex items-center gap-2 sm:gap-4">
            <a href="#features" className="hidden text-sm text-zinc-400 hover:text-zinc-100 sm:block">
              Features
            </a>
            <a href="#how-it-works" className="hidden text-sm text-zinc-400 hover:text-zinc-100 sm:block">
              How it works
            </a>
            <a href="#pricing" className="hidden text-sm text-zinc-400 hover:text-zinc-100 sm:block">
              Pricing
            </a>
            {user ? (
              <Link
                href="/dashboard"
                className="rounded-lg bg-accent-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-accent-500"
              >
                Open dashboard
              </Link>
            ) : (
              <>
                <Link href="/login" className="px-2 text-sm text-zinc-300 hover:text-white">
                  Log in
                </Link>
                <Link
                  href="/signup"
                  className="rounded-lg bg-accent-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-accent-500"
                >
                  Sign up free
                </Link>
              </>
            )}
          </nav>
        </div>
      </header>

      {/* Hero */}
      <section className="relative overflow-hidden">
        <div
          aria-hidden
          className="pointer-events-none absolute inset-0 bg-[radial-gradient(60%_50%_at_50%_0%,rgba(124,58,237,0.25),transparent)]"
        />
        <div className="mx-auto max-w-6xl px-4 pb-20 pt-20 text-center sm:px-6 sm:pt-28">
          <p className="mx-auto mb-4 w-fit rounded-full border border-accent-500/30 bg-accent-600/10 px-4 py-1 text-xs font-medium text-accent-400">
            Chat-first AI video creation
          </p>
          <h1 className="mx-auto max-w-3xl text-4xl font-bold tracking-tight text-white sm:text-6xl">
            Describe it.{" "}
            <span className="bg-gradient-to-r from-accent-400 to-cyan-400 bg-clip-text text-transparent">
              Watch it.
            </span>
          </h1>
          <p className="mx-auto mt-6 max-w-2xl text-lg text-zinc-400">
            FrameFlow is an AI assistant that turns plain-language descriptions into real
            generated video clips. Chat to refine your idea, pick your format, and get a
            downloadable video — no editing skills required.
          </p>
          <div className="mt-8 flex flex-wrap items-center justify-center gap-4">
            <Link
              href={user ? "/dashboard" : "/signup"}
              className="rounded-lg bg-accent-600 px-8 py-3.5 text-base font-semibold text-white shadow-xl shadow-accent-600/25 transition-colors hover:bg-accent-500"
            >
              Create a video
            </Link>
            <a
              href="#how-it-works"
              className="rounded-lg border border-surface-600 px-8 py-3.5 text-base font-medium text-zinc-300 transition-colors hover:bg-surface-800"
            >
              See how it works
            </a>
          </div>

          {/* Chat preview (static illustration of the product) */}
          <div className="mx-auto mt-16 max-w-2xl rounded-2xl border border-surface-600 bg-surface-900/80 p-4 text-left shadow-2xl sm:p-6">
            <div className="mb-4 flex items-center gap-2 border-b border-surface-700 pb-3">
              <span className="size-2.5 rounded-full bg-red-500/70" />
              <span className="size-2.5 rounded-full bg-amber-500/70" />
              <span className="size-2.5 rounded-full bg-emerald-500/70" />
              <span className="ml-2 text-xs text-zinc-500">FrameFlow chat — preview</span>
            </div>
            <div className="space-y-3 text-sm">
              <div className="ml-auto w-fit max-w-[85%] rounded-2xl rounded-br-sm bg-accent-600 px-4 py-2.5 text-white">
                Create a cinematic 10-second video of a futuristic car driving through neon
                Tokyo at night
              </div>
              <div className="w-fit max-w-[85%] rounded-2xl rounded-bl-sm bg-surface-700 px-4 py-2.5 text-zinc-200">
                Nice concept! I&apos;ve loaded it into the generation panel (16:9 · 10s ·
                Cinematic). Tell me things like &ldquo;make it vertical for Reels&rdquo; or hit
                Generate when you&apos;re ready.
              </div>
              <div className="ml-auto w-fit max-w-[85%] rounded-2xl rounded-br-sm bg-accent-600 px-4 py-2.5 text-white">
                Make it vertical for Instagram Reels
              </div>
              <div className="w-fit max-w-[85%] rounded-2xl rounded-bl-sm bg-surface-700 px-4 py-2.5 text-zinc-200">
                Done — switched to vertical 9:16 (great for Reels/Shorts). Ready to generate?
              </div>
              <div className="flex items-center gap-3 rounded-xl border border-surface-600 bg-surface-800 px-4 py-3">
                <span className="flex size-9 items-center justify-center rounded-lg bg-accent-600/20 text-accent-400">
                  ▶
                </span>
                <div>
                  <p className="font-medium text-zinc-100">neon-tokyo-drive.mp4</p>
                  <p className="text-xs text-emerald-400">Completed · 9:16 · 10s</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features */}
      <section id="features" className="mx-auto max-w-6xl px-4 py-20 sm:px-6">
        <h2 className="text-center text-3xl font-bold text-white">
          Everything you need to go from idea to clip
        </h2>
        <div className="mt-12 grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {FEATURES.map((f) => (
            <div
              key={f.title}
              className="rounded-xl border border-surface-700 bg-surface-900 p-6 transition-colors hover:border-accent-500/40"
            >
              <div className="mb-3 text-2xl">{f.icon}</div>
              <h3 className="font-semibold text-white">{f.title}</h3>
              <p className="mt-2 text-sm leading-relaxed text-zinc-400">{f.description}</p>
            </div>
          ))}
        </div>
      </section>

      {/* How it works */}
      <section id="how-it-works" className="border-y border-surface-700/60 bg-surface-900/50">
        <div className="mx-auto max-w-6xl px-4 py-20 sm:px-6">
          <h2 className="text-center text-3xl font-bold text-white">How it works</h2>
          <div className="mt-12 grid gap-8 md:grid-cols-4">
            {STEPS.map((s, i) => (
              <div key={s.title} className="relative">
                <div className="mb-4 flex size-10 items-center justify-center rounded-full bg-accent-600/20 font-bold text-accent-400">
                  {i + 1}
                </div>
                <h3 className="font-semibold text-white">{s.title}</h3>
                <p className="mt-2 text-sm text-zinc-400">{s.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Example prompts */}
      <section className="mx-auto max-w-6xl px-4 py-20 sm:px-6">
        <h2 className="text-center text-3xl font-bold text-white">Start from a prompt like…</h2>
        <div className="mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {STARTER_PROMPTS.map((p) => (
            <Link
              key={p}
              href={user ? "/dashboard" : "/signup"}
              className="group rounded-xl border border-surface-700 bg-surface-900 p-5 text-sm text-zinc-300 transition-colors hover:border-accent-500/50 hover:text-white"
            >
              <span className="mb-2 block text-accent-400">✦</span>
              {p}
              <span className="mt-3 block text-xs text-zinc-500 group-hover:text-accent-400">
                Use this prompt →
              </span>
            </Link>
          ))}
        </div>
      </section>

      {/* Pricing placeholder */}
      <section id="pricing" className="border-t border-surface-700/60 bg-surface-900/50">
        <div className="mx-auto max-w-6xl px-4 py-20 sm:px-6">
          <h2 className="text-center text-3xl font-bold text-white">Simple pricing</h2>
          <p className="mx-auto mt-3 max-w-xl text-center text-sm text-zinc-500">
            Billing isn&apos;t connected yet — these tiers show where payments will plug in.
          </p>
          <div className="mt-12 grid gap-6 md:grid-cols-3">
            {PRICING.map((tier) => (
              <div
                key={tier.name}
                className={
                  tier.highlight
                    ? "rounded-2xl border border-accent-500/60 bg-surface-900 p-6 shadow-xl shadow-accent-600/10"
                    : "rounded-2xl border border-surface-700 bg-surface-900 p-6"
                }
              >
                <h3 className="font-semibold text-white">{tier.name}</h3>
                <p className="mt-2 text-3xl font-bold text-white">
                  {tier.price}
                  <span className="text-sm font-normal text-zinc-500">/month</span>
                </p>
                <ul className="mt-4 space-y-2 text-sm text-zinc-400">
                  {tier.features.map((feat) => (
                    <li key={feat} className="flex gap-2">
                      <span className="text-accent-400">✓</span>
                      {feat}
                    </li>
                  ))}
                </ul>
                <Link
                  href={user ? "/dashboard" : "/signup"}
                  className={
                    tier.highlight
                      ? "mt-6 block rounded-lg bg-accent-600 py-2.5 text-center text-sm font-medium text-white transition-colors hover:bg-accent-500"
                      : "mt-6 block rounded-lg border border-surface-600 py-2.5 text-center text-sm font-medium text-zinc-300 transition-colors hover:bg-surface-800"
                  }
                >
                  Get started
                </Link>
              </div>
            ))}
          </div>
        </div>
      </section>

      <footer className="border-t border-surface-700/60">
        <div className="mx-auto flex max-w-6xl flex-col items-center justify-between gap-3 px-4 py-8 text-sm text-zinc-500 sm:flex-row sm:px-6">
          <span className="flex items-center gap-2">
            <Logo /> FrameFlow — AI text-to-video chatbot
          </span>
          <span>Built with Next.js · Bring your own video provider</span>
        </div>
      </footer>
    </div>
  );
}

function Logo() {
  return (
    <span
      aria-hidden
      className="flex size-7 items-center justify-center rounded-lg bg-gradient-to-br from-accent-500 to-cyan-500 text-sm font-black text-white"
    >
      F
    </span>
  );
}

const FEATURES = [
  {
    icon: "💬",
    title: "Conversational creation",
    description:
      "Describe your video in plain language. The assistant refines your prompt, suggests improvements and applies follow-up edits like “make it vertical”.",
  },
  {
    icon: "🎬",
    title: "Real video generation",
    description:
      "Run an open-source model fully on your own hardware — or plug in a cloud provider. Live queued → processing → completed status either way.",
  },
  {
    icon: "🎛️",
    title: "Full creative control",
    description:
      "Aspect ratio, duration, style presets, camera movement, motion strength, seed and negative prompts — all in a clean settings panel.",
  },
  {
    icon: "📚",
    title: "Video library",
    description:
      "Every generation is saved with its prompt and settings. Search, filter, sort, replay, download, regenerate or delete.",
  },
  {
    icon: "🔒",
    title: "Private & secure",
    description:
      "Your account, chats and videos are protected by secure sessions. Provider API keys never leave the server.",
  },
  {
    icon: "🧩",
    title: "Self-hostable",
    description:
      "A bundled local inference worker runs the model on your machine with zero external AI APIs. Cloud adapters stay optional.",
  },
];

const STEPS = [
  {
    title: "Describe your idea",
    description: "Type what you want to see, exactly like you'd tell a friend.",
  },
  {
    title: "Refine in chat",
    description:
      "Ask for changes — style, location, duration, format. The assistant updates your generation setup live.",
  },
  {
    title: "Generate",
    description:
      "One click sends the request to a real AI video model. Track progress right in the chat.",
  },
  {
    title: "Download & share",
    description: "Replay the clip, download the MP4, or regenerate with tweaks.",
  },
];

const PRICING = [
  {
    name: "Starter",
    price: "$0",
    features: ["Mock/dev mode", "Chat prompt refinement", "Video library", "Community support"],
    highlight: false,
  },
  {
    name: "Creator",
    price: "$19",
    features: [
      "Real provider generations",
      "HD quality",
      "All style presets",
      "Priority queue (coming soon)",
    ],
    highlight: true,
  },
  {
    name: "Studio",
    price: "$49",
    features: [
      "Everything in Creator",
      "Longer durations",
      "Team workspaces (coming soon)",
      "API access (coming soon)",
    ],
    highlight: false,
  },
];
