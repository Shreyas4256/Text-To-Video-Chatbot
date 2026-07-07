# FrameFlow — AI Text-to-Video Chatbot

FrameFlow is a full-stack AI video-creation assistant. Describe a video in plain
language ("a futuristic car driving through neon Tokyo at night"), refine it in
chat ("make it vertical for Reels", "use anime style", "make it 15 seconds"),
and generate a real video through a configurable text-to-video provider — with
live status tracking, a personal video library, and secure authentication.

Built with **Next.js 15 (App Router) · TypeScript · Tailwind CSS 4 · Prisma ·
PostgreSQL**.

## Features

- **Chat-first workflow** — an assistant that parses follow-up instructions
  (aspect ratio, duration, style, camera, motion, location changes) and keeps a
  live generation draft in sync with the settings panel.
- **Real provider integration** — pluggable adapter layer with implementations
  for [Replicate](https://replicate.com) and [fal.ai](https://fal.ai), plus a
  clearly-labeled development-only mock. Production refuses to run in mock mode.
- **Optional LLM prompt enhancement** — with an `ANTHROPIC_API_KEY`, prompts are
  rewritten by Claude before generation; otherwise a deterministic rule-based
  enhancer is used.
- **Full generation lifecycle** — queued → processing → completed/failed, with
  poll-on-read status sync, optional provider webhooks, retry, regenerate,
  download (authenticated proxy), and delete.
- **Video library** — search, status filter, sorting, cursor pagination, detail
  view with settings and enhanced prompt.
- **Secure auth** — email/password with bcrypt, DB-backed hashed session
  tokens, HTTP-only cookies, protected routes, per-user authorization on every
  resource, rate limiting on auth/message/generation endpoints.
- **Polished dark UI** — responsive (mobile drawer/sheet), skeleton loaders,
  toasts, empty states, keyboard accessible.

## Project structure

```
prisma/schema.prisma          Database models + migrations
src/
  middleware.ts               Cookie gate for /dashboard, /login, /signup
  app/
    page.tsx                  Landing page
    login/ signup/            Auth pages
    dashboard/                Protected app (overview, chat/[id], library, settings)
    api/
      auth/                   signup, login, logout, me
      account/                Profile + password updates
      conversations/          CRUD + chat turns (assistant runs server-side)
      generations/            Create/list, status (poll-on-read), retry, download, delete
      webhooks/video/         Provider webhook receiver (signal-only, re-verified)
  components/                 UI (chat view, settings panel, library, sidebar, toasts…)
  lib/
    env.ts                    Zod-validated environment (fails fast, mock banned in prod)
    auth.ts / password.ts     Sessions + bcrypt
    assistant.ts              Chat assistant + prompt enhancement (LLM or rule-based)
    generation-service.ts     Idempotent job submission + status sync
    rate-limit.ts             In-memory sliding-window limiter
    video-providers/
      provider-interface.ts   Adapter contract
      replicate.ts fal.ts     Real providers
      mock.ts                 Dev-only simulated provider
      provider-factory.ts     Selection via VIDEO_PROVIDER
```

## Setup

### 1. Install

```bash
npm install
```

Requirements: Node 20+, PostgreSQL 14+.

### 2. Configure environment

```bash
cp .env.example .env
```

| Variable | Required | Purpose |
| --- | --- | --- |
| `DATABASE_URL` | ✅ | PostgreSQL connection string |
| `SESSION_SECRET` | ✅ | ≥32 chars; hashes session tokens (`openssl rand -hex 32`) |
| `VIDEO_PROVIDER` | ✅ | `mock` (dev only), `replicate`, or `fal` |
| `REPLICATE_API_TOKEN` | with `replicate` | Replicate API token |
| `REPLICATE_VIDEO_MODEL` | optional | Default `wan-video/wan-2.2-t2v-fast` |
| `FAL_KEY` | with `fal` | fal.ai API key |
| `FAL_VIDEO_MODEL` | optional | Default `fal-ai/ltx-video` |
| `ANTHROPIC_API_KEY` | optional | Enables LLM prompt enhancement |
| `ANTHROPIC_MODEL` | optional | Default `claude-haiku-4-5` |
| `APP_URL` + `WEBHOOK_SECRET` | optional | Enables provider → app webhooks |

Environment is validated with Zod at first use — misconfiguration fails with a
readable error listing exactly what's wrong.

### 3. Database

```bash
# create a database, e.g.:
createdb frameflow
# apply migrations + generate the Prisma client:
npx prisma migrate deploy   # (or `npx prisma migrate dev` during development)
```

### 4. Run locally

```bash
npm run dev
# open http://localhost:3000
```

## Mock mode (development without an API key)

Set `VIDEO_PROVIDER="mock"`. Generations run through the full real pipeline
(database records, provider request rows, queued → processing → completed) but
are fulfilled after ~16 s with a bundled synthetic sample clip
(`public/mock/sample-video.mp4`). A persistent banner marks mock mode in the
dashboard. Include `[fail]` in a prompt to exercise the failure/retry path.

**Mock mode is refused in production** — `next start` with
`VIDEO_PROVIDER=mock` fails env validation by design.

## Connecting a real provider

### Replicate

1. Create a token at <https://replicate.com/account/api-tokens> (usage is paid).
2. `.env`:
   ```
   VIDEO_PROVIDER="replicate"
   REPLICATE_API_TOKEN="r8_..."
   REPLICATE_VIDEO_MODEL="wan-video/wan-2.2-t2v-fast"   # or another t2v model
   ```

### fal.ai

1. Create a key at <https://fal.ai/dashboard/keys> (usage is paid).
2. `.env`:
   ```
   VIDEO_PROVIDER="fal"
   FAL_KEY="..."
   FAL_VIDEO_MODEL="fal-ai/ltx-video"                   # or e.g. a Kling model
   ```

Input mapping sends the common denominator (`prompt`, `negative_prompt`,
`aspect_ratio`, `seed`). Some models expose extra parameters (duration,
resolution, camera control) under different names — adjust the adapter in
`src/lib/video-providers/` for your chosen model if needed. Provider files are
small and isolated; adding a new provider means implementing the
`VideoProvider` interface and registering it in `provider-factory.ts`.

Status updates use **poll-on-read** (the status endpoint re-checks the
provider, throttled to once per 3 s per job) — no worker process needed, which
keeps the app deployable on serverless. If `APP_URL` + `WEBHOOK_SECRET` are
set, provider webhooks are registered too; the webhook is treated as a signal
only and the status is always re-verified against the provider API.

## Production deployment

```bash
npm run build
npm start
```

- Works on any Node host (VPS, Docker, Railway, Render) and on Vercel.
- Provide a managed PostgreSQL (`DATABASE_URL`) and run
  `npx prisma migrate deploy` during release.
- Set a real `VIDEO_PROVIDER` (mock is rejected), a strong `SESSION_SECRET`,
  and — if you want webhooks — a public `APP_URL` + `WEBHOOK_SECRET`.
- The in-memory rate limiter is per-instance; for multi-instance deployments
  swap `src/lib/rate-limit.ts` for a shared store (e.g. Upstash Redis) behind
  the same function signature.
- Generated video URLs are stored as returned by the provider. Some providers
  expire their URLs — for long-term storage, copy completed videos to object
  storage (S3/R2) in `generation-service.ts` where the `COMPLETED` transition
  is persisted.

## Scripts

| Command | Purpose |
| --- | --- |
| `npm run dev` | Development server |
| `npm run build` | Production build |
| `npm start` | Production server |
| `npm run lint` | ESLint |
| `npx tsc --noEmit` | Typecheck |
| `npx prisma studio` | Browse the database |
