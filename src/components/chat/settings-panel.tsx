"use client";

import {
  ASPECT_RATIOS,
  CAMERA_MOVEMENTS,
  DURATIONS,
  QUALITIES,
  STYLE_PRESETS,
  estimateCredits,
  type DraftSettings,
} from "@/lib/constants";
import { Button, Input, Label, Select, Textarea } from "@/components/ui";

export interface DraftState {
  prompt: string;
  settings: DraftSettings;
}

/** Generation settings form bound to the shared chat draft. */
export function SettingsPanel({
  draft,
  onChange,
  onGenerate,
  generating,
  validationError,
}: {
  draft: DraftState;
  onChange: (draft: DraftState) => void;
  onGenerate: () => void;
  generating: boolean;
  validationError: string | null;
}) {
  const s = draft.settings;
  const set = (patch: Partial<DraftSettings>) =>
    onChange({ ...draft, settings: { ...s, ...patch } });

  return (
    <div className="flex h-full flex-col gap-4 overflow-y-auto p-4">
      <div>
        <Label htmlFor="gen-prompt">Prompt</Label>
        <Textarea
          id="gen-prompt"
          rows={4}
          maxLength={2000}
          placeholder="Describe your video…"
          value={draft.prompt}
          onChange={(e) => onChange({ ...draft, prompt: e.target.value })}
        />
      </div>

      <div>
        <Label htmlFor="gen-negative">Negative prompt (optional)</Label>
        <Textarea
          id="gen-negative"
          rows={2}
          maxLength={1000}
          placeholder="Things to avoid, e.g. blurry, low quality, text"
          value={s.negativePrompt ?? ""}
          onChange={(e) => set({ negativePrompt: e.target.value || undefined })}
        />
      </div>

      <div>
        <Label>Aspect ratio</Label>
        <div className="grid grid-cols-3 gap-2" role="radiogroup" aria-label="Aspect ratio">
          {ASPECT_RATIOS.map((ratio) => (
            <button
              key={ratio}
              type="button"
              role="radio"
              aria-checked={s.aspectRatio === ratio}
              onClick={() => set({ aspectRatio: ratio })}
              className={`rounded-lg border px-2 py-2 text-xs font-medium transition-colors ${
                s.aspectRatio === ratio
                  ? "border-accent-500 bg-accent-600/20 text-accent-300"
                  : "border-surface-600 bg-surface-800 text-zinc-400 hover:border-surface-500"
              }`}
            >
              <span
                aria-hidden
                className={`mx-auto mb-1 block rounded-sm border border-current opacity-70 ${
                  ratio === "16:9" ? "h-3 w-5" : ratio === "9:16" ? "h-5 w-3" : "size-4"
                }`}
              />
              {ratio}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div>
          <Label htmlFor="gen-duration">Duration</Label>
          <Select
            id="gen-duration"
            value={s.durationSec}
            onChange={(e) => set({ durationSec: Number(e.target.value) })}
          >
            {DURATIONS.map((d) => (
              <option key={d} value={d}>
                {d} seconds
              </option>
            ))}
          </Select>
        </div>
        <div>
          <Label htmlFor="gen-quality">Quality</Label>
          <Select
            id="gen-quality"
            value={s.quality}
            onChange={(e) => set({ quality: e.target.value as DraftSettings["quality"] })}
          >
            {QUALITIES.map((q) => (
              <option key={q} value={q}>
                {q[0].toUpperCase() + q.slice(1)}
              </option>
            ))}
          </Select>
        </div>
      </div>

      <div>
        <Label htmlFor="gen-style">Style preset</Label>
        <Select
          id="gen-style"
          value={s.style}
          onChange={(e) => set({ style: e.target.value })}
        >
          {STYLE_PRESETS.map((p) => (
            <option key={p.id} value={p.id}>
              {p.label}
            </option>
          ))}
        </Select>
      </div>

      <div>
        <Label htmlFor="gen-camera">Camera movement</Label>
        <Select
          id="gen-camera"
          value={s.cameraMovement}
          onChange={(e) => set({ cameraMovement: e.target.value })}
        >
          {CAMERA_MOVEMENTS.map((c) => (
            <option key={c.id} value={c.id}>
              {c.label}
            </option>
          ))}
        </Select>
      </div>

      <div>
        <Label htmlFor="gen-motion">
          Motion intensity: <span className="text-zinc-200">{s.motionStrength}</span>
        </Label>
        <input
          id="gen-motion"
          type="range"
          min={1}
          max={10}
          step={1}
          value={s.motionStrength}
          onChange={(e) => set({ motionStrength: Number(e.target.value) })}
          className="w-full accent-[--color-accent-500]"
        />
        <div className="flex justify-between text-[10px] text-zinc-600">
          <span>Subtle</span>
          <span>Wild</span>
        </div>
      </div>

      <div>
        <Label htmlFor="gen-seed">Seed (optional)</Label>
        <Input
          id="gen-seed"
          type="number"
          min={0}
          max={2147483647}
          placeholder="Random"
          value={s.seed ?? ""}
          onChange={(e) =>
            set({ seed: e.target.value === "" ? undefined : Number(e.target.value) })
          }
        />
      </div>

      {validationError && (
        <p role="alert" className="rounded-lg border border-red-800/60 bg-red-950/60 px-3 py-2 text-xs text-red-200">
          {validationError}
        </p>
      )}

      <div className="mt-auto space-y-2 border-t border-surface-700 pt-4">
        <p className="text-center text-[11px] text-zinc-500">
          Estimated cost: ~{estimateCredits(s.durationSec, s.quality)} credit
          {estimateCredits(s.durationSec, s.quality) === 1 ? "" : "s"} (billing placeholder)
        </p>
        <Button
          onClick={onGenerate}
          loading={generating}
          disabled={draft.prompt.trim().length < 3}
          className="w-full"
          size="lg"
        >
          🎬 Generate video
        </Button>
      </div>
    </div>
  );
}
