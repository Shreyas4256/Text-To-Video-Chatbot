import {
  CAMERA_MOVEMENTS,
  DEFAULT_SETTINGS,
  DURATIONS,
  DraftSettings,
  STYLE_PRESETS,
} from "@/lib/constants";
import { getEnv } from "@/lib/env";

export interface Draft {
  prompt: string;
  settings: DraftSettings;
}

export interface AssistantResult {
  reply: string;
  draft: Draft;
}

/* ------------------------------------------------------------------ */
/* Rule-based instruction parsing                                      */
/* ------------------------------------------------------------------ */

interface ParsedChange {
  description: string;
  apply(draft: Draft): void;
}

function parseInstructions(message: string): ParsedChange[] {
  const text = message.toLowerCase();
  const changes: ParsedChange[] = [];

  // Aspect ratio
  if (/\b(vertical|portrait|9:16|reels?|tiktok|shorts?|instagram stor)/.test(text)) {
    changes.push({
      description: "switched to vertical 9:16 (great for Reels/Shorts)",
      apply: (d) => (d.settings.aspectRatio = "9:16"),
    });
  } else if (/\b(square|1:1)\b/.test(text)) {
    changes.push({
      description: "switched to square 1:1",
      apply: (d) => (d.settings.aspectRatio = "1:1"),
    });
  } else if (/\b(horizontal|landscape|widescreen|16:9|youtube)\b/.test(text)) {
    changes.push({
      description: "switched to widescreen 16:9",
      apply: (d) => (d.settings.aspectRatio = "16:9"),
    });
  }

  // Duration — "15 seconds", "make it 10s"
  const durationMatch = text.match(/\b(\d{1,3})\s*(?:s\b|sec\b|secs\b|seconds?\b|-second)/);
  if (durationMatch) {
    const requested = parseInt(durationMatch[1], 10);
    const closest = [...DURATIONS].reduce((a, b) =>
      Math.abs(b - requested) < Math.abs(a - requested) ? b : a
    );
    changes.push({
      description:
        closest === requested
          ? `set duration to ${closest}s`
          : `set duration to ${closest}s (closest supported to ${requested}s)`,
      apply: (d) => (d.settings.durationSec = closest),
    });
  }

  // Style presets by keyword
  const styleKeywords: Record<string, string> = {
    anime: "anime",
    cartoon: "cartoon",
    cyberpunk: "cyberpunk",
    documentary: "documentary",
    realistic: "realistic",
    photorealistic: "realistic",
    "3d": "3d-animation",
    luxury: "luxury",
    commercial: "luxury",
    advert: "product-ad",
    "product ad": "product-ad",
    reel: "social-reel",
  };
  for (const [keyword, styleId] of Object.entries(styleKeywords)) {
    if (text.includes(keyword)) {
      const preset = STYLE_PRESETS.find((s) => s.id === styleId)!;
      changes.push({
        description: `applied the ${preset.label} style`,
        apply: (d) => (d.settings.style = styleId),
      });
      break;
    }
  }
  // "more cinematic" / "cinematic"
  if (/\bcinematic\b/.test(text) && !changes.some((c) => c.description.includes("style"))) {
    changes.push({
      description: "applied the Cinematic style",
      apply: (d) => (d.settings.style = "cinematic"),
    });
  }

  // Slow motion
  if (/slow[\s-]?mo(tion)?/.test(text)) {
    changes.push({
      description: "added slow motion to the prompt and lowered motion strength",
      apply: (d) => {
        d.settings.motionStrength = Math.min(d.settings.motionStrength, 3);
        if (!/slow motion/i.test(d.prompt)) d.prompt = `${d.prompt}, slow motion`;
      },
    });
  }

  // Faster / more motion
  if (/\b(faster|more (motion|action|dynamic)|energetic)\b/.test(text)) {
    changes.push({
      description: "increased motion strength",
      apply: (d) =>
        (d.settings.motionStrength = Math.min(10, d.settings.motionStrength + 3)),
    });
  }

  // Camera movements by keyword
  for (const cam of CAMERA_MOVEMENTS) {
    if (cam.id !== "static" && text.includes(cam.label.toLowerCase())) {
      changes.push({
        description: `set camera movement to ${cam.label}`,
        apply: (d) => (d.settings.cameraMovement = cam.id),
      });
      break;
    }
  }
  if (/\bdrone\b/.test(text) && !changes.some((c) => c.description.includes("camera"))) {
    changes.push({
      description: "set camera movement to FPV Drone",
      apply: (d) => (d.settings.cameraMovement = "fpv"),
    });
  }

  // Location change: "change the location to Mumbai", "set it in Paris"
  const locationMatch = message.match(
    /(?:change (?:the )?location to|set (?:it|the scene) in|move (?:it|the scene) to)\s+([A-Za-z][A-Za-z\s,'-]{1,60})/i
  );
  if (locationMatch) {
    const location = locationMatch[1].trim().replace(/[.!?]$/, "");
    changes.push({
      description: `moved the scene to ${location}`,
      apply: (d) => {
        d.prompt = `${d.prompt}, set in ${location}`;
      },
    });
  }

  // Quality
  if (/\b(high(er)? quality|best quality|hq)\b/.test(text)) {
    changes.push({
      description: "raised quality to high",
      apply: (d) => (d.settings.quality = "high"),
    });
  } else if (/\b(draft|quick preview|rough)\b/.test(text)) {
    changes.push({
      description: "set quality to draft for a quick preview",
      apply: (d) => (d.settings.quality = "draft"),
    });
  }

  return changes;
}

/** Heuristic: does this message look like a fresh scene description? */
function looksLikeNewScene(message: string, hasExistingPrompt: boolean): boolean {
  if (!hasExistingPrompt) return true;
  const text = message.toLowerCase().trim();
  const editVerbs =
    /^(make|change|set|add|remove|use|switch|turn|more|less|can you|please (make|change|set|add))/;
  return !editVerbs.test(text) && message.trim().split(/\s+/).length >= 5;
}

/* ------------------------------------------------------------------ */
/* Prompt enhancement                                                  */
/* ------------------------------------------------------------------ */

/** Deterministic enhancement used when no LLM key is configured. */
export function enhancePromptRuleBased(prompt: string, settings: DraftSettings): string {
  const style = STYLE_PRESETS.find((s) => s.id === settings.style);
  const camera = CAMERA_MOVEMENTS.find((c) => c.id === settings.cameraMovement);
  const parts = [prompt.trim().replace(/[.!?]+$/, "")];
  if (style) parts.push(style.promptFragment);
  if (camera && camera.id !== "static") parts.push(camera.promptFragment);
  if (settings.motionStrength >= 8) parts.push("high-energy dynamic motion");
  else if (settings.motionStrength <= 2) parts.push("subtle gentle motion");
  parts.push("high detail, smooth motion, professional color grading");
  return parts.join(", ");
}

/** LLM enhancement via the Anthropic API when ANTHROPIC_API_KEY is set. */
async function enhancePromptWithLlm(
  prompt: string,
  settings: DraftSettings
): Promise<string | null> {
  const env = getEnv();
  if (!env.ANTHROPIC_API_KEY) return null;
  try {
    const style = STYLE_PRESETS.find((s) => s.id === settings.style);
    const res = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: {
        "x-api-key": env.ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: env.ANTHROPIC_MODEL,
        max_tokens: 300,
        system:
          "You are a prompt engineer for text-to-video models. Rewrite the user's idea into one vivid, concrete video-generation prompt under 90 words. Describe subject, setting, lighting, motion and mood. Output ONLY the rewritten prompt, no preamble or quotes.",
        messages: [
          {
            role: "user",
            content: `Idea: ${prompt}\nStyle: ${style?.label ?? settings.style}\nCamera: ${settings.cameraMovement}\nAspect ratio: ${settings.aspectRatio}, duration ${settings.durationSec}s.`,
          },
        ],
      }),
    });
    if (!res.ok) {
      console.error(`[assistant] Anthropic enhancement failed: ${res.status}`);
      return null;
    }
    const data = (await res.json()) as { content?: { type: string; text?: string }[] };
    const text = data.content?.find((b) => b.type === "text")?.text?.trim();
    return text && text.length > 10 ? text : null;
  } catch (error) {
    console.error("[assistant] Anthropic enhancement error:", error);
    return null;
  }
}

/** Enhance a prompt: LLM when configured, deterministic fallback otherwise. */
export async function enhancePrompt(
  prompt: string,
  settings: DraftSettings
): Promise<string> {
  const llmResult = await enhancePromptWithLlm(prompt, settings);
  return llmResult ?? enhancePromptRuleBased(prompt, settings);
}

/* ------------------------------------------------------------------ */
/* Chat turns                                                          */
/* ------------------------------------------------------------------ */

function describeSettings(s: DraftSettings): string {
  const style = STYLE_PRESETS.find((p) => p.id === s.style)?.label ?? s.style;
  return `${s.aspectRatio} · ${s.durationSec}s · ${style} · ${s.quality} quality`;
}

/**
 * Produce the assistant's reply for a chat turn and the updated draft
 * (prompt + settings) shown in the settings panel. This never claims a
 * video is being generated — generation only starts when the user submits
 * the draft via the Generate action.
 */
export async function runAssistantTurn(
  message: string,
  currentDraft: Draft | null
): Promise<AssistantResult> {
  const draft: Draft = currentDraft
    ? {
        prompt: currentDraft.prompt,
        settings: { ...currentDraft.settings },
      }
    : { prompt: "", settings: { ...DEFAULT_SETTINGS } };

  const changes = parseInstructions(message);
  const isNewScene = looksLikeNewScene(message, draft.prompt.length > 0);

  if (isNewScene) {
    // Strip pure instruction phrasing out of the scene text where possible.
    draft.prompt = message.trim();
  }
  for (const change of changes) change.apply(draft);

  if (!draft.prompt) {
    return {
      reply:
        "Tell me what the video should show — subject, setting and mood — and I'll shape it into a strong generation prompt. For example: “A futuristic car driving through neon Tokyo at night.”",
      draft,
    };
  }

  const changeSummary =
    changes.length > 0
      ? `I've ${changes.map((c) => c.description).join(", ")}. `
      : "";

  const reply = isNewScene
    ? `Nice concept! ${changeSummary}I've loaded it into the generation panel (${describeSettings(draft.settings)}). Tweak any settings you like, or tell me things like “make it vertical for Reels”, “use anime style” or “make it 15 seconds”. Hit **Generate video** when you're happy.`
    : changes.length > 0
      ? `Done — ${changes.map((c) => c.description).join(", ")}. Current setup: ${describeSettings(draft.settings)}. Anything else, or ready to hit **Generate video**?`
      : `I didn't detect a specific change there. You can adjust the prompt or settings directly in the panel, or try instructions like “make it more cinematic”, “change the location to Mumbai”, “make it vertical” or “add slow motion”. Current setup: ${describeSettings(draft.settings)}.`;

  return { reply, draft };
}
