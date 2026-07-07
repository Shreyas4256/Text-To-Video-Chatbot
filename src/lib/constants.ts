/** Shared, client-safe constants (no secrets here). */

export const ASPECT_RATIOS = ["16:9", "9:16", "1:1"] as const;
export type AspectRatio = (typeof ASPECT_RATIOS)[number];

export const DURATIONS = [4, 5, 6, 8, 10, 15] as const;

export const QUALITIES = ["draft", "standard", "high"] as const;
export type Quality = (typeof QUALITIES)[number];

export interface StylePreset {
  id: string;
  label: string;
  /** Fragment appended to the prompt during rule-based enhancement. */
  promptFragment: string;
}

export const STYLE_PRESETS: StylePreset[] = [
  { id: "cinematic", label: "Cinematic", promptFragment: "cinematic lighting, film grain, shallow depth of field, anamorphic lens" },
  { id: "realistic", label: "Realistic", promptFragment: "photorealistic, natural lighting, high detail, true-to-life colors" },
  { id: "anime", label: "Anime", promptFragment: "anime style, vibrant cel shading, expressive linework, studio-quality animation" },
  { id: "3d-animation", label: "3D Animation", promptFragment: "3D animated, soft global illumination, subsurface scattering, polished render" },
  { id: "product-ad", label: "Product Advertisement", promptFragment: "premium product shot, studio lighting, clean background, macro detail" },
  { id: "documentary", label: "Documentary", promptFragment: "documentary style, handheld realism, natural color grade, observational framing" },
  { id: "cyberpunk", label: "Cyberpunk", promptFragment: "cyberpunk aesthetic, neon glow, rain-slick streets, holographic signage, high contrast" },
  { id: "cartoon", label: "Cartoon", promptFragment: "cartoon style, bold outlines, flat vivid colors, playful exaggerated motion" },
  { id: "luxury", label: "Luxury Commercial", promptFragment: "luxury commercial look, golden hour glow, elegant slow reveals, rich textures" },
  { id: "social-reel", label: "Social Media Reel", promptFragment: "punchy social-media look, crisp colors, energetic pacing, eye-catching composition" },
];

export interface CameraPreset {
  id: string;
  label: string;
  promptFragment: string;
}

export const CAMERA_MOVEMENTS: CameraPreset[] = [
  { id: "static", label: "Static", promptFragment: "static camera" },
  { id: "pan", label: "Pan", promptFragment: "smooth panning shot" },
  { id: "dolly-in", label: "Dolly In", promptFragment: "slow dolly-in" },
  { id: "dolly-out", label: "Dolly Out", promptFragment: "slow dolly-out" },
  { id: "tracking", label: "Tracking", promptFragment: "dynamic tracking shot" },
  { id: "crane", label: "Crane", promptFragment: "sweeping crane shot" },
  { id: "orbit", label: "Orbit", promptFragment: "orbiting camera movement" },
  { id: "handheld", label: "Handheld", promptFragment: "handheld camera feel" },
  { id: "fpv", label: "FPV Drone", promptFragment: "fast FPV drone flythrough" },
];

export interface DraftSettings {
  aspectRatio: AspectRatio;
  durationSec: number;
  style: string;
  cameraMovement: string;
  motionStrength: number;
  quality: Quality;
  seed?: number;
  negativePrompt?: string;
}

export const DEFAULT_SETTINGS: DraftSettings = {
  aspectRatio: "16:9",
  durationSec: 5,
  style: "cinematic",
  cameraMovement: "static",
  motionStrength: 5,
  quality: "standard",
};

export const STARTER_PROMPTS = [
  "A cinematic 10-second video of a futuristic car driving through neon Tokyo at night",
  "A golden retriever puppy chasing butterflies in a sunlit meadow, slow motion",
  "Aerial drone shot flying over turquoise ocean waves crashing on a tropical beach",
  "A steaming cup of coffee on a rainy window sill, cozy morning light, close-up",
  "An astronaut floating through a colorful nebula, stars sparkling all around",
  "Time-lapse of a bustling city intersection at dusk as lights flicker on",
];

export const GENERATION_STATUSES = ["QUEUED", "PROCESSING", "COMPLETED", "FAILED"] as const;
export type GenerationStatusValue = (typeof GENERATION_STATUSES)[number];

/** Placeholder pricing until a real billing system is connected. */
export function estimateCredits(durationSec: number, quality: Quality): number {
  const base = Math.ceil(durationSec / 5);
  const multiplier = quality === "high" ? 2 : quality === "draft" ? 0.5 : 1;
  return Math.max(1, Math.round(base * multiplier));
}
