/**
 * Maps audio frame features and PCA positions to visual attributes:
 * color (HSV→RGB), size, and opacity.
 */

import type { FrameFeatures, EmbeddedPoint } from "@shared/schema";

/**
 * Convert HSV to RGB. All values in [0,1].
 */
function hsvToRgb(h: number, s: number, v: number): [number, number, number] {
  h = ((h % 1) + 1) % 1; // wrap hue
  const i = Math.floor(h * 6);
  const f = h * 6 - i;
  const p = v * (1 - s);
  const q = v * (1 - f * s);
  const t = v * (1 - (1 - f) * s);
  switch (i % 6) {
    case 0: return [v, t, p];
    case 1: return [q, v, p];
    case 2: return [p, v, t];
    case 3: return [p, q, v];
    case 4: return [t, p, v];
    case 5: return [v, p, q];
    default: return [v, t, p];
  }
}

/**
 * Compute chroma concentration: how much energy is focused on one pitch class.
 * Higher value = more tonal/chordal, lower = noisy/atonal.
 */
function chromaConcentration(chroma: number[]): number {
  if (chroma.length === 0) return 0;
  return Math.max(...chroma);
}

/**
 * Normalize a value within observed min/max range.
 */
function norm(value: number, min: number, max: number): number {
  if (max <= min) return 0.5;
  return Math.max(0, Math.min(1, (value - min) / (max - min)));
}

/**
 * Map an array of FrameFeatures + PCA positions to EmbeddedPoints.
 */
export function mapToVisualPoints(
  frames: FrameFeatures[],
  positions: [number, number, number][]
): EmbeddedPoint[] {
  if (frames.length === 0) return [];

  // Compute normalization ranges across all frames
  const loudnesses = frames.map(f => f.loudness);
  const centroids = frames.map(f => f.centroid);
  const f0s = frames.filter(f => f.f0 !== null).map(f => Math.log2(f.f0!));

  const loudMin = Math.min(...loudnesses);
  const loudMax = Math.max(...loudnesses);
  const centMin = Math.min(...centroids);
  const centMax = Math.max(...centroids);
  const f0Min = f0s.length > 0 ? Math.min(...f0s) : 0;
  const f0Max = f0s.length > 0 ? Math.max(...f0s) : 1;

  return frames.map((frame, i) => {
    // ─── Color (HSV → RGB) ───────────────────────────────────
    // Hue: pitch-based (log f0 mapped to hue circle) or centroid fallback
    let hue: number;
    if (frame.f0 !== null && frame.f0Conf > 0.3) {
      hue = norm(Math.log2(frame.f0), f0Min, f0Max);
    } else {
      hue = norm(frame.centroid, centMin, centMax);
    }

    // Saturation: chroma concentration (tonal → saturated, noisy → desaturated)
    const chromaConc = chromaConcentration(frame.chroma);
    const saturation = 0.3 + chromaConc * 0.7;

    // Value/brightness: loudness
    const loudNorm = norm(frame.loudness, loudMin, loudMax);
    const brightness = 0.2 + loudNorm * 0.8;

    const color = hsvToRgb(hue, saturation, brightness);

    // ─── Size ────────────────────────────────────────────────
    // Emphasize clear musical events (loud + pitched)
    const baseSize = 0.3;
    const confFactor = 0.3 + 0.7 * frame.f0Conf;
    const loudFactor = 0.3 + 0.7 * loudNorm;
    const size = baseSize * loudFactor * confFactor;

    // ─── Opacity ─────────────────────────────────────────────
    // Lower for quiet/unvoiced segments
    const opacity = 0.2 + 0.8 * loudNorm * (0.3 + 0.7 * frame.f0Conf);

    return {
      time: frame.time,
      position: positions[i],
      color,
      size,
      opacity,
      loudness: frame.loudness,
      f0: frame.f0,
      f0Conf: frame.f0Conf,
      centroid: frame.centroid,
      chromaConcentration: chromaConc,
    };
  });
}
