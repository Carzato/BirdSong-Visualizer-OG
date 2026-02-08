import { z } from "zod";

// Audio frame features extracted per analysis window
export const frameFeatureSchema = z.object({
  time: z.number(),
  mfcc: z.array(z.number()),        // 40 MFCCs
  chroma: z.array(z.number()),       // 12 pitch classes
  f0: z.number().nullable(),         // fundamental frequency Hz (null if unvoiced)
  f0Conf: z.number(),                // pitch confidence 0-1
  loudness: z.number(),              // RMS amplitude
  centroid: z.number(),              // spectral centroid Hz
  bandwidth: z.number(),             // spectral spread Hz
  flatness: z.number(),              // spectral flatness 0-1
  flux: z.number(),                  // spectral flux
});

// 3D embedded point for visualization
export const embeddedPointSchema = z.object({
  time: z.number(),
  position: z.tuple([z.number(), z.number(), z.number()]),
  color: z.tuple([z.number(), z.number(), z.number()]),   // RGB 0-1
  size: z.number(),
  opacity: z.number(),
  loudness: z.number(),
  f0: z.number().nullable(),
  f0Conf: z.number(),
  centroid: z.number(),
  chromaConcentration: z.number(),
});

// Full manifold visualization data
export const manifoldDataSchema = z.object({
  audioUrl: z.string(),
  duration: z.number(),
  sampleRate: z.number(),
  points: z.array(embeddedPointSchema),
});

// Upload response from server
export const audioUploadResponseSchema = z.object({
  success: z.boolean(),
  data: z.object({
    audioUrl: z.string(),
  }).optional(),
  error: z.string().optional(),
});

export const visualStyleSchema = z.enum(["manifold", "trajectory", "cloud"]);

export type FrameFeatures = z.infer<typeof frameFeatureSchema>;
export type EmbeddedPoint = z.infer<typeof embeddedPointSchema>;
export type ManifoldData = z.infer<typeof manifoldDataSchema>;
export type AudioUploadResponse = z.infer<typeof audioUploadResponseSchema>;
export type VisualStyle = z.infer<typeof visualStyleSchema>;

export interface VisualizationSettings {
  visualStyle: VisualStyle;
  autoRotate: boolean;
  loopPlayback: boolean;
  showDebug: boolean;
  isFullscreen: boolean;
  trailLength: number;      // seconds of visible trail (0 = show all)
  followCamera: boolean;    // camera follows trajectory head
  pointScale: number;       // point size multiplier
}
