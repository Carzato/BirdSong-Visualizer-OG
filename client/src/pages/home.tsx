import { useState, useCallback } from "react";
import { UploadInterface } from "@/components/upload-interface";
import { VisualizationCanvas } from "@/components/visualization-canvas";
import { ControlPanel } from "@/components/control-panel";
import { FloatingSettings } from "@/components/floating-settings";
import { DebugPanel } from "@/components/debug-panel";
import { LoadingOverlay } from "@/components/loading-overlay";
import { decodeAudioFromUrl, extractFeatures } from "@/lib/audio-features";
import { computePCA } from "@/lib/pca";
import { mapToVisualPoints } from "@/lib/visual-mapper";
import type { ManifoldData, VisualizationSettings } from "@shared/schema";

export default function Home() {
  const [manifoldData, setManifoldData] = useState<ManifoldData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState("Uploading audio...");
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [settings, setSettings] = useState<VisualizationSettings>({
    visualStyle: "manifold",
    autoRotate: true,
    loopPlayback: true,
    showDebug: false,
    isFullscreen: false,
    trailLength: 15,     // 15 seconds of visible trail
    followCamera: false,
    pointScale: 1.0,
  });

  const handleUploadStart = useCallback(() => {
    setIsLoading(true);
    setLoadingMessage("Uploading audio file...");
  }, []);

  const handleAnalyzing = useCallback(() => {
    setLoadingMessage("Uploading to server...");
  }, []);

  const handleUploadComplete = useCallback(async (audioUrl: string) => {
    try {
      // Phase 1: Decode audio
      setLoadingMessage("Decoding audio...");
      const audioBuffer = await decodeAudioFromUrl(audioUrl);
      const duration = audioBuffer.duration;
      const sampleRate = audioBuffer.sampleRate;

      // Phase 2: Extract features (MFCCs, chroma, pitch, spectral)
      setLoadingMessage("Extracting audio features (MFCCs, chroma, pitch)...");
      const frames = await extractFeatures(audioBuffer, {
        frameSize: 2048,
        hopSize: 512,
        numMFCC: 40,
        numMelBands: 80,
        onProgress: (p) => {
          setLoadingMessage(
            `Extracting features... ${Math.round(p * 100)}%`
          );
        },
      });

      // Phase 3: PCA dimensionality reduction to 3D
      setLoadingMessage("Computing acoustic manifold (PCA → 3D)...");
      await new Promise(r => setTimeout(r, 10)); // yield for UI
      const pcaResult = computePCA(frames);

      // Phase 4: Map to visual attributes (color, size, opacity)
      setLoadingMessage("Mapping visual attributes...");
      await new Promise(r => setTimeout(r, 10));
      const points = mapToVisualPoints(frames, pcaResult.positions);

      // Downsample if too many points for smooth rendering
      const maxPoints = 8000;
      let finalPoints = points;
      if (points.length > maxPoints) {
        const step = Math.ceil(points.length / maxPoints);
        finalPoints = points.filter((_, i) => i % step === 0);
      }

      const data: ManifoldData = {
        audioUrl,
        duration,
        sampleRate,
        points: finalPoints,
      };

      setManifoldData(data);
      setIsLoading(false);
      setCurrentTime(0);
      setIsPlaying(false);
    } catch (err) {
      console.error("Client-side analysis failed:", err);
      setIsLoading(false);
    }
  }, []);

  const handleUploadError = useCallback(() => {
    setIsLoading(false);
  }, []);

  const handleTimeUpdate = useCallback((time: number) => {
    setCurrentTime(time);
  }, []);

  const handleSeek = useCallback((time: number) => {
    setCurrentTime(time);
  }, []);

  const handlePlayPause = useCallback(() => {
    setIsPlaying(prev => !prev);
  }, []);

  const handleSettingsChange = useCallback((newSettings: Partial<VisualizationSettings>) => {
    setSettings(prev => ({ ...prev, ...newSettings }));
  }, []);

  const handleReset = useCallback(() => {
    setManifoldData(null);
    setCurrentTime(0);
    setIsPlaying(false);
  }, []);

  const toggleFullscreen = useCallback(() => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen();
      handleSettingsChange({ isFullscreen: true });
    } else {
      document.exitFullscreen();
      handleSettingsChange({ isFullscreen: false });
    }
  }, [handleSettingsChange]);

  if (!manifoldData) {
    return (
      <div className="relative min-h-screen bg-gradient-radial from-background via-background to-black flex items-center justify-center">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-gray-900/20 via-background to-black" />
        <UploadInterface
          onUploadStart={handleUploadStart}
          onAnalyzing={handleAnalyzing}
          onUploadComplete={handleUploadComplete}
          onUploadError={handleUploadError}
        />
        {isLoading && <LoadingOverlay message={loadingMessage} />}
      </div>
    );
  }

  return (
    <div className="relative w-full h-screen overflow-hidden bg-black">
      <VisualizationCanvas
        data={manifoldData}
        currentTime={currentTime}
        isPlaying={isPlaying}
        settings={settings}
      />

      <FloatingSettings
        settings={settings}
        onSettingsChange={handleSettingsChange}
        onToggleFullscreen={toggleFullscreen}
        onReset={handleReset}
      />

      {settings.showDebug && (
        <DebugPanel data={manifoldData} currentTime={currentTime} />
      )}

      <ControlPanel
        data={manifoldData}
        currentTime={currentTime}
        isPlaying={isPlaying}
        settings={settings}
        onTimeUpdate={handleTimeUpdate}
        onSeek={handleSeek}
        onPlayPause={handlePlayPause}
        onSettingsChange={handleSettingsChange}
      />

      {isLoading && <LoadingOverlay message={loadingMessage} />}
    </div>
  );
}
