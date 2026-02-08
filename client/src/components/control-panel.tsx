import { useRef, useEffect, useCallback, useState } from "react";
import { Play, Pause, Volume2, VolumeX } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import type { ManifoldData, VisualizationSettings } from "@shared/schema";

interface ControlPanelProps {
  data: ManifoldData;
  currentTime: number;
  isPlaying: boolean;
  settings: VisualizationSettings;
  onTimeUpdate: (time: number) => void;
  onSeek: (time: number) => void;
  onPlayPause: () => void;
  onSettingsChange: (settings: Partial<VisualizationSettings>) => void;
}

function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
}

export function ControlPanel({
  data,
  currentTime,
  isPlaying,
  settings,
  onTimeUpdate,
  onSeek,
  onPlayPause,
  onSettingsChange,
}: ControlPanelProps) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [isMuted, setIsMuted] = useState(false);
  const [volume, setVolume] = useState(0.7);
  const [showVolume, setShowVolume] = useState(false);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    if (isPlaying) {
      audio.play().catch(() => {});
    } else {
      audio.pause();
    }
  }, [isPlaying]);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    let animationId: number;

    const pollTime = () => {
      if (audio && !audio.paused) {
        onTimeUpdate(audio.currentTime);
      }
      animationId = requestAnimationFrame(pollTime);
    };

    animationId = requestAnimationFrame(pollTime);

    const handleEnded = () => {
      if (settings.loopPlayback) {
        audio.currentTime = 0;
        audio.play().catch(() => {});
      } else {
        onPlayPause();
      }
    };

    audio.addEventListener("ended", handleEnded);

    return () => {
      cancelAnimationFrame(animationId);
      audio.removeEventListener("ended", handleEnded);
    };
  }, [onTimeUpdate, onPlayPause, settings.loopPlayback]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.code === "Space" && e.target === document.body) {
        e.preventDefault();
        onPlayPause();
      }
      if (e.code === "ArrowLeft") {
        e.preventDefault();
        onSeek(Math.max(0, currentTime - 5));
      }
      if (e.code === "ArrowRight") {
        e.preventDefault();
        onSeek(Math.min(data.duration, currentTime + 5));
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onPlayPause, onSeek, currentTime, data.duration]);

  const handleSeek = useCallback((value: number[]) => {
    const time = value[0];
    if (audioRef.current) {
      audioRef.current.currentTime = time;
    }
    onSeek(time);
  }, [onSeek]);

  const handleVolumeChange = useCallback((value: number[]) => {
    const vol = value[0];
    setVolume(vol);
    setIsMuted(vol === 0);
    if (audioRef.current) {
      audioRef.current.volume = vol;
    }
  }, []);

  const toggleMute = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.muted = !isMuted;
      setIsMuted(!isMuted);
    }
  }, [isMuted]);

  const progress = (currentTime / data.duration) * 100;

  return (
    <div
      className="fixed bottom-0 left-0 right-0 z-20 h-20 backdrop-blur-2xl bg-black/40 border-t border-white/5"
      data-testid="panel-controls"
    >
      <audio ref={audioRef} src={data.audioUrl} preload="auto" />

      <div className="h-full max-w-screen-xl mx-auto px-6 flex items-center gap-6">
        {/* Play/Pause + Volume */}
        <div className="flex items-center gap-3">
          <Button
            size="icon"
            variant="outline"
            className="w-12 h-12 rounded-full border-white/20"
            onClick={onPlayPause}
            data-testid="button-play-pause"
          >
            {isPlaying ? (
              <Pause className="w-5 h-5" />
            ) : (
              <Play className="w-5 h-5 ml-0.5" />
            )}
          </Button>

          <div
            className="relative"
            onMouseEnter={() => setShowVolume(true)}
            onMouseLeave={() => setShowVolume(false)}
          >
            <Button
              size="icon"
              variant="ghost"
              onClick={toggleMute}
              data-testid="button-volume"
            >
              {isMuted ? (
                <VolumeX className="w-4 h-4" />
              ) : (
                <Volume2 className="w-4 h-4" />
              )}
            </Button>

            {showVolume && (
              <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 p-2 backdrop-blur-xl bg-black/60 rounded-lg">
                <Slider
                  orientation="vertical"
                  className="h-20"
                  value={[isMuted ? 0 : volume]}
                  min={0}
                  max={1}
                  step={0.01}
                  onValueChange={handleVolumeChange}
                  data-testid="slider-volume"
                />
              </div>
            )}
          </div>

          <span className="font-mono text-xs text-muted-foreground w-24" data-testid="text-time">
            {formatTime(currentTime)} / {formatTime(data.duration)}
          </span>
        </div>

        {/* Timeline */}
        <div className="flex-1 relative">
          <div className="relative">
            <div className="absolute inset-0 h-1 rounded-full bg-white/10" />
            <div
              className="absolute h-1 rounded-full bg-gradient-to-r from-cyan-400 via-purple-500 to-pink-500"
              style={{ width: `${progress}%` }}
            />
            <Slider
              className="absolute inset-0"
              value={[currentTime]}
              min={0}
              max={data.duration}
              step={0.01}
              onValueChange={handleSeek}
              data-testid="slider-timeline"
            />
          </div>
        </div>

        {/* Trail length control */}
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground whitespace-nowrap">
            Trail {settings.trailLength > 0 ? `${settings.trailLength}s` : "All"}
          </span>
          <Slider
            className="w-20"
            value={[settings.trailLength]}
            min={0}
            max={60}
            step={1}
            onValueChange={(v) => onSettingsChange({ trailLength: v[0] })}
            data-testid="slider-trail"
          />
        </div>

        {/* Point scale */}
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground whitespace-nowrap">Size</span>
          <Slider
            className="w-16"
            value={[settings.pointScale]}
            min={0.2}
            max={3}
            step={0.1}
            onValueChange={(v) => onSettingsChange({ pointScale: v[0] })}
            data-testid="slider-point-scale"
          />
        </div>
      </div>
    </div>
  );
}
