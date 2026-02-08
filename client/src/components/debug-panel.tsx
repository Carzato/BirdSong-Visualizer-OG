import { useMemo } from "react";
import type { ManifoldData, EmbeddedPoint } from "@shared/schema";

interface DebugPanelProps {
  data: ManifoldData;
  currentTime: number;
}

export function DebugPanel({ data, currentTime }: DebugPanelProps) {
  const currentPoint = useMemo(() => {
    if (!data.points.length) return null;

    let closest = data.points[0];
    let minDiff = Math.abs(closest.time - currentTime);

    for (const point of data.points) {
      const diff = Math.abs(point.time - currentTime);
      if (diff < minDiff) {
        minDiff = diff;
        closest = point;
      }
    }

    return closest;
  }, [data.points, currentTime]);

  const visiblePointCount = useMemo(() => {
    return data.points.filter((p: EmbeddedPoint) => p.time <= currentTime).length;
  }, [data.points, currentTime]);

  const stats = useMemo(() => {
    const pts = data.points;
    const avgLoudness = pts.reduce((sum, p) => sum + p.loudness, 0) / pts.length;
    const avgChroma = pts.reduce((sum, p) => sum + p.chromaConcentration, 0) / pts.length;
    const pitchedCount = pts.filter(p => p.f0 !== null).length;
    return { avgLoudness, avgChroma, pitchedCount };
  }, [data.points]);

  return (
    <div
      className="fixed top-20 right-4 z-10 w-72 p-4 backdrop-blur-xl bg-black/70 rounded-xl border border-white/10"
      data-testid="panel-debug"
    >
      <h3 className="text-xs uppercase tracking-wide text-muted-foreground mb-4 font-medium border-b border-white/10 pb-2">
        Acoustic Manifold Debug
      </h3>

      <div className="space-y-4">
        <div className="space-y-2">
          <h4 className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold">
            Playback
          </h4>
          <div className="grid grid-cols-2 gap-2">
            <div className="flex flex-col">
              <span className="text-[10px] uppercase tracking-wider text-muted-foreground">Time</span>
              <span className="font-mono text-sm text-primary">{currentTime.toFixed(2)}s</span>
            </div>
            <div className="flex flex-col">
              <span className="text-[10px] uppercase tracking-wider text-muted-foreground">Duration</span>
              <span className="font-mono text-sm text-foreground">{data.duration.toFixed(2)}s</span>
            </div>
            <div className="flex flex-col">
              <span className="text-[10px] uppercase tracking-wider text-muted-foreground">Progress</span>
              <span className="font-mono text-sm text-foreground">{((currentTime / data.duration) * 100).toFixed(1)}%</span>
            </div>
            <div className="flex flex-col">
              <span className="text-[10px] uppercase tracking-wider text-muted-foreground">Sample Rate</span>
              <span className="font-mono text-sm text-foreground">{data.sampleRate} Hz</span>
            </div>
          </div>
        </div>

        <div className="space-y-2">
          <h4 className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold">
            Manifold
          </h4>
          <div className="grid grid-cols-2 gap-2">
            <div className="flex flex-col">
              <span className="text-[10px] uppercase tracking-wider text-muted-foreground">Total Points</span>
              <span className="font-mono text-sm text-foreground">{data.points.length}</span>
            </div>
            <div className="flex flex-col">
              <span className="text-[10px] uppercase tracking-wider text-muted-foreground">Visible</span>
              <span className="font-mono text-sm text-foreground">{visiblePointCount}</span>
            </div>
            <div className="flex flex-col">
              <span className="text-[10px] uppercase tracking-wider text-muted-foreground">Avg Loudness</span>
              <span className="font-mono text-sm text-foreground">{stats.avgLoudness.toFixed(4)}</span>
            </div>
            <div className="flex flex-col">
              <span className="text-[10px] uppercase tracking-wider text-muted-foreground">Avg Chroma</span>
              <span className="font-mono text-sm text-foreground">{stats.avgChroma.toFixed(3)}</span>
            </div>
            <div className="flex flex-col">
              <span className="text-[10px] uppercase tracking-wider text-muted-foreground">Pitched Frames</span>
              <span className="font-mono text-sm text-foreground">{stats.pitchedCount}/{data.points.length}</span>
            </div>
          </div>
        </div>

        {currentPoint && (
          <div className="space-y-2">
            <h4 className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold">
              Current Point (t={currentPoint.time.toFixed(3)}s)
            </h4>
            <div className="grid grid-cols-2 gap-2">
              <div className="flex flex-col">
                <span className="text-[10px] uppercase tracking-wider text-muted-foreground">Position</span>
                <span className="font-mono text-sm text-foreground">
                  {currentPoint.position[0].toFixed(2)}, {currentPoint.position[1].toFixed(2)}, {currentPoint.position[2].toFixed(2)}
                </span>
              </div>
              <div className="flex flex-col">
                <span className="text-[10px] uppercase tracking-wider text-muted-foreground">f0</span>
                <span className="font-mono text-sm text-foreground">
                  {currentPoint.f0 ? `${currentPoint.f0.toFixed(1)} Hz` : "—"}
                </span>
              </div>
              <div className="flex flex-col">
                <span className="text-[10px] uppercase tracking-wider text-muted-foreground">Confidence</span>
                <span className="font-mono text-sm text-foreground">{currentPoint.f0Conf.toFixed(3)}</span>
              </div>
              <div className="flex flex-col">
                <span className="text-[10px] uppercase tracking-wider text-muted-foreground">Centroid</span>
                <span className="font-mono text-sm text-foreground">{currentPoint.centroid.toFixed(0)} Hz</span>
              </div>
              <div className="flex flex-col">
                <span className="text-[10px] uppercase tracking-wider text-muted-foreground">Loudness</span>
                <span className="font-mono text-sm text-foreground">{currentPoint.loudness.toFixed(4)}</span>
              </div>
              <div className="flex flex-col">
                <span className="text-[10px] uppercase tracking-wider text-muted-foreground">Chroma Conc.</span>
                <span className="font-mono text-sm text-foreground">{currentPoint.chromaConcentration.toFixed(3)}</span>
              </div>
              <div className="flex flex-col">
                <span className="text-[10px] uppercase tracking-wider text-muted-foreground">Size</span>
                <span className="font-mono text-sm text-foreground">{currentPoint.size.toFixed(3)}</span>
              </div>
              <div className="flex flex-col">
                <span className="text-[10px] uppercase tracking-wider text-muted-foreground">Opacity</span>
                <span className="font-mono text-sm text-foreground">{currentPoint.opacity.toFixed(3)}</span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
