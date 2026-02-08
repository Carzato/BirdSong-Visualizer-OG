import { useRef, useMemo, useEffect, useCallback } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, PerspectiveCamera } from "@react-three/drei";
import * as THREE from "three";
import type { ManifoldData, VisualizationSettings, EmbeddedPoint } from "@shared/schema";

// ─── Types ─────────────────────────────────────────────────────────────────

interface VisualizationCanvasProps {
  data: ManifoldData;
  currentTime: number;
  isPlaying: boolean;
  settings: VisualizationSettings;
}

interface ManifoldVisualizationProps {
  data: ManifoldData;
  currentTime: number;
  settings: VisualizationSettings;
}

// ─── Custom shaders ────────────────────────────────────────────────────────

const pointVertexShader = `
  attribute float pointSize;
  attribute float pointOpacity;
  attribute float age;
  varying vec3 vColor;
  varying float vOpacity;
  varying float vAge;

  void main() {
    vColor = color;
    vOpacity = pointOpacity;
    vAge = age;
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = pointSize * (350.0 / -mvPosition.z);
    gl_PointSize = max(gl_PointSize, 1.0);
    gl_Position = projectionMatrix * mvPosition;
  }
`;

const pointFragmentShader = `
  varying vec3 vColor;
  varying float vOpacity;
  varying float vAge;

  void main() {
    float dist = length(gl_PointCoord - vec2(0.5));
    if (dist > 0.5) discard;

    // Smooth circular falloff with inner glow
    float alpha = 1.0 - smoothstep(0.0, 0.5, dist);
    float glow = exp(-dist * 5.0) * 0.6;

    // Age-based fade: recent points are bright, old ones fade
    float ageFade = max(0.0, 1.0 - vAge);
    ageFade = ageFade * ageFade; // quadratic falloff for smoother fade

    vec3 finalColor = vColor * (1.0 + glow);
    float finalAlpha = alpha * vOpacity * ageFade;

    if (finalAlpha < 0.01) discard;
    gl_FragColor = vec4(finalColor, finalAlpha);
  }
`;

// ─── Trajectory line shaders ───────────────────────────────────────────────

const lineVertexShader = `
  attribute float lineOpacity;
  varying float vLineOpacity;
  varying vec3 vLineColor;

  void main() {
    vLineOpacity = lineOpacity;
    vLineColor = color;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const lineFragmentShader = `
  varying float vLineOpacity;
  varying vec3 vLineColor;

  void main() {
    if (vLineOpacity < 0.01) discard;
    gl_FragColor = vec4(vLineColor, vLineOpacity);
  }
`;

// ─── Manifold visualization component ──────────────────────────────────────

function ManifoldVisualization({ data, currentTime, settings }: ManifoldVisualizationProps) {
  const pointsRef = useRef<THREE.Points>(null);
  const linesRef = useRef<THREE.LineSegments>(null);
  const headGlowRef = useRef<THREE.Mesh>(null);

  const points = data.points;
  const N = points.length;

  // Precompute sorted time indices for binary search
  const sortedTimes = useMemo(() => points.map(p => p.time), [points]);

  // Find the index of the frame closest to currentTime
  const findCurrentIndex = useCallback((time: number): number => {
    if (N === 0) return -1;
    let lo = 0, hi = N - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (sortedTimes[mid] < time) lo = mid + 1;
      else hi = mid;
    }
    // Check neighbors for closest
    if (lo > 0 && Math.abs(sortedTimes[lo - 1] - time) < Math.abs(sortedTimes[lo] - time)) {
      return lo - 1;
    }
    return lo;
  }, [N, sortedTimes]);

  // Buffer geometry attributes
  const { positionArray, colorArray, sizeArray, opacityArray, ageArray } = useMemo(() => {
    const positionArray = new Float32Array(N * 3);
    const colorArray = new Float32Array(N * 3);
    const sizeArray = new Float32Array(N);
    const opacityArray = new Float32Array(N);
    const ageArray = new Float32Array(N);

    for (let i = 0; i < N; i++) {
      const pt = points[i];
      positionArray[i * 3] = pt.position[0];
      positionArray[i * 3 + 1] = pt.position[1];
      positionArray[i * 3 + 2] = pt.position[2];
      colorArray[i * 3] = pt.color[0];
      colorArray[i * 3 + 1] = pt.color[1];
      colorArray[i * 3 + 2] = pt.color[2];
      sizeArray[i] = pt.size * settings.pointScale;
      opacityArray[i] = pt.opacity;
      ageArray[i] = 1; // start fully aged (invisible)
    }

    return { positionArray, colorArray, sizeArray, opacityArray, ageArray };
  }, [N, points, settings.pointScale]);

  // Trajectory line geometry: connect consecutive points
  const { linePositions, lineColors, lineOpacities, lineCount } = useMemo(() => {
    if (N < 2) return { linePositions: new Float32Array(0), lineColors: new Float32Array(0), lineOpacities: new Float32Array(0), lineCount: 0 };

    const maxGap = 0.15; // max time gap to connect (seconds)
    const segments: { from: number; to: number }[] = [];

    for (let i = 1; i < N; i++) {
      const dt = points[i].time - points[i - 1].time;
      if (dt > 0 && dt < maxGap) {
        segments.push({ from: i - 1, to: i });
      }
    }

    const lineCount = segments.length;
    const linePositions = new Float32Array(lineCount * 6); // 2 vertices * 3 components
    const lineColors = new Float32Array(lineCount * 6);
    const lineOpacities = new Float32Array(lineCount * 2);

    for (let s = 0; s < lineCount; s++) {
      const { from, to } = segments[s];
      const pf = points[from];
      const pt = points[to];

      linePositions[s * 6] = pf.position[0];
      linePositions[s * 6 + 1] = pf.position[1];
      linePositions[s * 6 + 2] = pf.position[2];
      linePositions[s * 6 + 3] = pt.position[0];
      linePositions[s * 6 + 4] = pt.position[1];
      linePositions[s * 6 + 5] = pt.position[2];

      // Blend colors between endpoints
      const avgR = (pf.color[0] + pt.color[0]) * 0.5;
      const avgG = (pf.color[1] + pt.color[1]) * 0.5;
      const avgB = (pf.color[2] + pt.color[2]) * 0.5;
      lineColors[s * 6] = avgR;
      lineColors[s * 6 + 1] = avgG;
      lineColors[s * 6 + 2] = avgB;
      lineColors[s * 6 + 3] = avgR;
      lineColors[s * 6 + 4] = avgG;
      lineColors[s * 6 + 5] = avgB;

      lineOpacities[s * 2] = 0;
      lineOpacities[s * 2 + 1] = 0;
    }

    return { linePositions, lineColors, lineOpacities, lineCount };
  }, [N, points]);

  // Per-frame update: compute age, visibility, opacity
  useFrame(() => {
    if (!pointsRef.current || N === 0) return;

    const geo = pointsRef.current.geometry;
    const sizes = geo.attributes.pointSize.array as Float32Array;
    const opacities = geo.attributes.pointOpacity.array as Float32Array;
    const ages = geo.attributes.age.array as Float32Array;
    const colors = geo.attributes.color.array as Float32Array;

    const trailLen = settings.trailLength > 0 ? settings.trailLength : Infinity;
    const currentIdx = findCurrentIndex(currentTime);

    for (let i = 0; i < N; i++) {
      const pt = points[i];
      const timeDiff = currentTime - pt.time;

      if (timeDiff < 0) {
        // Future point: not yet reached
        ages[i] = 1;
        opacities[i] = 0;
        sizes[i] = 0;
      } else if (trailLen === Infinity || timeDiff <= trailLen) {
        // Visible point: compute age as fraction of trail
        const normalizedAge = trailLen === Infinity ? 0 : timeDiff / trailLen;
        ages[i] = normalizedAge;
        opacities[i] = pt.opacity;
        sizes[i] = pt.size * settings.pointScale;

        // Boost brightness of the "head" (current position)
        if (timeDiff < 0.1) {
          const boost = 1 + (1 - timeDiff / 0.1) * 1.5;
          colors[i * 3] = Math.min(1, pt.color[0] * boost);
          colors[i * 3 + 1] = Math.min(1, pt.color[1] * boost);
          colors[i * 3 + 2] = Math.min(1, pt.color[2] * boost);
          sizes[i] *= 1 + (1 - timeDiff / 0.1) * 0.8;
        } else {
          colors[i * 3] = pt.color[0];
          colors[i * 3 + 1] = pt.color[1];
          colors[i * 3 + 2] = pt.color[2];
        }
      } else {
        // Too old: faded out
        ages[i] = 1;
        opacities[i] = 0;
        sizes[i] = 0;
      }
    }

    geo.attributes.pointSize.needsUpdate = true;
    geo.attributes.pointOpacity.needsUpdate = true;
    geo.attributes.age.needsUpdate = true;
    geo.attributes.color.needsUpdate = true;

    // Update trajectory line opacities
    if (linesRef.current && lineCount > 0) {
      const lineGeo = linesRef.current.geometry;
      const lineOps = lineGeo.attributes.lineOpacity.array as Float32Array;
      let segIdx = 0;
      for (let i = 1; i < N && segIdx < lineCount; i++) {
        const dt = points[i].time - points[i - 1].time;
        if (dt <= 0 || dt >= 0.15) continue;

        const fromAge = currentTime - points[i - 1].time;
        const toAge = currentTime - points[i].time;
        const maxAge = Math.max(fromAge, toAge);
        const minAge = Math.min(fromAge, toAge);

        if (minAge < 0 || (trailLen !== Infinity && maxAge > trailLen)) {
          lineOps[segIdx * 2] = 0;
          lineOps[segIdx * 2 + 1] = 0;
        } else {
          const fadeFrom = trailLen === Infinity ? 0.4 : 0.4 * Math.max(0, 1 - fromAge / trailLen);
          const fadeTo = trailLen === Infinity ? 0.4 : 0.4 * Math.max(0, 1 - toAge / trailLen);
          lineOps[segIdx * 2] = fadeFrom;
          lineOps[segIdx * 2 + 1] = fadeTo;
        }
        segIdx++;
      }
      lineGeo.attributes.lineOpacity.needsUpdate = true;
    }

    // Update head glow position
    if (headGlowRef.current && currentIdx >= 0 && currentIdx < N) {
      const pt = points[currentIdx];
      headGlowRef.current.position.set(pt.position[0], pt.position[1], pt.position[2]);
      const loudScale = 0.3 + pt.loudness * 2;
      headGlowRef.current.scale.setScalar(loudScale);
    }
  });

  return (
    <group>
      {/* Point cloud */}
      <points ref={pointsRef}>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" count={N} array={positionArray} itemSize={3} />
          <bufferAttribute attach="attributes-color" count={N} array={colorArray} itemSize={3} />
          <bufferAttribute attach="attributes-pointSize" count={N} array={sizeArray} itemSize={1} />
          <bufferAttribute attach="attributes-pointOpacity" count={N} array={opacityArray} itemSize={1} />
          <bufferAttribute attach="attributes-age" count={N} array={ageArray} itemSize={1} />
        </bufferGeometry>
        <shaderMaterial
          vertexColors
          transparent
          depthWrite={false}
          blending={THREE.AdditiveBlending}
          vertexShader={pointVertexShader}
          fragmentShader={pointFragmentShader}
        />
      </points>

      {/* Trajectory lines */}
      {lineCount > 0 && (
        <lineSegments ref={linesRef}>
          <bufferGeometry>
            <bufferAttribute attach="attributes-position" count={lineCount * 2} array={linePositions} itemSize={3} />
            <bufferAttribute attach="attributes-color" count={lineCount * 2} array={lineColors} itemSize={3} />
            <bufferAttribute attach="attributes-lineOpacity" count={lineCount * 2} array={lineOpacities} itemSize={1} />
          </bufferGeometry>
          <shaderMaterial
            vertexColors
            transparent
            depthWrite={false}
            blending={THREE.AdditiveBlending}
            vertexShader={lineVertexShader}
            fragmentShader={lineFragmentShader}
          />
        </lineSegments>
      )}

      {/* Glowing head marker at current position */}
      <mesh ref={headGlowRef}>
        <sphereGeometry args={[0.15, 16, 16]} />
        <meshBasicMaterial
          color="#ffffff"
          transparent
          opacity={0.6}
          blending={THREE.AdditiveBlending}
        />
      </mesh>
    </group>
  );
}

// ─── Camera follow ─────────────────────────────────────────────────────────

function CameraController({ data, currentTime, settings }: {
  data: ManifoldData;
  currentTime: number;
  settings: VisualizationSettings;
}) {
  const { camera } = useThree();
  const targetRef = useRef(new THREE.Vector3());
  const smoothTargetRef = useRef(new THREE.Vector3());

  useFrame((_, delta) => {
    // Auto-rotate
    if (settings.autoRotate && !settings.followCamera) {
      camera.position.applyAxisAngle(new THREE.Vector3(0, 1, 0), delta * 0.08);
      camera.lookAt(smoothTargetRef.current);
    }

    // Follow camera: smoothly track the trajectory head
    if (settings.followCamera && data.points.length > 0) {
      // Find current point
      let closest = data.points[0];
      let minDiff = Math.abs(closest.time - currentTime);
      for (const pt of data.points) {
        const diff = Math.abs(pt.time - currentTime);
        if (diff < minDiff) {
          minDiff = diff;
          closest = pt;
        }
      }

      targetRef.current.set(closest.position[0], closest.position[1], closest.position[2]);
      smoothTargetRef.current.lerp(targetRef.current, delta * 2);

      // Camera orbits around the target
      const offset = camera.position.clone().sub(smoothTargetRef.current);
      offset.applyAxisAngle(new THREE.Vector3(0, 1, 0), delta * 0.08);
      const desiredPos = smoothTargetRef.current.clone().add(offset.normalize().multiplyScalar(8));
      camera.position.lerp(desiredPos, delta * 1.5);
      camera.lookAt(smoothTargetRef.current);
    }
  });

  return null;
}

// ─── Main canvas export ────────────────────────────────────────────────────

export function VisualizationCanvas({
  data,
  currentTime,
  settings,
}: VisualizationCanvasProps) {
  return (
    <div className="absolute inset-0 z-0" data-testid="canvas-visualization">
      <Canvas
        gl={{
          antialias: true,
          alpha: true,
          powerPreference: "high-performance",
        }}
        dpr={[1, 2]}
      >
        <color attach="background" args={["#050510"]} />
        <fog attach="fog" args={["#050510", 15, 40]} />

        <PerspectiveCamera makeDefault position={[0, 3, 14]} fov={55} />
        <OrbitControls
          enableDamping
          dampingFactor={0.05}
          minDistance={3}
          maxDistance={35}
          enablePan={true}
          autoRotate={false}
        />

        <CameraController data={data} currentTime={currentTime} settings={settings} />

        <ambientLight intensity={0.05} />

        <ManifoldVisualization
          data={data}
          currentTime={currentTime}
          settings={settings}
        />

        {/* Subtle reference grid */}
        <gridHelper args={[20, 40, "#111133", "#0a0a22"]} position={[0, -6, 0]} />
      </Canvas>
    </div>
  );
}
