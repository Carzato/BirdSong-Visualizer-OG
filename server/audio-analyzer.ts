import fs from "fs";
import path from "path";
import ffmpeg from "fluent-ffmpeg";
import { randomUUID } from "crypto";
import os from "os";
import { Matrix, EigenvalueDecomposition } from "ml-matrix";

interface AudioFrame {
  t: number;
  mfccs: number[];
  pcaCoordinates?: {
    x: number;
    y: number;
    z: number;
  };
  frequency?: number;
  amplitude?: number;
}

interface VisualizationPoint {
  x: number;
  y: number;
  z: number;
  size: number;
  color: [number, number, number];
  time: number;
  mfccs: number[];
  frequency?: number;
  beatStrength?: number;
  complexity?: number;
  band?: 'low' | 'mid' | 'high';
  bandOffsetZ?: number;
  onsetLow?: number;
  onsetMid?: number;
  onsetHigh?: number;
}

interface Verse {
  id: number;
  name: string;
  start: number;
  end: number;
  points: VisualizationPoint[];
  edges: [number, number][];
}

interface AnalysisResult {
  duration: number;
  sampleRate: number;
  verses: Verse[];
}

function parseWavHeader(buffer: Buffer): { sampleRate: number; channels: number; bitsPerSample: number; dataOffset: number; dataLength: number } | null {
  if (buffer.length < 44) return null;
  
  const riff = buffer.toString("ascii", 0, 4);
  const wave = buffer.toString("ascii", 8, 12);
  
  if (riff !== "RIFF" || wave !== "WAVE") return null;
  
  let offset = 12;
  let sampleRate = 44100;
  let channels = 2;
  let bitsPerSample = 16;
  let dataOffset = 44;
  let dataLength = 0;
  
  while (offset < buffer.length - 8) {
    const chunkId = buffer.toString("ascii", offset, offset + 4);
    const chunkSize = buffer.readUInt32LE(offset + 4);
    
    if (chunkId === "fmt ") {
      channels = buffer.readUInt16LE(offset + 10);
      sampleRate = buffer.readUInt32LE(offset + 12);
      bitsPerSample = buffer.readUInt16LE(offset + 22);
    } else if (chunkId === "data") {
      dataOffset = offset + 8;
      dataLength = chunkSize;
      break;
    }
    
    offset += 8 + chunkSize;
  }
  
  return { sampleRate, channels, bitsPerSample, dataOffset, dataLength };
}

function extractPcmSamples(buffer: Buffer, header: { sampleRate: number; channels: number; bitsPerSample: number; dataOffset: number; dataLength: number }): Float32Array {
  const { dataOffset, dataLength, bitsPerSample, channels } = header;
  const bytesPerSample = bitsPerSample / 8;
  const numSamples = Math.floor(dataLength / (bytesPerSample * channels));
  const samples = new Float32Array(numSamples);
  
  for (let i = 0; i < numSamples; i++) {
    const offset = dataOffset + i * bytesPerSample * channels;
    if (offset + bytesPerSample > buffer.length) break;
    
    let sample: number;
    if (bitsPerSample === 16) {
      sample = buffer.readInt16LE(offset) / 32768;
    } else if (bitsPerSample === 8) {
      sample = (buffer.readUInt8(offset) - 128) / 128;
    } else if (bitsPerSample === 32) {
      sample = buffer.readInt32LE(offset) / 2147483648;
    } else {
      sample = buffer.readInt16LE(offset) / 32768;
    }
    
    samples[i] = sample;
  }
  
  return samples;
}

function computeRMS(samples: Float32Array, start: number, length: number): number {
  let sum = 0;
  const end = Math.min(start + length, samples.length);
  for (let i = start; i < end; i++) {
    sum += samples[i] * samples[i];
  }
  return Math.sqrt(sum / (end - start));
}

function computeZeroCrossingRate(samples: Float32Array, start: number, length: number): number {
  let crossings = 0;
  const end = Math.min(start + length, samples.length);
  for (let i = start + 1; i < end; i++) {
    if ((samples[i] >= 0 && samples[i - 1] < 0) || (samples[i] < 0 && samples[i - 1] >= 0)) {
      crossings++;
    }
  }
  return crossings / (end - start);
}

function estimatePitchFromZCR(zcr: number, sampleRate: number): number {
  const estimatedFreq = (zcr * sampleRate) / 2;
  return Math.max(80, Math.min(8000, estimatedFreq));
}

// Precomputed lookup tables (lazily initialized per frame size)
let _cachedFrameSize = 0;
let _hannWindow: Float32Array;
let _fftReal: Float32Array;
let _fftImag: Float32Array;
let _fftMagnitudes: Float32Array;

function ensureFFTBuffers(frameSize: number): void {
  if (_cachedFrameSize === frameSize) return;
  _cachedFrameSize = frameSize;
  _hannWindow = new Float32Array(frameSize);
  for (let i = 0; i < frameSize; i++) {
    _hannWindow[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (frameSize - 1)));
  }
  _fftReal = new Float32Array(frameSize);
  _fftImag = new Float32Array(frameSize);
  _fftMagnitudes = new Float32Array(frameSize / 2);
}

function computeFFT(samples: Float32Array, start: number, frameSize: number): Float32Array {
  ensureFFTBuffers(frameSize);
  const n = frameSize;
  const real = _fftReal;
  const imag = _fftImag;

  // Copy windowed input using precomputed Hann window
  for (let i = 0; i < n; i++) {
    const idx = start + i;
    real[i] = (idx < samples.length ? samples[idx] : 0) * _hannWindow[i];
    imag[i] = 0;
  }

  // Bit-reversal permutation
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    while (j & bit) {
      j ^= bit;
      bit >>= 1;
    }
    j ^= bit;
    if (i < j) {
      let tmp = real[i]; real[i] = real[j]; real[j] = tmp;
      tmp = imag[i]; imag[i] = imag[j]; imag[j] = tmp;
    }
  }

  // Butterfly stages
  for (let len = 2; len <= n; len *= 2) {
    const halfLen = len / 2;
    const angle = (-2 * Math.PI) / len;
    const wReal = Math.cos(angle);
    const wImag = Math.sin(angle);
    for (let i = 0; i < n; i += len) {
      let curReal = 1;
      let curImag = 0;
      for (let j = 0; j < halfLen; j++) {
        const idx = i + j + halfLen;
        const tReal = curReal * real[idx] - curImag * imag[idx];
        const tImag = curReal * imag[idx] + curImag * real[idx];
        real[idx] = real[i + j] - tReal;
        imag[idx] = imag[i + j] - tImag;
        real[i + j] += tReal;
        imag[i + j] += tImag;
        const newCurReal = curReal * wReal - curImag * wImag;
        curImag = curReal * wImag + curImag * wReal;
        curReal = newCurReal;
      }
    }
  }

  // Compute magnitude spectrum (first half only) — reuse buffer
  const magnitudes = _fftMagnitudes;
  for (let k = 0; k < n / 2; k++) {
    magnitudes[k] = Math.sqrt(real[k] * real[k] + imag[k] * imag[k]);
  }
  // Return a COPY since the caller may store the result
  return Float32Array.from(magnitudes);
}

function computeSpectralFeatures(fft: Float32Array, frameSize: number, sampleRate: number): { centroid: number; bandwidth: number } {
  const freqBinWidth = sampleRate / frameSize;
  
  let sumMag = 0;
  let sumFreqMag = 0;
  for (let k = 1; k < fft.length; k++) {
    const freq = k * freqBinWidth;
    sumMag += fft[k];
    sumFreqMag += freq * fft[k];
  }
  
  const centroid = sumMag > 0 ? sumFreqMag / sumMag : 1000;
  
  let sumBandwidth = 0;
  for (let k = 1; k < fft.length; k++) {
    const freq = k * freqBinWidth;
    sumBandwidth += fft[k] * Math.pow(freq - centroid, 2);
  }
  const bandwidth = sumMag > 0 ? Math.sqrt(sumBandwidth / sumMag) : 500;
  
  return { centroid, bandwidth };
}

function computeSpectralFlux(currentFFT: Float32Array, prevFFT: Float32Array | null): number {
  if (!prevFFT) return 0;
  
  let flux = 0;
  for (let k = 0; k < currentFFT.length; k++) {
    const diff = currentFFT[k] - prevFFT[k];
    if (diff > 0) {
      flux += diff * diff;
    }
  }
  return Math.sqrt(flux);
}

function computeSpectralFlatness(fft: Float32Array): number {
  const epsilon = 1e-10;
  let logSum = 0;
  let sum = 0;
  let count = 0;
  
  for (let k = 1; k < fft.length; k++) {
    const mag = Math.max(fft[k], epsilon);
    logSum += Math.log(mag);
    sum += mag;
    count++;
  }
  
  if (count === 0 || sum === 0) return 0;
  
  const geometricMean = Math.exp(logSum / count);
  const arithmeticMean = sum / count;
  
  return geometricMean / arithmeticMean;
}

function computeOnsetStrength(prevRMS: number, currentRMS: number): number {
  const diff = currentRMS - prevRMS;
  return diff > 0 ? Math.min(1, diff * 10) : 0;
}

function computeBandEnergies(
  fft: Float32Array,
  fftSize: number,
  sampleRate: number
): { low: number; mid: number; high: number } {
  const binFreq = (bin: number) => bin * sampleRate / fftSize;

  let low = 0, mid = 0, high = 0;

  for (let bin = 0; bin < fft.length; bin++) {
    const freq = binFreq(bin);
    const mag = fft[bin];

    if (freq >= 20 && freq < 250) {
      low += mag;
    } else if (freq >= 250 && freq < 2000) {
      mid += mag;
    } else if (freq >= 2000 && freq < 8000) {
      high += mag;
    }
  }

  return { low, mid, high };
}

// Mel scale conversion functions
function hzToMel(hz: number): number {
  return 2595 * Math.log10(1 + hz / 700);
}

function melToHz(mel: number): number {
  return 700 * (Math.pow(10, mel / 2595) - 1);
}

// Create Mel filterbank
function createMelFilterbank(
  numFilters: number,
  fftSize: number,
  sampleRate: number,
  lowFreq: number = 0,
  highFreq?: number
): number[][] {
  highFreq = highFreq || sampleRate / 2;

  const lowMel = hzToMel(lowFreq);
  const highMel = hzToMel(highFreq);
  const melPoints = Array.from(
    { length: numFilters + 2 },
    (_, i) => lowMel + (i * (highMel - lowMel)) / (numFilters + 1)
  );

  const hzPoints = melPoints.map(melToHz);
  const binPoints = hzPoints.map(hz => Math.floor((fftSize + 1) * hz / sampleRate));

  const filterbank: number[][] = [];
  for (let i = 1; i <= numFilters; i++) {
    const filter = new Array(Math.floor(fftSize / 2) + 1).fill(0);

    const left = binPoints[i - 1];
    const center = binPoints[i];
    const right = binPoints[i + 1];

    for (let j = left; j < center; j++) {
      filter[j] = (j - left) / (center - left);
    }
    for (let j = center; j < right; j++) {
      filter[j] = (right - j) / (right - center);
    }

    filterbank.push(filter);
  }

  return filterbank;
}

// Precomputed DCT cosine table (lazily initialized)
let _dctTable: Float64Array | null = null;
let _dctN = 0;
let _dctK = 0;

function ensureDCTTable(N: number, numCoefficients: number): void {
  if (_dctN === N && _dctK === numCoefficients) return;
  _dctN = N;
  _dctK = numCoefficients;
  _dctTable = new Float64Array(numCoefficients * N);
  for (let k = 0; k < numCoefficients; k++) {
    for (let n = 0; n < N; n++) {
      _dctTable[k * N + n] = Math.cos((Math.PI * k * (n + 0.5)) / N);
    }
  }
}

// Discrete Cosine Transform with precomputed cosine table
function dct(input: number[], numCoefficients: number): number[] {
  const N = input.length;
  ensureDCTTable(N, numCoefficients);
  const table = _dctTable!;
  const output: number[] = new Array(numCoefficients);

  for (let k = 0; k < numCoefficients; k++) {
    let sum = 0;
    const offset = k * N;
    for (let n = 0; n < N; n++) {
      sum += input[n] * table[offset + n];
    }
    output[k] = sum;
  }

  return output;
}

// Extract MFCCs from a pre-computed FFT magnitude spectrum using a cached filterbank
function extractMFCCsFromFFT(
  fft: Float32Array,
  filterbank: number[][],
  numMFCCs: number = 40
): number[] {
  // Apply filterbank to power spectrum
  const melEnergies: number[] = [];
  for (const filter of filterbank) {
    let energy = 0;
    for (let i = 0; i < fft.length; i++) {
      energy += (fft[i] * fft[i]) * filter[i];
    }
    melEnergies.push(Math.log(energy + 1e-10));
  }

  // Apply DCT to get MFCCs
  const mfccs = dct(melEnergies, numMFCCs);

  return mfccs;
}

// Calculate dominant frequency from FFT
function calculateDominantFrequency(fft: Float32Array, sampleRate: number, fftSize: number): number {
  let maxMag = 0;
  let maxBin = 0;

  for (let i = 1; i < fft.length; i++) {
    if (fft[i] > maxMag) {
      maxMag = fft[i];
      maxBin = i;
    }
  }

  return (maxBin * sampleRate) / fftSize;
}

function extractAudioFrames(samples: Float32Array, sampleRate: number): AudioFrame[] {
  const frameSize = 512;
  // Dynamically set hop size to cap total frames for performance
  // MFCC+PCA is heavier than simple FFT, so cap lower
  const MAX_FRAMES = 2000;
  const minHopSize = 256;
  const naturalFrames = Math.floor((samples.length - frameSize) / minHopSize);
  const hopSize = naturalFrames > MAX_FRAMES
    ? Math.floor((samples.length - frameSize) / MAX_FRAMES)
    : minHopSize;
  const numFrames = Math.floor((samples.length - frameSize) / hopSize);
  const frames: AudioFrame[] = [];

  // Pre-compute mel filterbank once (same for all frames)
  const numFilters = 26;
  const filterbank = createMelFilterbank(numFilters, frameSize, sampleRate);

  for (let i = 0; i < numFrames; i++) {
    const start = i * hopSize;
    const t = start / sampleRate;

    // Compute FFT once per frame and reuse for both MFCCs and frequency
    const fft = computeFFT(samples, start, frameSize);

    // Extract 40 MFCCs from the pre-computed FFT
    const mfccs = extractMFCCsFromFFT(fft, filterbank, 40);

    // Calculate amplitude for point sizing
    const amplitude = computeRMS(samples, start, frameSize);

    // Calculate dominant frequency from the same FFT
    const frequency = calculateDominantFrequency(fft, sampleRate, frameSize);

    frames.push({
      t,
      mfccs,
      amplitude,
      frequency,
    });
  }

  return frames;
}

// Expected MFCC dimension (must match extractMFCCs output length)
const PCA_MFCC_DIM = 40;

// Fallback 3D from first 3 MFCCs when PCA cannot be applied
function fallbackPcaCoordinates(frames: AudioFrame[]): AudioFrame[] {
  return frames.map((frame) => ({
    ...frame,
    pcaCoordinates: {
      x: (frame.mfccs[0] ?? 0) * 0.1,
      y: (frame.mfccs[1] ?? 0) * 0.1,
      z: (frame.mfccs[2] ?? 0) * 0.1,
    },
  }));
}

// Apply PCA to reduce MFCC dimensions from 40 to 3
function applyPCA(frames: AudioFrame[]): AudioFrame[] {
  if (frames.length === 0) return frames;
  if (frames.length < 2) return fallbackPcaCoordinates(frames);

  try {
    const numCols = PCA_MFCC_DIM;

    // For large datasets, subsample to compute PCA basis, then project all frames
    const PCA_SAMPLE_LIMIT = 2000;
    const sampleStep = frames.length > PCA_SAMPLE_LIMIT
      ? Math.floor(frames.length / PCA_SAMPLE_LIMIT)
      : 1;
    const sampledIndices: number[] = [];
    for (let i = 0; i < frames.length; i += sampleStep) {
      sampledIndices.push(i);
    }

    // Build sampled MFCC matrix for computing PCA basis
    const sampledData = sampledIndices.map((idx) => {
      const m = frames[idx].mfccs;
      if (m.length === numCols) return m;
      if (m.length > numCols) return m.slice(0, numCols);
      const row = [...m];
      while (row.length < numCols) row.push(0);
      return row;
    });

    const sampledMatrix = new Matrix(sampledData);

    // Compute means from sample
    const means = sampledMatrix.mean('column');

    // Center sampled data
    for (let col = 0; col < numCols; col++) {
      for (let row = 0; row < sampledMatrix.rows; row++) {
        sampledMatrix.set(row, col, sampledMatrix.get(row, col) - means[col]);
      }
    }

    // Compute covariance matrix from sample (40x40 — always small)
    const transposed = sampledMatrix.transpose();
    const covariance = transposed.mmul(sampledMatrix).div(sampledMatrix.rows - 1);

    // Eigenvalue decomposition
    const evd = new EigenvalueDecomposition(covariance, { assumeSymmetric: true });
    const eigenvectors = evd.eigenvectorMatrix;
    const eigenvalues = evd.realEigenvalues;

    if (eigenvectors.rows !== numCols || eigenvectors.columns !== numCols) {
      return fallbackPcaCoordinates(frames);
    }

    // Sort eigenvectors by eigenvalues (descending)
    const sorted = eigenvalues
      .map((val, idx) => ({ val, idx }))
      .sort((a, b) => b.val - a.val);

    // Extract top 3 principal component vectors (each length numCols)
    const pc0 = new Float64Array(numCols);
    const pc1 = new Float64Array(numCols);
    const pc2 = new Float64Array(numCols);
    const col0 = sorted[0].idx, col1 = sorted[1].idx, col2 = sorted[2].idx;
    for (let r = 0; r < numCols; r++) {
      pc0[r] = eigenvectors.get(r, col0);
      pc1[r] = eigenvectors.get(r, col1);
      pc2[r] = eigenvectors.get(r, col2);
    }

    // Project ALL frames using simple dot products (avoids building a huge Matrix)
    const coords: { x: number; y: number; z: number }[] = new Array(frames.length);
    let maxAbs = 0;
    for (let i = 0; i < frames.length; i++) {
      const m = frames[i].mfccs;
      let x = 0, y = 0, z = 0;
      for (let d = 0; d < numCols; d++) {
        const centered = (d < m.length ? m[d] : 0) - means[d];
        x += centered * pc0[d];
        y += centered * pc1[d];
        z += centered * pc2[d];
      }
      coords[i] = { x, y, z };
      maxAbs = Math.max(maxAbs, Math.abs(x), Math.abs(y), Math.abs(z));
    }

    const scale = maxAbs > 0 ? 5 / maxAbs : 1;

    return frames.map((frame, i) => ({
      ...frame,
      pcaCoordinates: {
        x: coords[i].x * scale,
        y: coords[i].y * scale,
        z: coords[i].z * scale,
      }
    }));
  } catch {
    return fallbackPcaCoordinates(frames);
  }
}

function segmentIntoVerses(frames: AudioFrame[], duration: number): { start: number; end: number; frames: AudioFrame[] }[] {
  const verses: { start: number; end: number; frames: AudioFrame[] }[] = [];
  
  if (frames.length === 0) return verses;

  const maxAmp = Math.max(...frames.map(f => f.amplitude || 0));
  const silenceThreshold = maxAmp * 0.1;

  let verseStart = 0;
  let verseFrames: AudioFrame[] = [];
  let inSilence = false;
  let silenceStart = 0;
  const minSilenceDuration = 0.2;
  const minVerseDuration = 0.3;

  for (const frame of frames) {
    const isSilent = (frame.amplitude || 0) < silenceThreshold;

    if (isSilent && !inSilence) {
      inSilence = true;
      silenceStart = frame.t;
    } else if (!isSilent && inSilence) {
      const silenceDuration = frame.t - silenceStart;
      if (silenceDuration > minSilenceDuration && verseFrames.length > 0) {
        const verseDuration = silenceStart - verseStart;
        if (verseDuration >= minVerseDuration) {
          verses.push({
            start: verseStart,
            end: silenceStart,
            frames: [...verseFrames],
          });
        }
        verseStart = frame.t;
        verseFrames = [];
      }
      inSilence = false;
    }

    if (!isSilent) {
      verseFrames.push(frame);
    }
  }

  if (verseFrames.length > 0) {
    verses.push({
      start: verseStart,
      end: duration,
      frames: verseFrames,
    });
  }

  if (verses.length === 0 && frames.length > 0) {
    const segmentDuration = duration / Math.min(4, Math.ceil(duration / 5));
    const numSegments = Math.ceil(duration / segmentDuration);
    for (let i = 0; i < numSegments; i++) {
      const start = i * segmentDuration;
      const end = Math.min((i + 1) * segmentDuration, duration);
      const segmentFrames = frames.filter(f => f.t >= start && f.t < end);
      if (segmentFrames.length > 0) {
        verses.push({ start, end, frames: segmentFrames });
      }
    }
  }

  return verses;
}

function normalizeValue(value: number, min: number, max: number): number {
  if (max === min) return 0.5;
  return Math.max(0, Math.min(1, (value - min) / (max - min)));
}

function findKNearestNeighbors(points: VisualizationPoint[], k: number): [number, number][] {
  const edges: [number, number][] = [];
  const edgeSet = new Set<string>();

  for (let i = 0; i < points.length; i++) {
    const distances: { index: number; dist: number }[] = [];

    for (let j = 0; j < points.length; j++) {
      if (i === j) continue;

      const dx = points[i].x - points[j].x;
      const dy = points[i].y - points[j].y;
      const dz = points[i].z - points[j].z;
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);

      distances.push({ index: j, dist });
    }

    distances.sort((a, b) => a.dist - b.dist);

    for (let n = 0; n < Math.min(k, distances.length); n++) {
      const j = distances[n].index;
      const edgeKey = i < j ? `${i}-${j}` : `${j}-${i}`;

      if (!edgeSet.has(edgeKey)) {
        edgeSet.add(edgeKey);
        edges.push([i, j]);
      }
    }
  }

  return edges;
}

// Map frequency to hue for color coding (red/orange/yellow gradient)
function mapFrequencyToHue(frequency: number): number {
  // Map frequency range 0-10kHz to hue 0-60° (red→orange→yellow)
  const normalized = Math.min(frequency / 10000, 1);
  return normalized * 60;
}

function mapFramesToVisualization(
  verseSegments: { start: number; end: number; frames: AudioFrame[] }[],
  duration: number
): Verse[] {
  const allFrames = verseSegments.flatMap(v => v.frames);

  if (allFrames.length === 0) {
    return [];
  }

  const amplitudeMax = Math.max(...allFrames.map(f => f.amplitude || 0));

  return verseSegments.map((segment, verseIndex) => {
    const maxPoints = 400;
    const step = Math.max(1, Math.floor(segment.frames.length / maxPoints));
    const sampledFrames = segment.frames.filter((_, i) => i % step === 0);

    const points: VisualizationPoint[] = sampledFrames.map(frame => {
      // Use PCA coordinates directly
      const { x, y, z } = frame.pcaCoordinates || { x: 0, y: 0, z: 0 };

      // Frequency-based color (match video: red/orange/yellow)
      const hue = mapFrequencyToHue(frame.frequency || 0);
      const saturation = 0.8;
      const lightness = 0.6;

      // Size based on amplitude
      const normalizedAmplitude = amplitudeMax > 0 ? (frame.amplitude || 0) / amplitudeMax : 0.5;
      const size = 0.05 + normalizedAmplitude * 0.15;

      return {
        x,
        y,
        z,
        size,
        color: [hue, saturation, lightness] as [number, number, number],
        time: frame.t,
        mfccs: frame.mfccs,
        frequency: frame.frequency,
      };
    });

    // Compute k-nearest neighbors based on Euclidean distance in 3D PCA space
    const edges = points.length > 1 ? findKNearestNeighbors(points, 3) : [];

    return {
      id: verseIndex,
      name: `Phrase ${verseIndex + 1}`,
      start: segment.start,
      end: segment.end,
      points,
      edges,
    };
  });
}

async function transcodeToWav(inputPath: string): Promise<string> {
  const tempWavPath = path.join(os.tmpdir(), `${randomUUID()}.wav`);

  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      command.kill("SIGKILL");
      reject(new Error("Audio transcoding timed out after 60 seconds."));
    }, 60000);

    const command = ffmpeg(inputPath)
      .toFormat("wav")
      .audioCodec("pcm_s16le")
      .audioChannels(1)
      .audioFrequency(22050)
      .on("error", (err) => {
        clearTimeout(timeout);
        reject(new Error(`Failed to transcode audio: ${err.message}`));
      })
      .on("end", () => {
        clearTimeout(timeout);
        resolve(tempWavPath);
      })
      .save(tempWavPath);
  });
}

export async function analyzeAudioFile(filePath: string): Promise<AnalysisResult> {
  const ext = path.extname(filePath).toLowerCase();
  const supportedFormats = [".wav", ".mp3", ".m4a"];
  
  if (!supportedFormats.includes(ext)) {
    throw new Error("Unsupported audio format. Please upload WAV, MP3, or M4A files.");
  }
  
  let wavFilePath = filePath;
  let needsCleanup = false;
  
  if (ext !== ".wav") {
    wavFilePath = await transcodeToWav(filePath);
    needsCleanup = true;
  }
  
  try {
    const buffer = fs.readFileSync(wavFilePath);
    
    const header = parseWavHeader(buffer);
    if (!header) {
      throw new Error("Failed to parse audio file. Please ensure it's a valid audio file.");
    }
    
    const sampleRate = header.sampleRate;
    const samples = extractPcmSamples(buffer, header);
    
    if (samples.length === 0) {
      throw new Error("Could not extract audio samples from the file.");
    }
    
    const duration = samples.length / sampleRate;

    const frames = extractAudioFrames(samples, sampleRate);

    // Apply PCA to reduce 40D MFCCs to 3D coordinates
    const framesWithPCA = applyPCA(frames);

    const verseSegments = segmentIntoVerses(framesWithPCA, duration);

    const verses = mapFramesToVisualization(verseSegments, duration);
    
    return {
      duration,
      sampleRate,
      verses,
    };
  } finally {
    if (needsCleanup && fs.existsSync(wavFilePath)) {
      fs.unlinkSync(wavFilePath);
    }
  }
}

// Keep this for future MP3/M4A support
function _generateFallbackSamples(buffer: Buffer, sampleRate: number): Float32Array {
  const fileSize = buffer.length;
  const estimatedDuration = Math.min(60, Math.max(5, fileSize / (44100 * 2)));
  const numSamples = Math.floor(estimatedDuration * sampleRate);
  const samples = new Float32Array(numSamples);
  
  for (let i = 0; i < numSamples; i++) {
    const bufferIndex = Math.floor((i / numSamples) * buffer.length);
    const byte1 = buffer[bufferIndex] || 0;
    const byte2 = buffer[Math.min(bufferIndex + 1, buffer.length - 1)] || 0;
    const value = ((byte1 << 8) | byte2) / 65535;
    samples[i] = (value - 0.5) * 2;
  }
  
  return samples;
}

export async function getAudioDuration(filePath: string): Promise<number> {
  const ext = path.extname(filePath).toLowerCase();
  
  if (ext === ".wav") {
    const buffer = fs.readFileSync(filePath);
    const header = parseWavHeader(buffer);
    if (header) {
      const numSamples = header.dataLength / (header.bitsPerSample / 8) / header.channels;
      return numSamples / header.sampleRate;
    }
  }
  
  return new Promise((resolve, reject) => {
    ffmpeg.ffprobe(filePath, (err, metadata) => {
      if (err) {
        reject(new Error(`Failed to get audio duration: ${err.message}`));
        return;
      }
      const duration = metadata.format.duration || 0;
      resolve(duration);
    });
  });
}
