/**
 * Client-side audio feature extraction using Web Audio API.
 * Computes MFCCs, chroma, pitch (YIN), and spectral descriptors per frame.
 */

import type { FrameFeatures } from "@shared/schema";

// ─── FFT (Cooley-Tukey radix-2, in-place) ─────────────────────────────────

function fft(re: Float64Array, im: Float64Array): void {
  const N = re.length;
  // Bit-reversal permutation
  for (let i = 1, j = 0; i < N; i++) {
    let bit = N >> 1;
    while (j & bit) { j ^= bit; bit >>= 1; }
    j ^= bit;
    if (i < j) {
      let tmp = re[i]; re[i] = re[j]; re[j] = tmp;
      tmp = im[i]; im[i] = im[j]; im[j] = tmp;
    }
  }
  // Butterfly stages
  for (let len = 2; len <= N; len <<= 1) {
    const half = len >> 1;
    const angle = -2 * Math.PI / len;
    const wRe = Math.cos(angle);
    const wIm = Math.sin(angle);
    for (let i = 0; i < N; i += len) {
      let curRe = 1, curIm = 0;
      for (let j = 0; j < half; j++) {
        const idx = i + j + half;
        const tRe = curRe * re[idx] - curIm * im[idx];
        const tIm = curRe * im[idx] + curIm * re[idx];
        re[idx] = re[i + j] - tRe;
        im[idx] = im[i + j] - tIm;
        re[i + j] += tRe;
        im[i + j] += tIm;
        const nRe = curRe * wRe - curIm * wIm;
        curIm = curRe * wIm + curIm * wRe;
        curRe = nRe;
      }
    }
  }
}

function computeMagnitudeSpectrum(
  samples: Float32Array,
  start: number,
  frameSize: number,
  window: Float64Array
): Float64Array {
  const re = new Float64Array(frameSize);
  const im = new Float64Array(frameSize);
  const end = Math.min(start + frameSize, samples.length);
  for (let i = 0; i < frameSize; i++) {
    re[i] = (start + i < end) ? samples[start + i] * window[i] : 0;
  }
  fft(re, im);
  const halfN = frameSize >> 1;
  const mag = new Float64Array(halfN + 1);
  for (let k = 0; k <= halfN; k++) {
    mag[k] = Math.sqrt(re[k] * re[k] + im[k] * im[k]);
  }
  return mag;
}

// ─── Hann window ───────────────────────────────────────────────────────────

function hannWindow(size: number): Float64Array {
  const w = new Float64Array(size);
  for (let i = 0; i < size; i++) {
    w[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / (size - 1)));
  }
  return w;
}

// ─── Mel filterbank ────────────────────────────────────────────────────────

function hzToMel(f: number): number { return 2595 * Math.log10(1 + f / 700); }
function melToHz(m: number): number { return 700 * (Math.pow(10, m / 2595) - 1); }

function createMelFilterbank(
  numFilters: number,
  fftSize: number,
  sampleRate: number,
  lowFreq = 20,
  highFreq?: number
): Float64Array[] {
  if (!highFreq) highFreq = sampleRate / 2;
  const numBins = (fftSize >> 1) + 1;
  const melLow = hzToMel(lowFreq);
  const melHigh = hzToMel(highFreq);
  const melPoints = new Float64Array(numFilters + 2);
  for (let i = 0; i < numFilters + 2; i++) {
    melPoints[i] = melLow + (melHigh - melLow) * i / (numFilters + 1);
  }
  const binPoints = new Float64Array(numFilters + 2);
  for (let i = 0; i < numFilters + 2; i++) {
    binPoints[i] = Math.floor((melToHz(melPoints[i]) / (sampleRate / 2)) * (numBins - 1));
  }
  const filters: Float64Array[] = [];
  for (let m = 0; m < numFilters; m++) {
    const filter = new Float64Array(numBins);
    const left = binPoints[m];
    const center = binPoints[m + 1];
    const right = binPoints[m + 2];
    for (let k = Math.floor(left); k <= Math.ceil(right) && k < numBins; k++) {
      if (k >= left && k <= center && center > left) {
        filter[k] = (k - left) / (center - left);
      } else if (k > center && k <= right && right > center) {
        filter[k] = (right - k) / (right - center);
      }
    }
    filters.push(filter);
  }
  return filters;
}

// ─── DCT (Type II) ────────────────────────────────────────────────────────

function dctII(input: Float64Array, numCoeffs: number): Float64Array {
  const N = input.length;
  const out = new Float64Array(numCoeffs);
  for (let k = 0; k < numCoeffs; k++) {
    let sum = 0;
    for (let n = 0; n < N; n++) {
      sum += input[n] * Math.cos(Math.PI * k * (n + 0.5) / N);
    }
    out[k] = sum;
  }
  return out;
}

// ─── MFCC extraction ──────────────────────────────────────────────────────

function computeMFCC(
  magnitude: Float64Array,
  melFilters: Float64Array[],
  numCoeffs: number
): number[] {
  const numFilters = melFilters.length;
  const melEnergies = new Float64Array(numFilters);
  for (let m = 0; m < numFilters; m++) {
    let energy = 0;
    for (let k = 0; k < magnitude.length && k < melFilters[m].length; k++) {
      energy += magnitude[k] * magnitude[k] * melFilters[m][k];
    }
    melEnergies[m] = Math.log(Math.max(energy, 1e-10));
  }
  const coeffs = dctII(melEnergies, numCoeffs);
  return Array.from(coeffs);
}

// ─── Chroma (pitch-class profile from FFT) ─────────────────────────────────

function computeChroma(
  magnitude: Float64Array,
  fftSize: number,
  sampleRate: number
): number[] {
  const chroma = new Float64Array(12);
  const halfN = magnitude.length;
  const refFreq = 440; // A4

  for (let k = 1; k < halfN; k++) {
    const freq = k * sampleRate / fftSize;
    if (freq < 30 || freq > 5000) continue;
    // Map frequency to pitch class: C=0, C#=1, ..., B=11
    const semitone = 12 * Math.log2(freq / refFreq);
    const pitchClass = ((Math.round(semitone) % 12) + 12) % 12;
    chroma[pitchClass] += magnitude[k] * magnitude[k];
  }

  // L1 normalize
  let sum = 0;
  for (let i = 0; i < 12; i++) sum += chroma[i];
  if (sum > 0) {
    for (let i = 0; i < 12; i++) chroma[i] /= sum;
  }
  return Array.from(chroma);
}

// ─── YIN pitch detection ───────────────────────────────────────────────────

function yinPitchDetect(
  samples: Float32Array,
  start: number,
  frameSize: number,
  sampleRate: number,
  threshold = 0.15
): { f0: number | null; confidence: number } {
  const halfW = Math.floor(frameSize / 2);
  const end = Math.min(start + frameSize, samples.length);
  const available = end - start;
  if (available < frameSize) return { f0: null, confidence: 0 };

  // Step 1: Difference function
  const diff = new Float64Array(halfW);
  for (let tau = 0; tau < halfW; tau++) {
    let sum = 0;
    for (let j = 0; j < halfW; j++) {
      const d = samples[start + j] - samples[start + j + tau];
      sum += d * d;
    }
    diff[tau] = sum;
  }

  // Step 2: Cumulative mean normalized difference
  const cmndf = new Float64Array(halfW);
  cmndf[0] = 1;
  let runningSum = 0;
  for (let tau = 1; tau < halfW; tau++) {
    runningSum += diff[tau];
    cmndf[tau] = runningSum > 0 ? (diff[tau] * tau / runningSum) : 1;
  }

  // Step 3: Absolute threshold
  // Find first tau where cmndf dips below threshold, then pick its minimum
  let tau = 2;
  const minPeriod = Math.floor(sampleRate / 2000); // max 2000 Hz
  const maxPeriod = Math.floor(sampleRate / 50);    // min 50 Hz

  let bestTau = -1;
  let bestVal = 1;

  for (tau = Math.max(2, minPeriod); tau < Math.min(halfW, maxPeriod); tau++) {
    if (cmndf[tau] < threshold) {
      // Find local minimum after crossing threshold
      while (tau + 1 < halfW && cmndf[tau + 1] < cmndf[tau]) {
        tau++;
      }
      bestTau = tau;
      bestVal = cmndf[tau];
      break;
    }
  }

  // If no threshold crossing found, find global minimum
  if (bestTau < 0) {
    for (tau = Math.max(2, minPeriod); tau < Math.min(halfW, maxPeriod); tau++) {
      if (cmndf[tau] < bestVal) {
        bestVal = cmndf[tau];
        bestTau = tau;
      }
    }
  }

  if (bestTau < 2) return { f0: null, confidence: 0 };

  // Step 4: Parabolic interpolation
  let refinedTau = bestTau;
  if (bestTau > 0 && bestTau < halfW - 1) {
    const a = cmndf[bestTau - 1];
    const b = cmndf[bestTau];
    const c = cmndf[bestTau + 1];
    const shift = (a - c) / (2 * (a - 2 * b + c));
    if (Math.abs(shift) < 1) refinedTau = bestTau + shift;
  }

  const f0 = sampleRate / refinedTau;
  const confidence = 1 - bestVal;

  if (f0 < 50 || f0 > 2000 || confidence < 0.3) {
    return { f0: null, confidence: 0 };
  }

  return { f0, confidence: Math.max(0, Math.min(1, confidence)) };
}

// ─── Spectral features ────────────────────────────────────────────────────

function computeSpectralCentroid(
  magnitude: Float64Array,
  fftSize: number,
  sampleRate: number
): number {
  let sumMag = 0, sumFreqMag = 0;
  for (let k = 1; k < magnitude.length; k++) {
    const freq = k * sampleRate / fftSize;
    sumMag += magnitude[k];
    sumFreqMag += freq * magnitude[k];
  }
  return sumMag > 0 ? sumFreqMag / sumMag : 0;
}

function computeSpectralBandwidth(
  magnitude: Float64Array,
  fftSize: number,
  sampleRate: number,
  centroid: number
): number {
  let sumMag = 0, sumSpread = 0;
  for (let k = 1; k < magnitude.length; k++) {
    const freq = k * sampleRate / fftSize;
    sumMag += magnitude[k];
    sumSpread += magnitude[k] * (freq - centroid) * (freq - centroid);
  }
  return sumMag > 0 ? Math.sqrt(sumSpread / sumMag) : 0;
}

function computeSpectralFlatness(magnitude: Float64Array): number {
  let logSum = 0, sum = 0, count = 0;
  for (let k = 1; k < magnitude.length; k++) {
    const val = Math.max(magnitude[k], 1e-10);
    logSum += Math.log(val);
    sum += val;
    count++;
  }
  if (count === 0 || sum === 0) return 0;
  const geoMean = Math.exp(logSum / count);
  const arithMean = sum / count;
  return geoMean / arithMean;
}

function computeSpectralFlux(
  current: Float64Array,
  previous: Float64Array | null
): number {
  if (!previous) return 0;
  let flux = 0;
  const len = Math.min(current.length, previous.length);
  for (let k = 0; k < len; k++) {
    const d = current[k] - previous[k];
    if (d > 0) flux += d * d;
  }
  return Math.sqrt(flux);
}

function computeRMS(samples: Float32Array, start: number, length: number): number {
  let sum = 0;
  const end = Math.min(start + length, samples.length);
  const n = end - start;
  if (n <= 0) return 0;
  for (let i = start; i < end; i++) {
    sum += samples[i] * samples[i];
  }
  return Math.sqrt(sum / n);
}

// ─── Main analysis pipeline ───────────────────────────────────────────────

export interface AnalysisConfig {
  frameSize: number;   // FFT window size (power of 2), default 2048
  hopSize: number;     // hop between frames, default 512
  numMFCC: number;     // number of MFCC coefficients, default 40
  numMelBands: number; // number of mel filterbank channels, default 80
  onProgress?: (progress: number) => void;
}

const defaultConfig: AnalysisConfig = {
  frameSize: 2048,
  hopSize: 512,
  numMFCC: 40,
  numMelBands: 80,
};

export async function extractFeatures(
  audioBuffer: AudioBuffer,
  config: Partial<AnalysisConfig> = {}
): Promise<FrameFeatures[]> {
  const cfg = { ...defaultConfig, ...config };
  const sampleRate = audioBuffer.sampleRate;

  // Mix to mono
  const samples = new Float32Array(audioBuffer.length);
  const numChannels = audioBuffer.numberOfChannels;
  for (let ch = 0; ch < numChannels; ch++) {
    const channelData = audioBuffer.getChannelData(ch);
    for (let i = 0; i < audioBuffer.length; i++) {
      samples[i] += channelData[i] / numChannels;
    }
  }

  // Precompute window and mel filterbank
  const window = hannWindow(cfg.frameSize);
  const melFilters = createMelFilterbank(cfg.numMelBands, cfg.frameSize, sampleRate);

  const numFrames = Math.floor((samples.length - cfg.frameSize) / cfg.hopSize);
  const frames: FrameFeatures[] = [];
  let prevMag: Float64Array | null = null;

  for (let i = 0; i < numFrames; i++) {
    const start = i * cfg.hopSize;
    const time = start / sampleRate;

    // Magnitude spectrum
    const mag = computeMagnitudeSpectrum(samples, start, cfg.frameSize, window);

    // MFCCs
    const mfcc = computeMFCC(mag, melFilters, cfg.numMFCC);

    // Chroma
    const chroma = computeChroma(mag, cfg.frameSize, sampleRate);

    // Pitch (YIN)
    const { f0, confidence: f0Conf } = yinPitchDetect(
      samples, start, cfg.frameSize, sampleRate
    );

    // Spectral features
    const loudness = computeRMS(samples, start, cfg.frameSize);
    const centroid = computeSpectralCentroid(mag, cfg.frameSize, sampleRate);
    const bandwidth = computeSpectralBandwidth(mag, cfg.frameSize, sampleRate, centroid);
    const flatness = computeSpectralFlatness(mag);
    const flux = computeSpectralFlux(mag, prevMag);

    frames.push({
      time,
      mfcc,
      chroma,
      f0,
      f0Conf,
      loudness,
      centroid,
      bandwidth,
      flatness,
      flux,
    });

    prevMag = mag;

    // Report progress every 100 frames
    if (cfg.onProgress && i % 100 === 0) {
      cfg.onProgress(i / numFrames);
      // Yield to UI thread
      await new Promise(r => setTimeout(r, 0));
    }
  }

  if (cfg.onProgress) cfg.onProgress(1);
  return frames;
}

/**
 * Decode an audio URL into an AudioBuffer using Web Audio API.
 */
export async function decodeAudioFromUrl(url: string): Promise<AudioBuffer> {
  let response: Response;
  try {
    response = await fetch(url);
  } catch (err) {
    throw new Error(`Network error fetching audio: ${err instanceof Error ? err.message : String(err)}`);
  }
  if (!response.ok) {
    throw new Error(`Failed to fetch audio (HTTP ${response.status}): ${response.statusText}`);
  }
  const arrayBuffer = await response.arrayBuffer();
  const audioCtx = new AudioContext({ sampleRate: 44100 });
  try {
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
    return audioBuffer;
  } catch (err) {
    throw new Error(`Failed to decode audio data: ${err instanceof Error ? err.message : String(err)}`);
  } finally {
    await audioCtx.close();
  }
}
