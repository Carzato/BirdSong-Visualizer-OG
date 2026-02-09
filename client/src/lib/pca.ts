/**
 * PCA (Principal Component Analysis) for projecting high-dimensional
 * audio features into 3D space for visualization.
 *
 * Uses eigendecomposition of the covariance matrix via power iteration.
 */

import type { FrameFeatures } from "@shared/schema";

export interface PCAResult {
  positions: [number, number, number][];
  eigenvalues: number[];
  mean: number[];
  components: number[][]; // top 3 eigenvectors
}

/**
 * Build the feature vector from a FrameFeatures record.
 * Concatenates MFCCs (40), chroma (12), scaled pitch, centroid, loudness,
 * bandwidth, and chroma concentration into a single vector.
 */
function buildFeatureVector(frame: FrameFeatures): number[] {
  const v: number[] = [];

  // MFCCs (40 dimensions)
  for (let i = 0; i < frame.mfcc.length; i++) {
    v.push(frame.mfcc[i]);
  }

  // Chroma (12 dimensions)
  for (let i = 0; i < frame.chroma.length; i++) {
    v.push(frame.chroma[i]);
  }

  // Scaled pitch: log2(f0) normalized roughly to [0,1] range
  // f0 range ~50-2000 Hz → log2 range ~5.6-11
  const logF0 = frame.f0 ? Math.log2(Math.max(1, frame.f0)) / 11 : 0;
  v.push(logF0 * frame.f0Conf); // weight by confidence

  // Spectral centroid (scaled by typical max ~8000 Hz)
  v.push(frame.centroid / 8000);

  // Loudness (RMS, typically 0-1 range)
  v.push(frame.loudness);

  // Bandwidth (scaled)
  v.push(frame.bandwidth / 4000);

  // Chroma concentration: max(chroma) — high when one pitch class dominates
  const chromaConc = Math.max(...frame.chroma);
  v.push(chromaConc);

  return v;
}

/**
 * Compute mean of each feature across all frames.
 */
function computeMean(data: number[][]): number[] {
  const N = data.length;
  const D = data[0].length;
  const mean = new Array(D).fill(0);
  for (let i = 0; i < N; i++) {
    for (let d = 0; d < D; d++) {
      mean[d] += data[i][d];
    }
  }
  for (let d = 0; d < D; d++) mean[d] /= N;
  return mean;
}

/**
 * Compute standard deviation of each feature.
 */
function computeStd(data: number[][], mean: number[]): number[] {
  const N = data.length;
  const D = data[0].length;
  const std = new Array(D).fill(0);
  for (let i = 0; i < N; i++) {
    for (let d = 0; d < D; d++) {
      const diff = data[i][d] - mean[d];
      std[d] += diff * diff;
    }
  }
  for (let d = 0; d < D; d++) {
    std[d] = Math.sqrt(std[d] / N);
    if (std[d] < 1e-10) std[d] = 1; // avoid division by zero
  }
  return std;
}

/**
 * Center and standardize data (z-score normalization).
 */
function standardize(data: number[][], mean: number[], std: number[]): number[][] {
  const N = data.length;
  const D = data[0].length;
  const result: number[][] = [];
  for (let i = 0; i < N; i++) {
    const row = new Array(D);
    for (let d = 0; d < D; d++) {
      row[d] = (data[i][d] - mean[d]) / std[d];
    }
    result.push(row);
  }
  return result;
}

/**
 * Compute covariance matrix (D×D) from standardized data.
 * Uses the efficient (1/N) * X^T * X formulation.
 */
function computeCovarianceMatrix(data: number[][]): number[][] {
  const N = data.length;
  const D = data[0].length;
  const cov: number[][] = [];
  for (let i = 0; i < D; i++) {
    cov[i] = new Array(D).fill(0);
  }
  for (let n = 0; n < N; n++) {
    for (let i = 0; i < D; i++) {
      for (let j = i; j < D; j++) {
        cov[i][j] += data[n][i] * data[n][j];
      }
    }
  }
  for (let i = 0; i < D; i++) {
    for (let j = i; j < D; j++) {
      cov[i][j] /= N;
      cov[j][i] = cov[i][j]; // symmetric
    }
  }
  return cov;
}

/**
 * Power iteration to find the dominant eigenvector of a matrix.
 */
function powerIteration(
  matrix: number[][],
  numIter = 200
): { eigenvalue: number; eigenvector: number[] } {
  const D = matrix.length;
  let v = new Array(D);
  // Deterministic init: alternating pattern to avoid degeneracy
  for (let i = 0; i < D; i++) v[i] = (i % 2 === 0 ? 1 : -1) * (1 + (i % 7) * 0.1);
  // Normalize
  let initNorm = 0;
  for (let i = 0; i < D; i++) initNorm += v[i] * v[i];
  initNorm = Math.sqrt(initNorm);
  if (initNorm > 0) for (let i = 0; i < D; i++) v[i] /= initNorm;

  let eigenvalue = 0;

  for (let iter = 0; iter < numIter; iter++) {
    // Matrix-vector multiply: w = M * v
    const w = new Array(D).fill(0);
    for (let i = 0; i < D; i++) {
      for (let j = 0; j < D; j++) {
        w[i] += matrix[i][j] * v[j];
      }
    }

    // Compute norm
    let norm = 0;
    for (let i = 0; i < D; i++) norm += w[i] * w[i];
    norm = Math.sqrt(norm);
    if (norm < 1e-10) break;

    eigenvalue = norm;
    for (let i = 0; i < D; i++) v[i] = w[i] / norm;
  }

  return { eigenvalue, eigenvector: v };
}

/**
 * Deflate the matrix by removing the contribution of an eigenvector.
 */
function deflateMatrix(matrix: number[][], eigenvalue: number, eigenvector: number[]): number[][] {
  const D = matrix.length;
  const result: number[][] = [];
  for (let i = 0; i < D; i++) {
    result[i] = new Array(D);
    for (let j = 0; j < D; j++) {
      result[i][j] = matrix[i][j] - eigenvalue * eigenvector[i] * eigenvector[j];
    }
  }
  return result;
}

/**
 * Run PCA on FrameFeatures and return 3D positions.
 */
export function computePCA(frames: FrameFeatures[]): PCAResult {
  if (frames.length === 0) {
    return { positions: [], eigenvalues: [], mean: [], components: [] };
  }

  // Build feature matrix
  const rawData = frames.map(buildFeatureVector);
  const D = rawData[0].length;

  // Standardize
  const mean = computeMean(rawData);
  const std = computeStd(rawData, mean);
  const data = standardize(rawData, mean, std);

  // Covariance matrix
  const cov = computeCovarianceMatrix(data);

  // Extract top 3 eigenvectors via power iteration + deflation
  const eigenvalues: number[] = [];
  const components: number[][] = [];
  let currentCov = cov;

  for (let k = 0; k < 3; k++) {
    const { eigenvalue, eigenvector } = powerIteration(currentCov);
    eigenvalues.push(eigenvalue);
    components.push(eigenvector);
    currentCov = deflateMatrix(currentCov, eigenvalue, eigenvector);
  }

  // Project data onto top 3 components
  const positions: [number, number, number][] = [];
  for (let i = 0; i < data.length; i++) {
    const pos: [number, number, number] = [0, 0, 0];
    for (let k = 0; k < 3; k++) {
      let dot = 0;
      for (let d = 0; d < D; d++) {
        dot += data[i][d] * components[k][d];
      }
      pos[k] = dot;
    }
    positions.push(pos);
  }

  // Normalize positions to roughly [-5, 5] range for visualization
  const ranges: [number, number][] = [[Infinity, -Infinity], [Infinity, -Infinity], [Infinity, -Infinity]];
  for (const pos of positions) {
    for (let k = 0; k < 3; k++) {
      if (pos[k] < ranges[k][0]) ranges[k][0] = pos[k];
      if (pos[k] > ranges[k][1]) ranges[k][1] = pos[k];
    }
  }
  const scale = 5;
  for (const pos of positions) {
    for (let k = 0; k < 3; k++) {
      const range = ranges[k][1] - ranges[k][0];
      if (range > 0) {
        pos[k] = ((pos[k] - ranges[k][0]) / range - 0.5) * 2 * scale;
      }
    }
  }

  return { positions, eigenvalues, mean, components };
}
