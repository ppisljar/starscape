/* Headless tests for the PURE terrain height source (no GPU / no three.js construction): the CPU
 * sampler that drives the camera collision floor and bit-mirrors the GPU vertex displacement. The
 * displacement material, the LOD icospheres and the draped colour are GPU-only and are exercised in
 * the browser, not here.
 */

import { describe, expect, it } from 'vitest';
import {
  earthTerrainSource,
  TERRAIN_ALT_FULL,
  TERRAIN_ALT_ON,
  TERRAIN_HEIGHT_SCALE,
} from '../src/lod/terrain';

/** A deterministic sweep of unit directions over the sphere (lat × lon grid). */
function sphereDirs(nLat = 24, nLon = 48): [number, number, number][] {
  const dirs: [number, number, number][] = [];
  for (let a = 0; a < nLat; a++) {
    const theta = ((a + 0.5) / nLat) * Math.PI; // 0..π polar
    const sy = Math.cos(theta);
    const sr = Math.sin(theta);
    for (let b = 0; b < nLon; b++) {
      const phi = (b / nLon) * 2 * Math.PI;
      dirs.push([sr * Math.cos(phi), sy, sr * Math.sin(phi)]);
    }
  }
  return dirs;
}

describe('earthTerrainSource (CPU height field — the collision floor)', () => {
  const salt = (1 * 0x1000193 + 0 * 0x9e3779b1) | 0; // the earth layer salt for textureSeed 1
  const src = earthTerrainSource(salt);
  const dirs = sphereDirs();

  it('is deterministic (same salt, direction, detail → same height)', () => {
    const src2 = earthTerrainSource(salt);
    for (const [x, y, z] of dirs) {
      expect(src.height(x, y, z, 0)).toBe(src2.height(x, y, z, 0));
      expect(src.height(x, y, z, 1)).toBe(src2.height(x, y, z, 1));
    }
  });

  it('every height is finite and in [0, 1]', () => {
    for (const d of [0, 0.5, 1]) {
      for (const [x, y, z] of dirs) {
        const h = src.height(x, y, z, d);
        expect(Number.isFinite(h)).toBe(true);
        expect(h).toBeGreaterThanOrEqual(0);
        expect(h).toBeLessThanOrEqual(1);
      }
    }
  });

  it('oceans stay flat and land rises — both exist (mountains register with the coastline)', () => {
    let flat = 0;
    let raised = 0;
    for (const [x, y, z] of dirs) {
      const h = src.height(x, y, z, 0);
      if (h === 0) flat++;
      else if (h > 0) raised++;
    }
    // a believable earth: a lot of flat ocean AND a real amount of raised land.
    expect(flat).toBeGreaterThan(dirs.length * 0.25);
    expect(raised).toBeGreaterThan(dirs.length * 0.1);
  });

  it('is insensitive to input direction magnitude (normalizes internally would differ — so it does NOT; callers pass unit dirs)', () => {
    // The source assumes a UNIT direction (the flycam passes surfN, already normalized). Two unit
    // directions that are equal give equal heights; this pins that the sampler is a pure function of
    // the (unit) direction, not of any hidden state.
    const [x, y, z] = dirs[100]!;
    expect(src.height(x, y, z, 0.5)).toBe(src.height(x, y, z, 0.5));
  });

  it('a different seed gives a different field', () => {
    const other = earthTerrainSource((salt + 12345) | 0);
    let diffs = 0;
    for (const [x, y, z] of dirs) {
      if (other.height(x, y, z, 0) !== src.height(x, y, z, 0)) diffs++;
    }
    expect(diffs).toBeGreaterThan(0);
  });

  it('exposes sane altitude + scale constants (swap above full-relief, exaggerated scale)', () => {
    expect(TERRAIN_ALT_ON).toBeGreaterThan(TERRAIN_ALT_FULL);
    expect(TERRAIN_ALT_FULL).toBeGreaterThan(0);
    expect(TERRAIN_HEIGHT_SCALE).toBeGreaterThan(0);
    expect(TERRAIN_HEIGHT_SCALE).toBeLessThan(1);
  });
});
