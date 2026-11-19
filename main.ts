// gta6-trophy-rarity-forecast-01 — entry (Node 22, standard library only)
//
// Forecasts each GTA VI trophy's earn rate on Exophase at the case's target,
// as five quantiles (5/25/50/75/95 %), scored by mean pinball loss on the
// logit scale. Runs offline, no network, deterministic.
//
// Strategy:
//   1. Flat history (all reads show the same rate): Exophase has not yet
//      refreshed its player counts between the checkpoint and the target. A
//      refresh dilutes earn rates. The dilution factor depends on the rate
//      itself (rare trophies drop more). A player-count tilt and a
//      trophies-per-player trend refine the median. An absolute spread floor
//      keeps the band wide enough to cover a sudden refresh.
//   2. Moving history: fits a least-squares line of logit(rate) vs hours since
//      first read (only reads in the last 24h), extrapolates to the target for
//      the median, and derives the spread from residual volatility scaled by
//      √(horizon days), plus a player-count volatility term.
//   3. Trophies at 100%: stay at 100% (q = [99.99, 99.99, 99.99, 99.99, 99.99]).
//
// Rates are clamped to [0.01, 99.99] before the logit, exactly as the scorer
// clamps; quantiles are emitted non-decreasing, rounded to 4 decimals.

import {
  mkdirSync,
  readdirSync,
  readFileSync,
  writeFileSync,
} from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
const Z = [-1.645, -0.674, 0, 0.674, 1.645];

// Flat-history dilution factor: rare trophies drop more on refresh.
const DILUTION_LOW = 0.60;
const DILUTION_HIGH = 0.95;
// Absolute minimum spread in logit units (covers unseen Exophase refreshes).
const SPREAD_FLOOR = 0.60;
// Minimum daily logit volatility for moving-history trophies.
const SIGMA_DAILY_MIN = 0.30;
// Cap on daily volatility so a noisy history cannot produce an absurd band.
const SIGMA_DAILY_MAX = 1.50;
// Overall cap on spread, in logit units.
const SPREAD_CAP = 3.0;
// Only the most recent reads are used to fit a trend.
const TREND_WINDOW_HOURS = 24;
// Downward median tilt per log unit of player-count growth.
const DILUTION_TILT = 0.20;
const DILUTION_MIN_GROWTH = 1.05;
// How much player-count volatility inflates the spread (logit units per
// std-dev of log players_tracked).
const PLAYER_VOL_SPREAD = 0.15;
// How much the trophies-per-player trend can tilt the median (logit units).
const EARNED_TILT = 0.15;

interface Trophy {
  id: number;
}

interface Read {
  slot: string;
  at: string;
  players_tracked: number;
  trophies_earned: number;
  rates: Array<number | null>;
}

interface Case {
  horizon_minutes: number;
  trophies: Trophy[];
  history: Read[];
  levels: number[];
}

const clamp = (x: number, lo: number, hi: number): number =>
  Math.min(hi, Math.max(lo, x));

const clampPct = (x: number): number => clamp(x, 0.01, 99.99);

const logit = (pct: number): number => {
  const p = clampPct(pct) / 100;
  return Math.log(p / (1 - p));
};

const invLogit = (l: number): number => {
  const p = 1 / (1 + Math.exp(-l));
  return clamp(p * 100, 0, 100);
};

const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;

// Population standard deviation.
const stdDev = (xs: number[]): number => {
  if (xs.length === 0) return 0;
  const mean = xs.reduce((a, b) => a + b, 0) / xs.length;
  const v = xs.reduce((a, b) => a + (b - mean) ** 2, 0) / xs.length;
  return Math.sqrt(v);
};

// Returns true if all non-null rates for trophy i are identical across reads.
function isFlatHistory(c: Case, i: number): boolean {
  let first: number | null = null;
  for (const h of c.history) {
    const r = h.rates?.[i];
    if (r === null || r <= 0) continue;
    if (first === null) first = r;
    else if (r !== first) return false;
  }
  return first !== null;
}

// Least-squares line: y = a + b*x, returns slope b (logit per hour).
// Uses only reads within TREND_WINDOW_HOURS before the checkpoint.
function slopeLogitPerHour(
  c: Case,
  i: number
): { a: number; b: number; residStd: number } | null {
  const xs: number[] = [];
  const ys: number[] = [];
  const t0 = new Date(c.history[0].at).getTime();
  const cutoffMs =
    new Date(c.history[c.history.length - 1].at).getTime() -
    TREND_WINDOW_HOURS * 60 * 60 * 1000;
  for (const h of c.history) {
    const r = h.rates?.[i];
    const atMs = new Date(h.at).getTime();
    if (r === null || r <= 0) continue;
    if (atMs < cutoffMs) continue;
    const hrs = (atMs - t0) / (1000 * 60 * 60);
    xs.push(hrs);
    ys.push(logit(r));
  }
  if (xs.length < 2) return null;
  const n = xs.length;
  const mx = xs.reduce((a, b) => a + b, 0) / n;
  const my = ys.reduce((a, b) => a + b, 0) / n;
  let sxx = 0;
  let sxy = 0;
  for (let k = 0; k < n; k++) {
    const dx = xs[k] - mx;
    sxx += dx * dx;
    sxy += dx * (ys[k] - my);
  }
  if (sxx === 0) return null;
  const b = sxy / sxx;
  const a = my - b * mx;
  const residStd = stdDev(
    ys.map((y, k) => y - (a + b * xs[k]))
  );
  return { a, b, residStd };
}

// Player-count log volatility over the last few reads (logit units).
function playerVolatility(c: Case): number {
  const ps = c.history
    .map((h) => h.players_tracked)
    .filter((p) => p > 0);
  if (ps.length < 2) return 0;
  const logs = ps.map((p) => Math.log(p));
  return stdDev(logs);
}

// Trophies-per-player trend over the last few reads.
// Returns a logit-tilt: negative means new players are earning fewer trophies
// per capita, suggesting dilution.
function earnedRatioTilt(c: Case): number {
  const ratios = c.history
    .map((h) => {
      const pt = h.players_tracked;
      const te = h.trophies_earned;
      if (!pt || pt <= 0) return null;
      return te / pt;
    })
    .filter((v) => v !== null) as number[];
  if (ratios.length < 2) return 0;
  // Compare the average of the last 2 reads to the average of the first 2.
  const n = ratios.length;
  const k = Math.min(2, n);
  const firstAvg =
    ratios.slice(0, k).reduce((a, b) => a + b, 0) / k;
  const lastAvg =
    ratios.slice(n - k, n).reduce((a, b) => a + b, 0) / k;
  if (firstAvg <= 0) return 0;
  const change = lastAvg / firstAvg - 1; // negative = dilution
  return clamp(change * EARNED_TILT, -EARNED_TILT, EARNED_TILT);
}

const dilutionFor = (rate: number): number =>
  DILUTION_LOW +
  (DILUTION_HIGH - DILUTION_LOW) * clamp(rate / 100, 0, 1);

function forecastCase(c: Case): Array<{ id: number; q: number[] }> {
  const dDays = c.horizon_minutes / 1440;
  const sqrtD = Math.sqrt(Math.max(dDays, 1 / 1440));
  const targetMs = new Date(c.target).getTime();

  const last = c.history[c.history.length - 1];
  const first = c.history[0];
  if (!last || !first) throw new Error('case history is empty');

  // Common signals across all trophies.
  const playerVol = playerVolatility(c);
  const earnedTilt = earnedRatioTilt(c);

  // Player-count dilution tilt (logit units).
  let baseTilt = 0;
  if (last.players_tracked > 0 && first.players_tracked > 0) {
    const growth = last.players_tracked / first.players_tracked;
    if (growth >= DILUTION_MIN_GROWTH) {
      baseTilt = DILUTION_TILT * Math.log(1 / growth);
    }
  }

  return c.trophies.map((t, i) => {
    const rLastRaw = last.rates?.[i];
    const rLast = rLastRaw == null || rLastRaw <= 0 ? 0.01 : rLastRaw;

    // A trophy at 100% stays at 100%: everyone tracked already has it, and the
    // page's "every rate but the one at 100% fell" on refresh confirms it does
    // not dilute.
    if (rLast >= 99.5) {
      return { id: t.id, q: [99.99, 99.99, 99.99, 99.99, 99.99] };
    }

    let lMedian: number;
    let spread: number;

    if (isFlatHistory(c, i)) {
      // History flat: page not yet refreshed. Expect dilution at target.
      lMedian =
        logit(rLast * dilutionFor(rLast)) + baseTilt + earnedTilt;
      // Absolute floor for short horizons (a refresh is a discrete event),
      // scaling with sqrt(days) for long horizons like the final's 4.5 days.
      spread = clamp(
        Math.max(SPREAD_FLOOR, SIGMA_DAILY_MIN * sqrtD) +
          playerVol * PLAYER_VOL_SPREAD,
        SPREAD_FLOOR,
        SPREAD_CAP
      );
    } else {
      const fit = slopeLogitPerHour(c, i);
      if (fit) {
        // Extrapolate trend to target (hours since first read).
        const hrsTarget = (targetMs - new Date(first.at).getTime()) / (1000 * 60 * 60);
        lMedian = fit.a + fit.b * hrsTarget + baseTilt + earnedTilt;
        // Spread from residual volatility, scaled by sqrt(horizon days),
        // plus player-count volatility.
        const sigmaDaily = clamp(fit.residStd * 24, SIGMA_DAILY_MIN, SIGMA_DAILY_MAX);
        spread = clamp(
          sigmaDaily * sqrtD + playerVol * PLAYER_VOL_SPREAD,
          SIGMA_DAILY_MIN * sqrtD,
          SPREAD_CAP
        );
      } else {
        lMedian = logit(rLast) + baseTilt + earnedTilt;
        spread = clamp(
          SPREAD_FLOOR * sqrtD + playerVol * PLAYER_VOL_SPREAD,
          SPREAD_FLOOR * sqrtD,
          SPREAD_CAP
        );
      }
    }

    const q = Z.map((z) =>
      round4(clamp(invLogit(lMedian + z * spread), 0, 100))
    );
    for (let k = 1; k < q.length; k++) {
      if (q[k] < q[k - 1]) q[k] = q[k - 1];
    }
    return { id: t.id, q };
  });
}

mkdirSync(OUTPUT_DIR, { recursive: true });
const inputs = readdirSync(INPUT_DIR)
  .filter((f) => f.endsWith('.in'))
  .sort();

for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (
    c.levels.length !== LEVELS.length ||
    c.levels.some((v, j) => v !== LEVELS[j])
  ) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}`);
  }
  const trophies = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies })}\n`);
}
