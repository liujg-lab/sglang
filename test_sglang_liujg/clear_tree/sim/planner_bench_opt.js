// Compiled (V8 JIT) port of the ClearTree planner core: mu-hat by snapshot sampling with global top-W and
// top-K insertion, and the nested subset DP. 1) checks against numpy reference cases, 2) times decisions.
"use strict";
const fs_ = require("fs");
const C = 8, KS = [4, 8, 16], KMAX = 16;

function logit(p) { p = Math.min(Math.max(p, 1e-4), 1 - 1e-4); return Math.log(p / (1 - p)); }

// Library rows are preprocessed once (logit q1 and normalised q^beta); a real planner caches this per row.
function prepPool(pool, P, beta) {
  const lq1 = new Float64Array(P), qn = new Float64Array(P * C);
  for (let r = 0; r < P; r++) {
    lq1[r] = logit(pool[r * C]);
    let s = 0;
    for (let c = 0; c < C; c++) { const w = beta === 1 ? pool[r * C + c] : Math.pow(pool[r * C + c], beta); qn[r * C + c] = w; s += w; }
    for (let c = 0; c < C; c++) qn[r * C + c] /= s;
  }
  return { lq1, qn };
}

// scratch buffers
const topS = new Float64Array(16), topP = new Float64Array(16), merged = new Float64Array(KMAX);

// mu-hat for nR requests; idx: draw indices (nR*N*nF) into pool rows; cur: per-request tree values sorted desc.
function muHat(nR, N, nF, W, nT, pool, prep, idx, fs, fp, cur, dl, a, g, out /* nR*3 */) {
  for (let r = 0; r < nR; r++) {
    // prefix sums of the current tree (top KMAX)
    let cK = [0, 0, 0], acc = 0;
    for (let k = 0; k < KMAX; k++) {
      acc += k < nT ? cur[r * nT + k] : 0;
      if (k === 3) cK[0] = acc; else if (k === 7) cK[1] = acc; else if (k === 15) cK[2] = acc;
    }
    let s4 = 0, s8 = 0, s16 = 0;
    for (let d = 0; d < N; d++) {
      // global top-W by draft score among nF*C candidates (insertion into a small sorted buffer)
      let cnt = 0;
      for (let f = 0; f < nF; f++) {
        const row = idx[(r * N + d) * nF + f];
        const fsv = fs[r * nF + f], fpv = fp[r * nF + f];
        if (fsv === 0) continue;
        const p = 1 / (1 + Math.exp(-(a + g * prep.lq1[row] + dl[r])));
        for (let c = 0; c < C; c++) {
          const sc = fsv * pool[row * C + c];
          if (cnt === W && sc <= topS[W - 1]) continue;
          const pv = fpv * p * prep.qn[row * C + c];
          let j = cnt < W ? cnt++ : W - 1;
          while (j > 0 && topS[j - 1] < sc) { topS[j] = topS[j - 1]; topP[j] = topP[j - 1]; j--; }
          topS[j] = sc; topP[j] = pv;
        }
      }
      // sort the selected pi values desc (W <= 16), then merge with the tree's values, top KMAX
      for (let i = 1; i < cnt; i++) { const v = topP[i]; let j = i; while (j > 0 && topP[j - 1] < v) { topP[j] = topP[j - 1]; j--; } topP[j] = v; }
      let ia = 0, ib = 0, sum = 0;
      for (let k = 0; k < KMAX; k++) {
        const va = ia < nT ? cur[r * nT + ia] : -1, vb = ib < cnt ? topP[ib] : -1;
        let v;
        if (va < 0 && vb < 0) v = 0; else if (va >= vb) { v = va; ia++; } else { v = vb; ib++; }
        sum += v;
        if (k === 3) s4 += sum; else if (k === 7) s8 += sum; else if (k === 15) s16 += sum;
      }
    }
    out[r * 3] = s4 / N - cK[0]; out[r * 3 + 1] = s8 / N - cK[1]; out[r * 3 + 2] = s16 / N - cK[2];
  }
}

// nested subset DP: u (n*H), el (n*H 0/1), costs by popcount; returns xs[k] arrays (Float64Array 2^n)
const POP = new Int32Array(256);
for (let s = 0; s < 256; s++) { let c = 0, t = s; while (t) { c += t & 1; t >>= 1; } POP[s] = c; }
function nestedDP(n, H, u, el, costs, theta, xsOut) {
  const M = 1 << n;
  let V = new Float64Array(M), U = new Float64Array(M);
  for (let k = H - 1; k >= 0; k--) {
    let e = 0;
    for (let i = 0; i < n; i++) if (el[i * H + k]) e |= 1 << i;
    const x = xsOut[k];
    U[0] = 0;
    for (let S = 1; S < M; S++) { const low = S & -S; U[S] = U[S ^ low] + u[(31 - Math.clz32(low)) * H + k]; }
    for (let S = 0; S < M; S++) {
      x[S] = (S & ~e) ? -1e300 : U[S] - (theta * costs[POP[S]] + 1e-9 * POP[S]) + V[S];
    }
    x[0] = 0;
    // subset max (sum over subsets, max version)
    for (let S = 0; S < M; S++) V[S] = x[S];
    for (let i = 0; i < n; i++) { const b = 1 << i; for (let S = 0; S < M; S++) if (S & b) { const o = V[S ^ b]; if (o > V[S]) V[S] = o; } }
  }
}

// ---------------- 1) correctness against numpy ----------------
const cases = JSON.parse(fs_.readFileSync(__dirname + "/cases.json"));
let worstMu = 0, worstDp = 0;
for (const cs of cases.mu) {
  const pool = Float64Array.from(cs.pool), prep = prepPool(pool, cs.P, cs.beta);
  const out = new Float64Array(cs.nR * 3);
  muHat(cs.nR, cs.N, cs.nF, cs.W, cs.nT, pool, prep, Int32Array.from(cs.idx), Float64Array.from(cs.fs),
        Float64Array.from(cs.fp), Float64Array.from(cs.cur), Float64Array.from(cs.dl), cs.a, cs.g, out);
  for (let r = 0; r < cs.nR; r++) for (let k = 0; k < 3; k++) worstMu = Math.max(worstMu, Math.abs(out[r * 3 + k] - cs.ref[k][r]));
}
for (const cs of cases.dp) {
  const xs = []; for (let k = 0; k < cs.H; k++) xs.push(new Float64Array(1 << cs.n));
  nestedDP(cs.n, cs.H, Float64Array.from(cs.u), Int32Array.from(cs.el), Float64Array.from(cs.costs), cs.theta, xs);
  for (let k = 0; k < cs.H; k++) for (let s = 0; s < (1 << cs.n); s++) {
    const a = xs[k][s], b = cs.ref[k][s];
    if (a <= -1e299 || b <= -1e299) { if (!(a <= -1e299 && b <= -1e299)) worstDp = Infinity; } else worstDp = Math.max(worstDp, Math.abs(a - b));
  }
}


// ---------------- 2) timing ----------------
let seed = 12345;
function rnd() { seed ^= seed << 13; seed ^= seed >>> 17; seed ^= seed << 5; return (seed >>> 0) / 4294967296; }
function bench(b, W, H, depth, iters, NS) {
  const N = NS, P = 512, nF = W, nT = W * depth;
  const pool = new Float64Array(P * C);
  for (let r = 0; r < P; r++) { let s = 0; const row = []; for (let c = 0; c < C; c++) { const v = Math.pow(rnd(), 2); row.push(v); s += v; }
    row.sort((x, y) => y - x); for (let c = 0; c < C; c++) pool[r * C + c] = row[c] / s * 0.9; }
  const prep = prepPool(pool, P, 1);
  const fs = new Float64Array(b * nF), fp = new Float64Array(b * nF), cur = new Float64Array(b * nT), dl = new Float64Array(b);
  for (let i = 0; i < fs.length; i++) { fs[i] = rnd() * 0.3; fp[i] = rnd() * 0.3; }
  for (let r = 0; r < b; r++) { const v = []; for (let k = 0; k < nT; k++) v.push(rnd() * 0.9); v.sort((x, y) => y - x); for (let k = 0; k < nT; k++) cur[r * nT + k] = v[k]; }
  const idx = new Int32Array(b * N * nF), out = new Float64Array(b * 3);
  const u = new Float64Array(b * H), el = new Int32Array(b * H).fill(1), costs = new Float64Array(b + 1);
  for (let k = 1; k <= b; k++) costs[k] = 1.5 + 0.02 * k * W;
  const xs = []; for (let k = 0; k < H; k++) xs.push(new Float64Array(1 << b));
  const tFirst = [], tNext = [];
  for (let it = 0; it < iters; it++) {
    for (let i = 0; i < idx.length; i++) idx[i] = (rnd() * P) | 0;
    // first decision: mu for all requests (all K at once) + one DP per K
    let t0 = process.hrtime.bigint();
    muHat(b, N, nF, W, nT, pool, prep, idx, fs, fp, cur, dl, 1.5, 1.0, out);
    for (let q = 0; q < 3; q++) {
      for (let i = 0; i < b; i++) for (let k = 0; k < H; k++) u[i * H + k] = out[i * 3 + q] * Math.pow(0.7, k);
      nestedDP(b, H, u, el, costs, 0.1, xs);
    }
    let t1 = process.hrtime.bigint();
    // later decision: mu for active requests + one DP
    muHat(b, N, nF, W, nT, pool, prep, idx, fs, fp, cur, dl, 1.5, 1.0, out);
    for (let i = 0; i < b; i++) for (let k = 0; k < H; k++) u[i * H + k] = out[i * 3] * Math.pow(0.7, k);
    nestedDP(b, H, u, el, costs, 0.1, xs);
    let t2 = process.hrtime.bigint();
    if (it >= iters / 5) { tFirst.push(Number(t1 - t0) / 1e3); tNext.push(Number(t2 - t1) / 1e3); }
  }
  const q = (a, p) => { const s = a.slice().sort((x, y) => x - y); return s[Math.floor(p * (s.length - 1))]; };
  const mean = a => a.reduce((x, y) => x + y, 0) / a.length;
  console.log(`b=${b} W=${W} N_snap=${N} H=${H}: first decision (3 K) mean ${mean(tFirst).toFixed(1)} us p99 ${q(tFirst, 0.99).toFixed(1)} us | ` +
              `later decision mean ${mean(tNext).toFixed(1)} us p99 ${q(tNext, 0.99).toFixed(1)} us`);
}
for (const W of [4, 8]) for (const [NS, H] of [[16, 9], [8, 9], [16, 5], [8, 5]]) bench(8, W, H, 3, 5000, NS);
