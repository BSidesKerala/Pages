/*
 * Cyberpunk Kathakali character animation engine.
 *
 * The source illustration is split into layers (see assets/layers). Each layer is a textured mesh
 * whose vertices are bound to bones of a 2D skeleton (assets/rig.json) and deformed with linear
 * blend skinning. Animation controllers drive the bones; the background plate never moves.
 *
 * Controllers: typing schedule, finger articulation, arm IK (shoulder, elbow, wrist), head,
 * eye blinks, breathing, scarf cloth chains (particle physics), cybernetic lights.
 *
 * Rendering uses WebGL on the page canvas, with a Canvas 2D fallback.
 */
(function () {
  'use strict';

  const DATA = window.KATHAKALI_DATA;
  if (!DATA) { console.error('kathakali-data.js missing'); return; }
  const W = DATA.layers._size[0];
  const H = DATA.layers._size[1];
  const LOOP = 8;                // seconds in one animation cycle
  const TAU = Math.PI * 2;
  const DEG = Math.PI / 180;
  const ASSET_BASE = (document.currentScript && document.currentScript.dataset.assets) || 'assets/';

  // ------------------------------------------------------------------ math helpers
  const clamp = (x, a, b) => (x < a ? a : x > b ? b : x);
  const smooth = (e0, e1, x) => { const t = clamp((x - e0) / (e1 - e0), 0, 1); return t * t * (3 - 2 * t); };
  const gauss = (x, mu, s) => { const d = (x - mu) / s; return Math.exp(-d * d); };
  const wrapDelta = (t, t0) => { let d = t - t0; d -= Math.round(d / LOOP) * LOOP; return d; };
  const harm = (k, tau, phase) => Math.sin(TAU * k * tau / LOOP + phase); // k cycles per loop

  // affine matrix [a, b, c, d, e, f]: x' = a x + c y + e, y' = b x + d y + f
  const mat = () => new Float64Array([1, 0, 0, 1, 0, 0]);
  function mul(o, m, n) {
    const a = m[0] * n[0] + m[2] * n[1], b = m[1] * n[0] + m[3] * n[1];
    const c = m[0] * n[2] + m[2] * n[3], d = m[1] * n[2] + m[3] * n[3];
    const e = m[0] * n[4] + m[2] * n[5] + m[4], f = m[1] * n[4] + m[3] * n[5] + m[5];
    o[0] = a; o[1] = b; o[2] = c; o[3] = d; o[4] = e; o[5] = f; return o;
  }
  function setTRS(o, hx, hy, ox, oy, rot, sx, sy) {
    // T(h) T(o) R(rot) S(sx, sy) T(-h)
    const cs = Math.cos(rot), sn = Math.sin(rot);
    o[0] = cs * sx; o[1] = sn * sx; o[2] = -sn * sy; o[3] = cs * sy;
    o[4] = hx + ox - (o[0] * hx + o[2] * hy);
    o[5] = hy + oy - (o[1] * hx + o[3] * hy);
    return o;
  }
  const apX = (m, x, y) => m[0] * x + m[2] * y + m[4];
  const apY = (m, x, y) => m[1] * x + m[3] * y + m[5];
  const matAngle = (m) => Math.atan2(m[1], m[0]);
  function segDist(px, py, ax, ay, bx, by) {
    const vx = bx - ax, vy = by - ay, wx = px - ax, wy = py - ay;
    const L = vx * vx + vy * vy;
    const t = L > 0 ? clamp((wx * vx + wy * vy) / L, 0, 1) : 0;
    const dx = wx - t * vx, dy = wy - t * vy;
    return Math.sqrt(dx * dx + dy * dy);
  }
  function polySignedDist(px, py, poly) {
    let inside = false, md = 1e9;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      const [xi, yi] = poly[i], [xj, yj] = poly[j];
      if ((yi > py) !== (yj > py) && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi) inside = !inside;
      md = Math.min(md, segDist(px, py, xi, yi, xj, yj));
    }
    return inside ? md : -md;
  }
  // Deterministic PRNG so the typing pattern is identical every loop and every visit.
  function rng(seed) {
    let s = seed >>> 0;
    return () => { s = (s + 0x6d2b79f5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  }

  // ------------------------------------------------------------------ skeleton
  class Bone {
    constructor(def, parent) {
      this.name = def.name; this.parent = parent; this.group = def.group || null;
      this.inherit = def.inherit !== false;
      this.hx = def.head[0]; this.hy = def.head[1]; this.tx = def.tail[0]; this.ty = def.tail[1];
      this.restAngle = Math.atan2(this.ty - this.hy, this.tx - this.hx);
      this.len = Math.hypot(this.tx - this.hx, this.ty - this.hy);
      this.local = mat(); this.world = mat(); this.override = null;
      this.reset();
    }
    reset() { this.rot = 0; this.ox = 0; this.oy = 0; this.sx = 1; this.sy = 1; }
  }

  class Skeleton {
    constructor(defs) {
      this.bones = []; this.map = {};
      for (const d of defs) {
        const b = new Bone(d, d.parent ? this.map[d.parent] : null);
        this.bones.push(b); this.map[d.name] = b;
      }
      this._tmp = mat(); this._tmp2 = mat();
    }
    get(n) { return this.map[n]; }
    reset() { for (const b of this.bones) { b.reset(); b.override = null; } }
    update() {
      for (const b of this.bones) this.updateBone(b);
    }
    updateBone(b) {
      if (b.override) { b.world.set(b.override); return; }
      setTRS(b.local, b.hx, b.hy, b.ox, b.oy, b.rot, b.sx, b.sy);
      const p = b.parent;
      if (!p) { b.world.set(b.local); return; }
      if (b.inherit) { mul(b.world, p.world, b.local); return; }
      // follow parent position and rotation, but not its scale
      const px = apX(p.world, b.hx, b.hy), py = apY(p.world, b.hx, b.hy);
      const th = matAngle(p.world);
      setTRS(this._tmp, b.hx, b.hy, px - b.hx, py - b.hy, th, 1, 1);
      mul(b.world, this._tmp, b.local);
    }
    worldAngle(b) { return matAngle(b.world); }
    headPos(b) { return [apX(b.world, b.hx, b.hy), apY(b.world, b.hx, b.hy)]; }
    tailPos(b) { return [apX(b.world, b.tx, b.ty), apY(b.world, b.tx, b.ty)]; }
  }

  // ------------------------------------------------------------------ scarf cloth chains
  class ClothChain {
    constructor(name, def, skel) {
      this.name = name; this.anchor = skel.get(def.anchor);
      this.rest = def.points.map((p) => [p[0], p[1]]);
      this.n = this.rest.length;
      this.wind = def.wind; this.phase = def.phase * TAU;
      this.len = []; this.restAng = []; this.perp = [];
      for (let i = 0; i < this.n - 1; i++) {
        const [ax, ay] = this.rest[i], [bx, by] = this.rest[i + 1];
        this.len.push(Math.hypot(bx - ax, by - ay));
        const ang = Math.atan2(by - ay, bx - ax);
        this.restAng.push(ang);
        this.perp.push([-Math.sin(ang), Math.cos(ang)]);
      }
      this.total = this.len.reduce((a, b) => a + b, 0);
      this.s = [0]; for (let i = 0; i < this.n - 1; i++) this.s.push(this.s[i] + this.len[i] / this.total);
      this.x = this.rest.map((p) => p.slice()); this.px = this.rest.map((p) => p.slice());
      // bones: a stub on the root side bound to the anchor, then one bone per segment
      this.bones = [];
      const [r0x, r0y] = this.rest[0], [r1x, r1y] = this.rest[1];
      const ux = (r0x - r1x) / this.len[0], uy = (r0y - r1y) / this.len[0];
      this.stub = { hx: r0x + ux * 70, hy: r0y + uy * 70, tx: r0x, ty: r0y };
      for (let i = 0; i < this.n - 1; i++) {
        const [hx, hy] = this.rest[i], [tx, ty] = this.rest[i + 1];
        this.bones.push({ name: `${name}_${i}`, hx, hy, tx, ty, world: mat() });
      }
    }
    target(i) { const m = this.anchor.world, [x, y] = this.rest[i]; return [apX(m, x, y), apY(m, x, y)]; }
    step(dt, time, gust) {
      const tau = ((time % LOOP) + LOOP) % LOOP;
      const n = this.n;
      const [t0x, t0y] = this.target(0);
      this.x[0][0] = t0x; this.x[0][1] = t0y; this.px[0][0] = t0x; this.px[0][1] = t0y;
      const damp = 0.035;
      for (let i = 1; i < n; i++) {
        const s = this.s[i];
        const [tx, ty] = this.target(i);
        const k = 70 * Math.pow(1 - s, 1.6) + 7;             // shape memory: stiff root, loose tip
        const seg = Math.min(i - 1, n - 2);
        const [qx, qy] = this.perp[seg];
        const ph = this.phase - 2.4 * s;                         // wave travels from root to tip
        const flutter = 0.62 * Math.sin(TAU * 3 * tau / LOOP + ph) + 0.30 * Math.sin(TAU * 5 * tau / LOOP + 1.7 * ph + 0.4)
          + 0.16 * Math.sin(TAU * 9 * tau / LOOP + 2.3 * ph + 1.1) + 0.08 * Math.sin(TAU * 14 * tau / LOOP + 3.1 * ph);
        const g = gust * (0.78 + 0.22 * Math.sin(TAU * tau / LOOP + this.phase * 0.5)) * Math.pow(s, 0.9);
        const along = 0.35 + 0.25 * Math.sin(TAU * 2 * tau / LOOP + this.phase);
        const ax = k * (tx - this.x[i][0]) + g * (qx * flutter * 520 + this.wind[0] * along * 120);
        const ay = k * (ty - this.x[i][1]) + g * (qy * flutter * 520 + this.wind[1] * along * 120);
        const cx = this.x[i][0], cy = this.x[i][1];
        this.x[i][0] = cx + (cx - this.px[i][0]) * (1 - damp) + ax * dt * dt;
        this.x[i][1] = cy + (cy - this.px[i][1]) * (1 - damp) + ay * dt * dt;
        this.px[i][0] = cx; this.px[i][1] = cy;
      }
      for (let it = 0; it < 3; it++) {
        for (let i = 1; i < n; i++) {
          const a = this.x[i - 1], b = this.x[i];
          const dx = b[0] - a[0], dy = b[1] - a[1];
          const d = Math.hypot(dx, dy) || 1e-6;
          const L = this.len[i - 1];
          const f = (d - L) / d;
          if (i === 1) { b[0] -= dx * f; b[1] -= dy * f; }
          else { a[0] += dx * f * 0.25; a[1] += dy * f * 0.25; b[0] -= dx * f * 0.75; b[1] -= dy * f * 0.75; }
        }
      }
    }
    snap() { for (let i = 0; i < this.n; i++) { const t = this.target(i); this.x[i] = t.slice(); this.px[i] = t.slice(); } }
    updateBones() {
      for (let i = 0; i < this.n - 1; i++) {
        const b = this.bones[i], a = this.x[i], c = this.x[i + 1];
        const ang = Math.atan2(c[1] - a[1], c[0] - a[0]) - this.restAng[i];
        setTRS(b.world, b.hx, b.hy, a[0] - b.hx, a[1] - b.hy, ang, 1, 1);
      }
    }
  }

  // ------------------------------------------------------------------ meshes and skin weights
  function decodeCover(c) {
    const raw = atob(c.bits); const bytes = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
    return { gw: c.gw, gh: c.gh, test(ix, iy) { if (ix < 0 || iy < 0 || ix >= c.gw || iy >= c.gh) return false; const k = iy * c.gw + ix; return (bytes[k >> 3] >> (7 - (k & 7))) & 1; } };
  }

  // Build a grid mesh over a layer's rectangle, dropping cells with no opaque pixels.
  function buildMesh(rect, cell, cover) {
    const nx = Math.max(1, Math.ceil(rect.w / cell)), ny = Math.max(1, Math.ceil(rect.h / cell));
    const cw = rect.w / nx, ch = rect.h / ny;
    const B = DATA.block;
    const used = new Uint8Array(nx * ny);
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      if (!cover) { used[j * nx + i] = 1; continue; }
      const x0 = Math.floor((i * cw - 1) / B), x1 = Math.floor(((i + 1) * cw + 1) / B);
      const y0 = Math.floor((j * ch - 1) / B), y1 = Math.floor(((j + 1) * ch + 1) / B);
      let hit = 0;
      for (let y = y0; y <= y1 && !hit; y++) for (let x = x0; x <= x1; x++) if (cover.test(x, y)) { hit = 1; break; }
      used[j * nx + i] = hit;
    }
    const vid = new Int32Array((nx + 1) * (ny + 1)).fill(-1);
    const pos = [], uv = [], idx = [];
    const vert = (i, j) => {
      const k = j * (nx + 1) + i;
      if (vid[k] < 0) { vid[k] = pos.length / 2; pos.push(rect.x + i * cw, rect.y + j * ch); uv.push(i / nx, j / ny); }
      return vid[k];
    };
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      if (!used[j * nx + i]) continue;
      const a = vert(i, j), b = vert(i + 1, j), c = vert(i, j + 1), d = vert(i + 1, j + 1);
      idx.push(a, b, c, b, d, c);
    }
    return { rest: new Float32Array(pos), uv: new Float32Array(uv), idx: new Uint16Array(idx), out: new Float32Array(pos.length) };
  }

  // Skin influence: list of {world, hx, hy, tx, ty, group, region}
  function computeWeights(rest, infl, opts) {
    const nv = rest.length / 2, K = 4;
    const bi = new Uint16Array(nv * K), bw = new Float32Array(nv * K);
    const p = opts.power || 3, eps = opts.eps || 8, soft = opts.groupSoft || 4;
    const w = new Float64Array(infl.length), dist = new Float64Array(infl.length);
    for (let v = 0; v < nv; v++) {
      const x = rest[v * 2], y = rest[v * 2 + 1];
      const gmin = {};
      for (let k = 0; k < infl.length; k++) {
        const b = infl[k];
        const d = segDist(x, y, b.hx, b.hy, b.tx, b.ty);
        dist[k] = d;
        if (b.group) gmin[b.group] = Math.min(gmin[b.group] === undefined ? 1e9 : gmin[b.group], d);
      }
      let best = 1e9; for (const g in gmin) best = Math.min(best, gmin[g]);
      let sum = 0;
      for (let k = 0; k < infl.length; k++) {
        const b = infl[k];
        let wk = Math.pow((b.reach || 40) / (dist[k] + eps), p);
        if (b.group) { const dg = gmin[b.group] - best; wk *= Math.exp(-(dg / soft) * (dg / soft)); }
        if (b.region) { const sd = polySignedDist(x, y, b.region.poly); wk *= smooth(-b.region.feather, b.region.feather * 0.5, sd); }
        w[k] = wk; sum += wk;
      }
      // top-K
      const order = Array.from(w.keys()).sort((a, b) => w[b] - w[a]).slice(0, K);
      let s2 = 0; for (const k of order) s2 += w[k];
      if (s2 <= 0) { order[0] = 0; w[0] = 1; s2 = 1; }
      for (let j = 0; j < K; j++) {
        const k = order[j];
        bi[v * K + j] = k === undefined ? 0 : k;
        bw[v * K + j] = k === undefined ? 0 : w[k] / s2;
      }
    }
    return { bi, bw, K };
  }

  function skinInto(out, rest, skin, infl) {
    const { bi, bw, K } = skin; const nv = rest.length / 2;
    for (let v = 0; v < nv; v++) {
      const x = rest[v * 2], y = rest[v * 2 + 1];
      let ox = 0, oy = 0;
      for (let j = 0; j < K; j++) {
        const wt = bw[v * K + j]; if (wt === 0) continue;
        const m = infl[bi[v * K + j]].world;
        ox += wt * (m[0] * x + m[2] * y + m[4]);
        oy += wt * (m[1] * x + m[3] * y + m[5]);
      }
      out[v * 2] = ox; out[v * 2 + 1] = oy;
    }
  }

  // ------------------------------------------------------------------ the character
  class Character {
    constructor() {
      const rig = DATA.rig;
      this.skel = new Skeleton(rig.bones);
      this.chains = {};
      for (const [k, def] of Object.entries(rig.chains)) if (k !== 'about') this.chains[k] = new ClothChain(k, def, this.skel);
      this.layers = [];
      this.time = 0;
      this.buildTyping();
      this.fingerSign = {};
      for (const f of ['f1', 'f2', 'f3', 'f4', 'l2', 'l3', 'l4']) {
        const a = this.skel.get(f + 'a'), b = this.skel.get(f + 'b');
        let d = b.restAngle - a.restAngle; d = Math.atan2(Math.sin(d), Math.cos(d));
        this.fingerSign[f] = Math.sign(d) || 1;
      }
      this.fingerSign.l1 = this.fingerSign.l2; this.fingerSign.l5 = this.fingerSign.l4;
    }

    // ---- typing schedule: deterministic keystrokes over the 8 s loop
    intensity(tau) {
      // 0-2 s typing, 2-3 s pause with head tilt, 3-4 s blink and resume, 4-6 s fast, 6-8 s steady
      const keys = [[0, 1], [1.85, 1], [2.2, 0], [3.42, 0], [3.95, 1], [4.25, 1.45], [5.75, 1.45], [6.2, 1], [8, 1]];
      for (let i = 0; i < keys.length - 1; i++) {
        const [t0, v0] = keys[i], [t1, v1] = keys[i + 1];
        if (tau >= t0 && tau <= t1) { const u = smooth(t0, t1, tau); return v0 + (v1 - v0) * u; }
      }
      return 1;
    }
    buildTyping() {
      const R = rng(1977);
      const fingers = ['f1', 'f2', 'f3', 'f4'];
      const wts = [0.34, 0.31, 0.26, 0.09];
      const strikes = [];
      let t = 0.09, last = -1;
      while (t < 7.86) {
        const I = this.intensity(t);
        if (I < 0.1) { t += 0.04; continue; }
        let f;
        do { let r = R(), acc = 0; f = 0; for (let i = 0; i < 4; i++) { acc += wts[i]; if (r <= acc) { f = i; break; } } }
        while (f === last && R() > 0.12);
        last = f;
        strikes.push({ t, f: fingers[f], amp: 0.65 + 0.35 * R() });
        let gap = (0.12 + 0.13 * R()) / Math.max(I, 0.35);
        if (R() < 0.08) gap += 0.22 + 0.2 * R();       // short hesitation
        t += gap;
      }
      this.strikes = strikes;
      this.strikesBy = {}; for (const f of fingers) this.strikesBy[f] = strikes.filter((s) => s.f === f);
    }
    fingerFlex(f, tau) {
      let v = 0;
      for (const s of this.strikesBy[f]) {
        const d = wrapDelta(tau, s.t);
        if (d < -0.25 || d > 0.25) continue;
        v += s.amp * (-0.6 * gauss(d, -0.075, 0.045) + 1.0 * gauss(d, 0.0, 0.032) - 0.18 * gauss(d, 0.075, 0.04));
      }
      return v;
    }
    pressEnv(tau, sigma = 0.05) {
      let v = 0;
      for (const s of this.strikes) { const d = wrapDelta(tau, s.t); if (Math.abs(d) < 0.3) v += s.amp * gauss(d, 0.01, sigma); }
      return v;
    }
    activity(tau) { // smoothed keystroke density, 0..~1.5
      let v = 0;
      for (const s of this.strikes) { const d = wrapDelta(tau, s.t); if (Math.abs(d) < 0.6) v += gauss(d, 0, 0.25); }
      return v / 2.2;
    }
    blink(tau, offset) {
      const one = (t0, depth) => {
        const d = wrapDelta(tau, t0 + offset);
        if (d < -0.01 || d > 0.32) return 0;
        if (d < 0.065) return depth * smooth(0, 0.065, d);           // close fast
        if (d < 0.11) return depth;                                   // held shut
        return depth * (1 - smooth(0.11, 0.3, d));                    // open slower
      };
      return Math.max(one(3.3, 1), one(6.85, 0.92), one(0.62, 0.0));
    }
    breath(tau) { // 2 breaths per loop; inhale shorter than exhale
      const ph = (tau % 4) / 4;
      return ph < 0.42 ? smooth(0, 0.42, ph) : 1 - smooth(0.42, 1, ph);
    }
    tilt(tau) { return smooth(2.0, 2.6, tau) * (1 - smooth(3.45, 4.4, tau)); }

    // ---- pose: drive every bone for loop time tau
    pose(tau, opts) {
      const S = this.skel; S.reset();
      const I = this.intensity(tau), Ic = Math.min(I, 1), P = 1 - clamp(I, 0, 1);
      const br = this.breath(tau), tilt = this.tilt(tau);
      const press = this.pressEnv(tau), act = this.activity(tau);
      this.I = I; this.press = press;

      // breathing: chest expands, shoulders ride up
      const chest = S.get('chest');
      chest.sx = 1 + 0.0045 * br; chest.sy = 1 + 0.0075 * br;
      S.get('pelvis').oy = 0;
      const shR = S.get('shoulder_r'), shL = S.get('shoulder_l');
      shR.oy = -0.7 * br - 0.6 * clamp(I - 1, 0, 1) + 0.6 * P;
      shR.rot = (0.25 * harm(1, tau, 0.9) + 0.35 * clamp(I - 1, 0, 1) * harm(4, tau, 0.2)) * DEG;
      shL.oy = -0.5 * br; shL.rot = 0.12 * harm(1, tau, 2.0) * DEG;

      // head: idle sway, look toward the screen during the pause, small nods while typing
      const hr = (0.4 * harm(1, tau, 0.6) + 0.22 * harm(3, tau, 2.1) + 0.1 * harm(7, tau, 0.4)) * DEG
        - 2.2 * DEG * tilt + 0.25 * DEG * act * harm(6, tau, 1.0);
      const neck = S.get('neck'), head = S.get('head');
      neck.rot = 0.3 * hr; head.rot = 0.7 * hr;
      head.ox = -1.6 * tilt + 0.4 * harm(1, tau, 1.9);
      head.oy = 2.0 * tilt + (0.7 * harm(2, tau, 0.8) + 0.35 * harm(5, tau, 2.4)) * Ic + 0.6 * act * gauss(Math.sin(TAU * 3 * tau / LOOP), 1, 0.6);
      neck.oy = 0.3 * br;

      // support hand and laptop: steady grip, tiny shifts and key-strike recoil
      const fl = S.get('forearm_l'), hl = S.get('hand_l'), lap = S.get('laptop');
      fl.rot = (0.12 * harm(1, tau, 0.3)) * DEG;
      hl.rot = (0.18 * harm(2, tau, 1.2) - 0.1 * harm(1, tau, 0.3)) * DEG;
      lap.rot = (0.08 * harm(3, tau, 0.5)) * DEG; lap.oy = 0.35 * press;
      const grip = 0.12 * harm(2, tau, 0.4) + 0.08 * harm(5, tau, 1.7) + 0.25 * gauss(wrapDelta(tau, 5.1), 0, 0.35);
      const lf = { l1: 0, l2: 0.3, l3: 0.6, l4: 0.9, l5: 1.2 };
      for (const [f, ph] of Object.entries(lf)) {
        const g = (grip + 0.06 * harm(3, tau, ph * 2)) * this.fingerSign[f];
        if (f === 'l1' || f === 'l5') S.get(f).rot = 4 * g * DEG;
        else { S.get(f + 'a').rot = 3 * g * DEG; S.get(f + 'b').rot = 4 * g * DEG; }
      }

      // waist cloth sways from the hips
      S.get('cloth_a').rot = (0.9 * harm(3, tau, 0.5) + 0.5 * harm(5, tau, 2.0)) * DEG;
      S.get('cloth_b').rot = (1.4 * harm(3, tau, -0.4) + 0.6 * harm(7, tau, 1.0)) * DEG;

      // update the upper body first so the arm IK has current shoulder and laptop positions
      for (const n of ['world', 'pelvis', 'chest', 'neck', 'head', 'shoulder_r', 'shoulder_l', 'upperarm_l', 'forearm_l', 'hand_l', 'laptop']) S.updateBone(S.get(n));

      // typing arm: wrist target rides on the laptop, IK solves shoulder and elbow
      const ua = S.get('upperarm_r'), fa = S.get('forearm_r'), hand = S.get('hand_r');
      const amp = 0.35 + 0.65 * Ic + 0.35 * clamp(I - 1, 0, 1);
      let wx = fa.tx + amp * (4.2 * harm(1, tau, 0.4) + 2.2 * harm(3, tau, 1.3) + 0.9 * harm(7, tau, 2.2));
      let wy = fa.ty + amp * (1.3 * harm(2, tau, 0.1) + 0.6 * harm(5, tau, 0.9)) - 2.0 * P + 0.8 * this.pressEnv(tau, 0.06);
      const lw = lap.world; const tx = apX(lw, wx, wy), ty = apY(lw, wx, wy);
      this.solveArm(ua, fa, tx, ty);
      S.updateBone(ua); S.updateBone(fa);
      // keep the hand level with the keyboard, roll toward the striking finger, lift during the pause
      const faDelta = S.worldAngle(fa) - matAngle(lw) - (fa.restAngle - Math.atan2(fa.ty - fa.hy, fa.tx - fa.hx)) - 0;
      let roll = 0;
      for (const [f, dir] of [['f1', 1], ['f2', 0.4], ['f3', -0.5], ['f4', -1]]) {
        for (const s of this.strikesBy[f]) { const d = wrapDelta(tau, s.t); if (Math.abs(d) < 0.25) roll += dir * s.amp * gauss(d, 0, 0.07); }
      }
      hand.rot = -(S.worldAngle(fa) - matAngle(lw)) + (0.7 * roll + 1.1 * P + 0.25 * harm(2, tau, 0.7)) * DEG;
      void faDelta;

      // fingers: hover, lift, strike, settle
      const hover = -0.2 * Ic - 0.4 * P + 0.05 * harm(4, tau, 1.0);
      const fphase = { f1: 0.0, f2: 0.9, f3: 1.8, f4: 2.7 };
      this.flex = {};
      for (const f of ['f1', 'f2', 'f3', 'f4']) {
        const fx = hover + 0.06 * harm(5, tau, fphase[f]) + this.fingerFlex(f, tau);
        this.flex[f] = fx;
        const s = this.fingerSign[f] * fx;
        S.get(f + 'a').rot = 6.5 * s * DEG;
        S.get(f + 'b').rot = 7.5 * s * DEG;
        const c = S.get(f + 'c'); if (c) c.rot = 4.5 * s * DEG;
      }
      S.update();
    }

    solveArm(ua, fa, tx, ty) {
      // two-bone IK in world space; the elbow keeps its rest-side bend
      const S = this.skel;
      const pw = ua.parent.world;
      const sx = apX(pw, ua.hx, ua.hy), sy = apY(pw, ua.hx, ua.hy);
      const L1 = ua.len, L2 = fa.len;
      let dx = tx - sx, dy = ty - sy; let d = Math.hypot(dx, dy);
      d = clamp(d, Math.abs(L1 - L2) + 1, L1 + L2 - 0.5);
      const base = Math.atan2(dy, dx);
      const cosA = clamp((L1 * L1 + d * d - L2 * L2) / (2 * L1 * d), -1, 1);
      const A = Math.acos(cosA);
      // rest bend side
      const rdx = fa.tx - ua.hx, rdy = fa.ty - ua.hy;
      const restBase = Math.atan2(rdy, rdx);
      let side = Math.sign(Math.atan2(Math.sin(ua.restAngle - restBase), Math.cos(ua.restAngle - restBase))) || 1;
      const upperAng = base + side * A;
      const ex = sx + L1 * Math.cos(upperAng), ey = sy + L1 * Math.sin(upperAng);
      const foreAng = Math.atan2(ty - ey, tx - ex);
      const parentAng = matAngle(pw);
      ua.rot = upperAng - ua.restAngle - parentAng;
      fa.rot = (foreAng - fa.restAngle) - (upperAng - ua.restAngle);
      void S;
    }
  }

  // ------------------------------------------------------------------ layer setup
  function influencesFor(char, skinDef, layerName) {
    const S = char.skel;
    if (skinDef.mode === 'chains') {
      const infl = [];
      for (const cn of skinDef.chains) {
        const ch = char.chains[cn];
        infl.push({ world: ch.anchor.world, hx: ch.stub.hx, hy: ch.stub.hy, tx: ch.stub.tx, ty: ch.stub.ty, group: cn, reach: 40 });
        ch.bones.forEach((b) => infl.push({ world: b.world, hx: b.hx, hy: b.hy, tx: b.tx, ty: b.ty, group: cn, reach: 40 }));
      }
      return infl;
    }
    if (skinDef.mode === 'auto') {
      return skinDef.bones.map((n) => {
        const b = S.get(n);
        const r = skinDef.regions && skinDef.regions[n];
        return { world: b.world, hx: b.hx, hy: b.hy, tx: b.tx, ty: b.ty, group: b.group, region: r || null, reach: 40, bone: b };
      });
    }
    if (skinDef.mode === 'rigid') { const b = S.get(skinDef.bone); return [{ world: b.world, hx: b.hx, hy: b.hy, tx: b.tx, ty: b.ty }]; }
    if (skinDef.mode === 'gradient') {
      return ['chest', 'neck', 'head'].map((n) => { const b = S.get(n); return { world: b.world, hx: b.hx, hy: b.hy, tx: b.tx, ty: b.ty }; });
    }
    throw new Error('unknown skin mode ' + layerName);
  }

  function weightsFor(skinDef, rest, infl) {
    if (skinDef.mode === 'rigid') {
      const nv = rest.length / 2; const bi = new Uint16Array(nv * 4), bw = new Float32Array(nv * 4);
      for (let v = 0; v < nv; v++) bw[v * 4] = 1; return { bi, bw, K: 4 };
    }
    if (skinDef.mode === 'gradient') {
      const nv = rest.length / 2; const bi = new Uint16Array(nv * 4), bw = new Float32Array(nv * 4);
      const y0 = skinDef.from.at[1], y1 = skinDef.to.at[1], mx = skinDef.to.max, nw = skinDef.extra.w;
      for (let v = 0; v < nv; v++) {
        const x = rest[v * 2], y = rest[v * 2 + 1];
        let r = clamp((y0 - y) / (y0 - y1), 0, 1) * clamp((1030 - x) / 90, 0, 1);
        r = r * r * (3 - 2 * r);
        bi[v * 4] = 0; bi[v * 4 + 1] = 1; bi[v * 4 + 2] = 2;
        bw[v * 4 + 2] = mx * r; bw[v * 4 + 1] = nw * r; bw[v * 4] = 1 - (mx + nw) * r;
      }
      return { bi, bw, K: 4 };
    }
    const opts = skinDef.mode === 'chains' ? { power: 4, eps: 6, groupSoft: 5 } : { power: skinDef.power || 3, eps: 8, groupSoft: 3 };
    return computeWeights(rest, infl, opts);
  }

  function setupLayers(char, coarse) {
    const L = DATA.layers, skins = DATA.rig.skins;
    const covers = {};
    for (const k in DATA.cover) covers[k] = decodeCover(DATA.cover[k]);
    const scale = coarse ? 2.6 : 1;
    const out = [];
    const mk = (name, kind, extra) => {
      const rect = L[name];
      return Object.assign({ name, kind, rect, img: null, alpha: 1, mode: 0 }, extra || {});
    };
    out.push(mk('plate', 'static', { mesh: { rest: new Float32Array([0, 0, W, 0, 0, H, W, H]), uv: new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]), idx: new Uint16Array([0, 1, 2, 1, 3, 2]), out: new Float32Array([0, 0, W, 0, 0, H, W, H]) }, file: 'plate.png' }));
    const skinLayer = (name, skinName, cell) => {
      const def = skins[skinName];
      const c = def.mode === 'rigid' ? 1e6 : Math.round((cell || def.cell || 12) * scale);
      const mesh = def.mode === 'rigid'
        ? (() => { const r = L[name]; const p = new Float32Array([r.x, r.y, r.x + r.w, r.y, r.x, r.y + r.h, r.x + r.w, r.y + r.h]); return { rest: p, uv: new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]), idx: new Uint16Array([0, 1, 2, 1, 3, 2]), out: new Float32Array(p) }; })()
        : buildMesh(L[name], c, covers[name]);
      const infl = influencesFor(char, def, skinName);
      const skin = weightsFor(def, mesh.rest, infl);
      return { mesh, infl, skin };
    };
    for (const name of L._order) {
      out.push(mk(name, 'skinned', Object.assign(skinLayer(name, name), { file: name + '.png' })));
      if (name === 'body') {
        for (const b of L._bars) out.push(mk(b, 'overlay', Object.assign(skinLayer(b, 'body', 40), { file: b + '.png', fx: 'bar' })));
        for (const l of L._leds) out.push(mk(l, 'overlay', Object.assign(skinLayer(l, 'body', 40), { file: l + '.png', fx: 'led' })));
        out.push(mk('sheen_body', 'overlay', Object.assign(skinLayer('sheen_body', 'body', 24), { file: 'sheen_body.png', fx: 'sheen', mode: 2 })));
      }
      if (name === 'head') {
        for (const e of L._eyes) out.push(mk(e, 'overlay', Object.assign(skinLayer(e, 'head'), { file: e + '.png', fx: 'eye', mode: 1 })));
      }
      if (name === 'laptop') out.push({ name: 'keyglow', kind: 'sprite', fx: 'keyglow', mode: 3, rect: { x: 0, y: 0, w: 64, h: 64 }, sprites: [] });
    }
    return out;
  }

  // ------------------------------------------------------------------ renderers
  const VS = `
attribute vec2 aPos; attribute vec2 aUV; uniform vec2 uInv; varying vec2 vUV;
void main(){ vUV = aUV; vec2 p = aPos * uInv; gl_Position = vec4(p.x * 2.0 - 1.0, 1.0 - p.y * 2.0, 0.0, 1.0); }`;
  const FS = `
precision mediump float;
uniform sampler2D uTex; uniform float uAlpha; uniform int uMode; uniform vec4 uP; uniform vec4 uRect; uniform vec3 uTint;
varying vec2 vUV;
void main(){
  vec4 c = texture2D(uTex, vUV);
  if (uMode == 1) {
    // eyelids: upper lid sweeps down to the eye's centre line, lower lid rises a little
    float lid = uP.x; float cu = uP.y; float cv = uP.z;
    float dx = (vUV.x - cu) * 1.6;
    float arch = 0.10 * (1.0 - lid) * (1.0 - dx * dx);
    float top = mix(-0.08, cv + 0.015, lid) - arch;
    float bot = mix(1.08, cv + 0.025, lid * 0.92);
    float vis = smoothstep(top, top + 0.06, vUV.y) * (1.0 - smoothstep(bot - 0.05, bot, vUV.y));
    vis *= 1.0 - 0.35 * lid;
    gl_FragColor = c * vis * uAlpha;
  } else if (uMode == 2) {
    // moving specular sheen over armor plates (additive)
    vec2 p = uRect.xy + vUV * uRect.zw;
    float d = dot(p, normalize(uP.xy)) - uP.z;
    float band = exp(-(d * d) / (uP.w * uP.w));
    gl_FragColor = vec4(uTint * c.a * band * uAlpha, 0.0);
  } else if (uMode == 3) {
    gl_FragColor = vec4(uTint * c.a * uAlpha, 0.0);
  } else {
    gl_FragColor = c * uAlpha;
  }
}`;

  class GLRenderer {
    constructor(canvas) {
      const gl = canvas.getContext('webgl', { alpha: false, antialias: true, premultipliedAlpha: true, preserveDrawingBuffer: false, powerPreference: 'low-power' });
      if (!gl) throw new Error('no webgl');
      this.gl = gl; this.canvas = canvas;
      const sh = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s); if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s)); return s; };
      const p = gl.createProgram(); gl.attachShader(p, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(p, sh(gl.FRAGMENT_SHADER, FS)); gl.linkProgram(p);
      if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
      this.p = p; gl.useProgram(p);
      this.loc = {};
      for (const n of ['aPos', 'aUV']) this.loc[n] = gl.getAttribLocation(p, n);
      for (const n of ['uInv', 'uTex', 'uAlpha', 'uMode', 'uP', 'uRect', 'uTint']) this.loc[n] = gl.getUniformLocation(p, n);
      gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
      gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true);
      gl.uniform2f(this.loc.uInv, 1 / W, 1 / H);
      gl.uniform1i(this.loc.uTex, 0);
      this.quad = this.makeBuffers({ rest: new Float32Array(8), uv: new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]), idx: new Uint16Array([0, 1, 2, 1, 3, 2]) });
    }
    makeBuffers(mesh) {
      const gl = this.gl;
      const pos = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, pos); gl.bufferData(gl.ARRAY_BUFFER, mesh.rest.byteLength, gl.DYNAMIC_DRAW);
      const uv = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, uv); gl.bufferData(gl.ARRAY_BUFFER, mesh.uv, gl.STATIC_DRAW);
      const idx = gl.createBuffer(); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, idx); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, mesh.idx, gl.STATIC_DRAW);
      return { pos, uv, idx, count: mesh.idx.length };
    }
    texture(img) {
      const gl = this.gl; const t = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img); // throws on tainted file:// images
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      return t;
    }
    prepare(layers) {
      for (const l of layers) {
        if (l.kind === 'sprite') { l.tex = this.texture(l.img); continue; }
        l.tex = this.texture(l.img);
        l.buf = this.makeBuffers(l.mesh);
      }
    }
    resize(w, h) { this.canvas.width = w; this.canvas.height = h; this.gl.viewport(0, 0, w, h); }
    draw(layers) {
      const gl = this.gl, loc = this.loc;
      gl.clearColor(0.02, 0.01, 0.03, 1); gl.clear(gl.COLOR_BUFFER_BIT);
      for (const l of layers) {
        if (l.hidden || l.alpha <= 0.001) continue;
        gl.uniform1i(loc.uMode, l.mode); gl.uniform1f(loc.uAlpha, l.alpha);
        if (l.uP) gl.uniform4fv(loc.uP, l.uP);
        if (l.tint) gl.uniform3fv(loc.uTint, l.tint);
        gl.uniform4f(loc.uRect, l.rect.x, l.rect.y, l.rect.w, l.rect.h);
        gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, l.tex);
        if (l.kind === 'sprite') {
          for (const s of l.sprites) {
            if (s.a < 0.01) continue;
            const r = s.r; const q = new Float32Array([s.x - r, s.y - r, s.x + r, s.y - r, s.x - r, s.y + r, s.x + r, s.y + r]);
            gl.uniform1f(loc.uAlpha, s.a);
            this.bindAndDraw(this.quad, q);
          }
          continue;
        }
        this.bindAndDraw(l.buf, l.mesh.out);
      }
    }
    bindAndDraw(b, positions) {
      const gl = this.gl, loc = this.loc;
      gl.bindBuffer(gl.ARRAY_BUFFER, b.pos); gl.bufferSubData(gl.ARRAY_BUFFER, 0, positions);
      gl.enableVertexAttribArray(loc.aPos); gl.vertexAttribPointer(loc.aPos, 2, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, b.uv);
      gl.enableVertexAttribArray(loc.aUV); gl.vertexAttribPointer(loc.aUV, 2, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, b.idx);
      gl.drawElements(gl.TRIANGLES, b.count, gl.UNSIGNED_SHORT, 0);
    }
  }

  // Canvas 2D fallback: affine-textured triangles (coarser meshes), lighter effects.
  class Ctx2DRenderer {
    constructor(canvas) { this.canvas = canvas; this.ctx = canvas.getContext('2d'); }
    prepare() {}
    resize(w, h) { this.canvas.width = w; this.canvas.height = h; }
    draw(layers) {
      const ctx = this.ctx, k = this.canvas.width / W;
      ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.fillStyle = '#050208'; ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
      for (const l of layers) {
        if (l.hidden || l.alpha <= 0.001) continue;
        if (l.kind === 'sprite') {
          ctx.globalCompositeOperation = 'lighter';
          for (const s of l.sprites) { if (s.a < 0.01) continue; ctx.globalAlpha = s.a * 0.8; ctx.setTransform(k, 0, 0, k, 0, 0); ctx.drawImage(l.img, s.x - s.r, s.y - s.r, s.r * 2, s.r * 2); }
          ctx.globalCompositeOperation = 'source-over'; ctx.globalAlpha = 1; continue;
        }
        if (l.mode === 2) continue; // sheen needs the shader
        ctx.globalAlpha = l.alpha;
        const m = l.mesh, img = l.img, iw = img.naturalWidth || img.width, ih = img.naturalHeight || img.height;
        if (l.mode === 1) { // eye: squash toward centre line as the lid closes
          const lid = l.uP[0]; const sy = Math.max(0.04, 1 - lid);
          const o = m.out; const cx = (o[0] + o[2] + o[4] + o[6]) / 4, cy = o[1] + (o[5] - o[1]) * l.uP[2];
          ctx.setTransform(k, 0, 0, k * sy, 0, cy * k * (1 - sy));
          ctx.drawImage(img, o[0], o[1], o[2] - o[0], o[5] - o[1]); void cx; continue;
        }
        const P = m.out, U = m.uv, I = m.idx;
        for (let t = 0; t < I.length; t += 3) {
          const a = I[t], b = I[t + 1], c = I[t + 2];
          this.tri(ctx, img, iw, ih, k, P[a * 2], P[a * 2 + 1], P[b * 2], P[b * 2 + 1], P[c * 2], P[c * 2 + 1], U[a * 2] * iw, U[a * 2 + 1] * ih, U[b * 2] * iw, U[b * 2 + 1] * ih, U[c * 2] * iw, U[c * 2 + 1] * ih);
        }
      }
      ctx.globalAlpha = 1;
    }
    tri(ctx, img, iw, ih, k, x0, y0, x1, y1, x2, y2, u0, v0, u1, v1, u2, v2) {
      // expand slightly to hide seams between neighbouring triangles
      const cx = (x0 + x1 + x2) / 3, cy = (y0 + y1 + y2) / 3, e = 0.6;
      const ex = (x, y) => { const dx = x - cx, dy = y - cy, d = Math.hypot(dx, dy) || 1; return [x + dx / d * e, y + dy / d * e]; };
      const [X0, Y0] = ex(x0, y0), [X1, Y1] = ex(x1, y1), [X2, Y2] = ex(x2, y2);
      ctx.save();
      ctx.setTransform(k, 0, 0, k, 0, 0);
      ctx.beginPath(); ctx.moveTo(X0, Y0); ctx.lineTo(X1, Y1); ctx.lineTo(X2, Y2); ctx.closePath(); ctx.clip();
      const den = (u1 - u0) * (v2 - v0) - (u2 - u0) * (v1 - v0);
      if (Math.abs(den) < 1e-9) { ctx.restore(); return; }
      const a = ((x1 - x0) * (v2 - v0) - (x2 - x0) * (v1 - v0)) / den;
      const b = ((y1 - y0) * (v2 - v0) - (y2 - y0) * (v1 - v0)) / den;
      const c = ((x2 - x0) * (u1 - u0) - (x1 - x0) * (u2 - u0)) / den;
      const d = ((y2 - y0) * (u1 - u0) - (y1 - y0) * (u2 - u0)) / den;
      const e2 = x0 - a * u0 - c * v0, f = y0 - b * u0 - d * v0;
      ctx.transform(a, b, c, d, e2, f);
      ctx.drawImage(img, 0, 0);
      ctx.restore();
    }
  }

  // ------------------------------------------------------------------ effects (cybernetic details, eyes, glow)
  function updateEffects(char, layers, tau) {
    const S = char.skel;
    let bi = 0, li = 0;
    for (const l of layers) {
      if (l.fx === 'bar') {
        const i = bi++;
        const n = 0.55 * harm(3 + i * 2, tau, i * 1.3) + 0.3 * harm(7 + i, tau, i * 2.1) + 0.55 * char.activity(tau - i * 0.06) - 0.2;
        l.alpha = 0.32 + 0.68 * clamp(0.5 + 0.5 * n, 0, 1);
      } else if (l.fx === 'led') {
        const i = li++;
        const pats = [
          () => 0.5 + 0.5 * harm(2, tau, 0),                                       // slow breathe
          () => smooth(0.2, 0.3, (tau * 1.0) % 1) * (1 - smooth(0.55, 0.65, (tau * 1.0) % 1)), // 1 Hz blink
          () => Math.max(gauss(wrapDelta(tau, 1.0), 0, 0.05), gauss(wrapDelta(tau, 1.22), 0, 0.05), gauss(wrapDelta(tau, 5.0), 0, 0.05), gauss(wrapDelta(tau, 5.22), 0, 0.05)),
          () => 0.5 + 0.5 * harm(4, tau, 1.5),
          () => smooth(0.0, 0.08, (tau * 0.5) % 1) * (1 - smooth(0.4, 0.5, (tau * 0.5) % 1)),
          () => clamp(char.activity(tau) * 0.9, 0, 1),
          () => 0.5 + 0.5 * harm(1, tau, 2.5),
          () => gauss(wrapDelta(tau, 6.5), 0, 0.4),
          () => 0.5 + 0.5 * harm(6, tau, 0.3),
          () => smooth(0.5, 0.56, (tau * 0.75) % 1) * (1 - smooth(0.7, 0.8, (tau * 0.75) % 1))
        ];
        l.alpha = 0.3 + 0.7 * clamp(pats[i % pats.length](), 0, 1);
      } else if (l.fx === 'sheen') {
        // one slow reflection sweep across the armor per loop, plus a faint secondary glint
        const sweep = smooth(4.4, 6.6, tau);
        l.uP = [0.78, 0.62, 520 + sweep * 980, 34];
        l.tint = [1.0, 0.82, 0.86];
        l.alpha = 0.28 * Math.sin(Math.PI * sweep);
      } else if (l.fx === 'eye') {
        const isR = l.name === 'eye_r';
        const lid = char.blink(tau, isR ? -0.012 : 0);
        const meta = DATA.layers[l.name];
        l.uP = [lid, meta.cu, meta.cv, 0];
        l.alpha = 1;
      } else if (l.fx === 'keyglow') {
        l.tint = [1.0, 0.42, 0.55];
        l.alpha = 1;
        l.sprites.length = 0;
        for (const f of ['f1', 'f2', 'f3', 'f4']) {
          const tip = S.get(f + 'c') || S.get(f + 'b');
          const [x, y] = S.tailPos(tip);
          let a = 0;
          for (const s of char.strikesBy[f]) { const d = wrapDelta(tau, s.t); if (d > -0.05 && d < 0.4) a += s.amp * gauss(d, 0.03, 0.09); }
          l.sprites.push({ x: x - 2, y: y + 3, r: 17, a: clamp(a, 0, 1) * 0.55 });
        }
      }
    }
  }

  // ------------------------------------------------------------------ app
  function loadImage(src) {
    return new Promise((res, rej) => { const i = new Image(); i.decoding = 'async'; i.onload = () => res(i); i.onerror = () => rej(new Error('load ' + src)); i.src = src; });
  }
  function glowSprite() {
    const c = document.createElement('canvas'); c.width = c.height = 64;
    const g = c.getContext('2d'); const gr = g.createRadialGradient(32, 32, 0, 32, 32, 32);
    gr.addColorStop(0, 'rgba(255,255,255,1)'); gr.addColorStop(0.35, 'rgba(255,255,255,0.45)'); gr.addColorStop(1, 'rgba(255,255,255,0)');
    g.fillStyle = gr; g.fillRect(0, 0, 64, 64); return c;
  }

  class App {
    constructor(canvas, options) {
      this.canvas = canvas; this.opts = Object.assign({ debug: false }, options || {});
      this.char = new Character();
      this.timeScale = 1; this.paused = false; this.simTime = 0; this.forcedTime = null;
      this.reduced = window.matchMedia ? window.matchMedia('(prefers-reduced-motion: reduce)') : { matches: false };
      this.visible = true; this.raf = 0; this.last = 0; this.acc = 0;
      this.listeners = [];
    }
    async init() {
      let renderer = null, coarse = false;
      try { renderer = new GLRenderer(this.canvas); } catch (e) { renderer = null; }
      const files = {};
      const layersAll = () => setupLayers(this.char, coarse);
      let layers = layersAll();
      await Promise.all(layers.map(async (l) => {
        if (l.kind === 'sprite') { l.img = glowSprite(); return; }
        const src = ASSET_BASE + 'layers/' + l.file;
        files[l.name] = files[l.name] || await loadImage(src);
        l.img = files[l.name];
      }));
      if (renderer) {
        try { renderer.prepare(layers); }
        catch (e) { console.warn('WebGL textures unavailable (opened from disk?), using Canvas 2D:', e.message); renderer = null; }
      }
      if (!renderer) {
        coarse = true;
        const c2 = this.canvas.cloneNode(false); this.canvas.replaceWith(c2); this.canvas = c2;
        renderer = new Ctx2DRenderer(this.canvas);
        const imgs = Object.fromEntries(layers.map((l) => [l.name, l.img]));
        layers = layersAll(); layers.forEach((l) => { l.img = imgs[l.name]; });
      }
      this.renderer = renderer; this.layers = layers;
      this.renderer.kind = renderer instanceof GLRenderer ? 'webgl' : 'canvas2d';
      this.resize();
      if ('ResizeObserver' in window) { this.ro = new ResizeObserver(() => this.resize()); this.ro.observe(this.canvas.parentElement || this.canvas); }
      else window.addEventListener('resize', () => this.resize());
      document.addEventListener('visibilitychange', () => this.updateRunning());
      if ('IntersectionObserver' in window) {
        this.io = new IntersectionObserver((es) => { this.visible = es[0].isIntersecting; this.updateRunning(); });
        this.io.observe(this.canvas);
      }
      const onRM = () => this.updateRunning();
      if (this.reduced.addEventListener) this.reduced.addEventListener('change', onRM); else if (this.reduced.addListener) this.reduced.addListener(onRM);
      // settle the cloth into its periodic motion before the first frame
      this.char.pose(0); for (const ch of Object.values(this.char.chains)) ch.snap();
      this.simulate(LOOP * 2, true);
      this.simTime = 0;
      this.renderFrame();
      this.updateRunning();
      this.canvas.dataset.ready = '1';
      return this;
    }
    resize() {
      const el = this.canvas; const r = el.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const maxW = this.renderer.kind === 'webgl' ? W * 1.5 : W;
      const w = Math.max(2, Math.round(Math.min(r.width * dpr, maxW)));
      const h = Math.max(2, Math.round(w * H / W));
      if (el.width !== w || el.height !== h) { this.renderer.resize(w, h); this.renderFrame(); }
    }
    get motionAllowed() { return !(this.reduced && this.reduced.matches) || this.opts.ignoreReducedMotion; }
    updateRunning() {
      const run = this.motionAllowed && !document.hidden && this.visible && !this.paused;
      if (run && !this.raf) { this.last = performance.now(); this.raf = requestAnimationFrame((t) => this.tick(t)); }
      if (!run && this.raf) { cancelAnimationFrame(this.raf); this.raf = 0; }
      if (!this.motionAllowed) { this.forcedTimeRM = 0.6; this.renderFrame(); }
      else this.forcedTimeRM = null;
    }
    simulate(seconds, warm) {
      const dt = 1 / 120;
      this.acc += seconds;
      while (this.acc >= dt) {
        this.simTime += dt; this.acc -= dt;
        const tau = ((this.simTime % LOOP) + LOOP) % LOOP;
        this.char.pose(tau);
        for (const ch of Object.values(this.char.chains)) ch.step(dt, this.simTime, 1);
      }
    }
    tick(now) {
      this.raf = 0;
      const dt = Math.min((now - this.last) / 1000, 0.1); this.last = now;
      if (this.forcedTime === null) this.simulate(dt * this.timeScale, false);
      this.renderFrame();
      for (const f of this.listeners) f(this);
      this.updateRunning();
    }
    loopTime() {
      if (this.forcedTimeRM != null) return this.forcedTimeRM;
      if (this.forcedTime != null) return this.forcedTime;
      return ((this.simTime % LOOP) + LOOP) % LOOP;
    }
    renderFrame() {
      if (!this.renderer) return;
      const tau = this.loopTime();
      this.char.pose(tau);
      for (const ch of Object.values(this.char.chains)) ch.updateBones();
      updateEffects(this.char, this.layers, tau);
      for (const l of this.layers) {
        if (l.kind === 'skinned' || l.kind === 'overlay') skinInto(l.mesh.out, l.mesh.rest, l.skin, l.infl);
      }
      this.renderer.draw(this.layers);
    }
    // used by preview.html and the automated frame capture
    seek(t) {
      // deterministic: rebuild cloth state from a settled loop and run up to t
      this.simTime = 0; this.acc = 0;
      this.char.pose(0); for (const ch of Object.values(this.char.chains)) ch.snap();
      this.simulate(LOOP * 2, true);
      this.simTime = 0; this.acc = 0;
      this.simulate(t, true);
      this.renderFrame();
    }
  }

  window.KathakaliAnimation = {
    App, Character, LOOP, W, H,
    start(canvas, options) { const app = new App(canvas, options); return app.init(); }
  };

  const auto = document.querySelector('canvas[data-kathakali]');
  if (auto && auto.dataset.kathakali !== 'manual') {
    window.KathakaliAnimation.start(auto).then((app) => { window.kathakali = app; }).catch((e) => console.error(e));
  }
})();
