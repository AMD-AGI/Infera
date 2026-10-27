// The board's graph, without the page: where cards go (lanes × rows in the order the tasks run) and how the edges
// between them are routed. Pure functions, tested in web/test/graph.test.js. An edge is the server's
// ({from, to, kind, blocking, ...}).
//
// Edges are orthogonal lines with rounded corners that run only in the space between cards: the lane gaps (vertical
// channels) and the gaps between rows (horizontal channels). Where several edges share a channel each gets its own
// track, and the tracks and the ports on a card's side are ordered so that lines do not cross where they need not
// (the channel-routing rule: a line that leaves a channel inside another line's span goes on that side of it).

export const NODE_W = 248, NODE_H = 76, COMPACT_H = 30, LANE_GAP = 72, LAYER_GAP = 34, COMPACT_GAP = 14, MARGIN_X = 56,
  HEADER_H = 40, AXIS_W = 52;
export const laneX = (col) => AXIS_W + MARGIN_X + col * (NODE_W + LANE_GAP);
const laneOf = (x) => Math.round((x - laneX(0)) / (NODE_W + LANE_GAP));
const dayOf = (iso) => (iso ? new Date(iso).toLocaleDateString('sv').slice(5) : '');  // local MM-DD
const TRACK_V = 8, TRACK_H = 6, PORT_SIDE = 8, RADIUS = 6;
// The drafts' band starts this far above the first draft row; its title is drawn in it, under the gap above.
export const DRAFT_BAND = 28;

// The lane (column index among n) under x on the canvas, for a card or a lane title dropped there.
export const laneAt = (x, n) => Math.max(0, Math.min(n - 1, Math.floor((x - laneX(0) + LANE_GAP / 2) / (NODE_W + LANE_GAP))));

// The order the tasks run in: each task with its time, the start, or for one not started yet its creation but no
// earlier than its upstream tasks' times (it cannot start before them), so a task never comes before the tasks it
// waits for. Equal times (a template's tasks are made at once) go by how deep the task is in the graph, then by lane.
// Drafts ("~…") come after every date. Returns {order: [task], when: (id) => time}.
export function runOrder(tasks, edges, laneIds) {
  const byId = new Map(tasks.map((t) => [t.id, t]));
  const ups = new Map();
  for (const e of edges) if (e.from !== e.to && byId.has(e.from) && byId.has(e.to)) ups.set(e.to, [...(ups.get(e.to) || []), e.from]);
  const memo = (f) => {  // f(id, rec) over the upstream tasks; a cycle is cut where it closes
    const done = new Map(), open = new Set();
    const rec = (id) => {
      if (done.has(id)) return done.get(id);
      if (open.has(id)) return null;
      open.add(id);
      const v = f(id, rec);
      open.delete(id);
      done.set(id, v);
      return v;
    };
    return rec;
  };
  const when = memo((id, rec) => {
    const t = byId.get(id), draft = t.status === 'draft';
    let k = t.started || t.created || '';
    // a task that is not a draft does not wait for a draft's time
    if (!t.started) for (const u of ups.get(id) || []) if (draft || byId.get(u).status !== 'draft') k = [k, (rec(u) || '').replace(/^~/, '')].sort()[1];
    return draft ? `~${k}` : k;
  });
  const depth = memo((id, rec) => Math.max(-1, ...(ups.get(id) || []).map((u) => rec(u) ?? -1)) + 1);
  const cmp = (a, b) => (a > b) - (a < b);
  const order = [...tasks].sort((a, b) => cmp(when(a.id), when(b.id)) || depth(a.id) - depth(b.id)
    || laneIds.indexOf(a.type) - laneIds.indexOf(b.type) || cmp(a.id, b.id));
  return { order, when };
}

// Rows follow runOrder, top to bottom; each lane is a column: the first task top left, the last bottom right. A task
// goes on the current row unless its lane is taken there, its time is on a later day, or it is linked to a task on that
// row, so every edge spans rows. Drafts have no start: they get their own rows at the bottom (the draft band). An edge
// from a task that started later points up. edges: all of them, also the kinds the board hides, so hiding a kind keeps
// the rows. drawn: the edges that will be drawn (default: edges); the gaps between rows grow to give each its own
// track (so cards may move down a little). Returns the cards' places, the rows, the band's top (draftTop), the size,
// and one SVG path per drawn edge (paths, same order).
export function layout(tasks, edges, laneIds, compact, drawn = edges) {
  const { order, when } = runOrder(tasks, edges, laneIds);
  const layer = new Map(), laneRow = new Map(), rows = [];  // rows[i]: {day, draft}
  for (const t of order) {
    const row = rows.length - 1, draft = t.status === 'draft', day = draft ? '草稿' : dayOf(when(t.id));  // i18n: not shown (the drafts band has no day label)
    const linked = edges.some((e) => (e.to === t.id && layer.get(e.from) === row) || (e.from === t.id && layer.get(e.to) === row));
    const l = row < 0 || day !== rows[row].day || laneRow.get(t.type) === row || linked ? row + 1 : row;
    if (l > row) rows.push({ day, draft });
    layer.set(t.id, l);
    laneRow.set(t.type, l);
  }
  const heights = rows.map(() => COMPACT_H);
  for (const t of tasks) if (!compact.has(t.id)) heights[layer.get(t.id)] = NODE_H;
  const place = (need) => {
    const tops = [];
    let y = HEADER_H;
    rows.forEach((r, i) => {
      if (r.draft && !rows[i - 1]?.draft) y += DRAFT_BAND;  // room for the draft band's title
      tops.push(y);
      // collapsed rows sit closer, unless an edge needs the gap below them to turn
      const base = heights[i] === COMPACT_H && heights[i + 1] === COMPACT_H && !need.has(i) ? COMPACT_GAP : LAYER_GAP;
      y += heights[i] + Math.max(base, need.get(i) || 0);
    });
    const pos = new Map();
    // the draft band takes the bottom of the gap above the first draft row (band): lines keep above it
    const band = (i) => (rows[i]?.draft && !rows[i - 1]?.draft ? DRAFT_BAND : 0);
    for (const t of tasks) {
      const l = layer.get(t.id);
      const above = l ? tops[l] - tops[l - 1] - heights[l - 1] - band(l) : LAYER_GAP, below = l < rows.length - 1 ? tops[l + 1] - tops[l] - heights[l] - band(l + 1) : LAYER_GAP;
      pos.set(t.id, { x: laneX(laneIds.indexOf(t.type)), y: tops[l], w: NODE_W, h: compact.has(t.id) ? COMPACT_H : NODE_H, rowH: heights[l], layer: l, above, below, band: band(l) });
    }
    return { tops, pos, bottom: y };
  };
  const routed = drawn.filter((e) => layer.has(e.from) && layer.has(e.to));
  let P = place(new Map()), R = route(routed, P.pos);
  // a gap between rows that more edges run along than fit grows to take them
  const need = new Map([...R.tracks].map(([i, n]) => [i, (n - 1) * TRACK_H + 20]));
  if ([...need].some(([i, h]) => h > LAYER_GAP || (heights[i] === COMPACT_H && heights[i + 1] === COMPACT_H && h > COMPACT_GAP))) {
    P = place(need);
    R = route(routed, P.pos);
  }
  const path = new Map(routed.map((e, i) => [e, R.paths[i]]));
  const draftRow = rows.findIndex((r) => r.draft);
  return { pos: P.pos, rows: rows.map((r, i) => ({ ...r, y: P.tops[i], h: heights[i] })), draftTop: draftRow < 0 ? null : P.tops[draftRow] - DRAFT_BAND,
    width: laneX(Math.max(laneIds.length, 1) - 1) + NODE_W + MARGIN_X, height: Math.max(P.bottom, 320),  // room for the empty state
    paths: drawn.map((e) => path.get(e) ?? '') };
}

// The edges that another path along the given edges already implies (A→C when A→B→C is there): drawing them adds
// lines, mostly long ones, and no order the others do not show. Only the blocking kinds (产出依赖, 代码基线, 等子任务)
// make a path: 评审覆盖 and 父子任务 do not order the tasks. Only 产出依赖 and 代码基线 are left out: 等子任务 also
// waits for the upstream's subtasks, which no path shows. An edge left out stops counting as a path at once, so in a
// cycle one way round stays drawn. Returns a Set of the edges.
export function redundant(edges) {
  const flow = edges.filter((e) => e.blocking);
  const next = new Map();
  for (const e of flow) next.set(e.from, [...(next.get(e.from) || []), e]);
  const reaches = (from, to, skip) => {
    const seen = new Set([from]), stack = [from];
    while (stack.length) {
      for (const e of next.get(stack.pop()) || []) {
        if (e === skip || seen.has(e.to)) continue;
        if (e.to === to) return true;
        seen.add(e.to);
        stack.push(e.to);
      }
    }
    return false;
  };
  const out = new Set();
  for (const e of flow) {
    if (e.kind === 'children' || e.from === e.to || !reaches(e.from, e.to, e)) continue;
    out.add(e);
    next.set(e.from, next.get(e.from).filter((x) => x !== e));
  }
  return out;
}

// The shapes (a = upstream card, b = downstream card):
//   tt       next row down: a's bottom, along the gap between the rows, into b's top;
//   side     same lane, rows apart: out of a's side, along the lane gap, into b's same side (right going down,
//            left going up);
//   gap      other lane, rows apart: out of a's side toward b, along the lane gap next to a, along the gap between
//            rows next to b, into b's top (b below) or bottom (b above).
function shape(a, b) {
  const la = laneOf(a.x), lb = laneOf(b.x);
  if (b.layer === a.layer + 1) return { kind: 'tt', out: 'bottom', into: 'top' };
  if (la === lb) return b.layer > a.layer ? { kind: 'side', out: 'right', into: 'right', gap: la } : { kind: 'side', out: 'left', into: 'left', gap: la - 1 };
  const right = lb > la;
  return { kind: 'gap', out: right ? 'right' : 'left', into: b.layer > a.layer ? 'top' : 'bottom', gap: right ? la : la - 1 };
}

// The x of the middle of lane gap g (between lanes g and g+1; -1 and the last lane's index: the margins).
const gapX = (g) => (g < 0 ? laneX(0) - MARGIN_X / 2 : laneX(g) + NODE_W + LANE_GAP / 2);
const gapRoom = (g, n) => (g < 0 || g >= n - 1 ? MARGIN_X - 16 : LANE_GAP - 24);
// The y of the middle of the gap between a card's row and the row above (top: above the draft band, if any) or below.
const rowGapY = (p, side) => (side === 'top' ? p.y - p.band - p.above / 2 : p.y + p.rowH + p.below / 2);

// One SVG path per edge (same order), from the upstream card to the downstream card, and the number of tracks in
// each gap between rows that lines run across ({tracks}: row above the gap -> n).
function route(edges, pos) {
  const lanes = Math.max(0, ...[...pos.values()].map((p) => laneOf(p.x))) + 1;
  const items = edges.map((e) => {
    const a = pos.get(e.from), b = pos.get(e.to);
    return { e, a, b, ...shape(a, b) };
  });
  pickSides(items);
  for (const it of items) {
    // where the line goes after it leaves a side (rough: lane gaps and row gaps by their middles), for ordering
    it.rowGap = it.kind === 'tt' ? rowGapY(it.a, 'bottom') : rowGapY(it.b, it.into);
    it.far = { out: it.kind === 'tt' ? it.b.x + NODE_W / 2 : it.kind === 'gap' ? gapX(it.gap) : 0,
      into: it.kind === 'tt' ? it.a.x + NODE_W / 2 : it.kind === 'gap' ? gapX(it.gap) : 0 };
  }
  placePorts(items, pos);
  laneTracks(items, lanes);
  const tracks = rowTracks(items);
  return { paths: items.map(draw), tracks };
}

// A line that stays in its lane goes round the side whose gap fewer other lines run along or across over its rows.
function pickSides(items) {
  // the part of a line that runs across a row gap: [the row above the gap, from, to], in lanes (lane gap g is at g + 0.5)
  const across = (o) => (o.kind === 'tt' ? [o.a.layer, laneOf(o.a.x), laneOf(o.b.x)]
    : o.kind === 'gap' ? [o.into === 'top' ? o.b.layer - 1 : o.b.layer, o.gap + 0.5, laneOf(o.b.x)] : null);
  const busy = (g, it) => {
    const r0 = Math.min(it.a.layer, it.b.layer), r1 = Math.max(it.a.layer, it.b.layer);
    let n = 0;
    for (const o of items) {
      const h = across(o);
      if (h && h[0] >= r0 && h[0] < r1 && Math.min(h[1], h[2]) < g + 0.5 && Math.max(h[1], h[2]) > g + 0.5) n++;
      if (o.kind === 'gap' && o.gap === g && Math.max(o.a.layer, o.b.layer) > r0 && Math.min(o.a.layer, o.b.layer) < r1) n++;
    }
    return n;
  };
  for (const it of items) {
    if (it.kind !== 'side') continue;
    const l = laneOf(it.a.x), [first, other] = it.out === 'right' ? [l, l - 1] : [l - 1, l];
    if (busy(other, it) < busy(first, it)) Object.assign(it, { gap: other, out: other === l ? 'right' : 'left', into: other === l ? 'right' : 'left' });
  }
}

// The places on a card's side where lines leave or come in (it.pa, it.pb), ordered so that lines on the same side
// do not cross each other.
function placePorts(items, pos) {
  const sides = new Map();
  const add = (id, side, it, end) => {
    const k = `${id}|${side}`;
    if (!sides.has(k)) sides.set(k, []);
    sides.get(k).push({ it, end });
  };
  for (const it of items) {
    add(it.e.from, it.out, it, 'out');
    add(it.e.to, it.into, it, 'into');
  }
  const otherY = (p) => (p.end === 'out' ? p.it.b.y : p.it.a.y);
  for (const [k, list] of sides) {
    const [id, side] = k.split('|'), p0 = pos.get(id);
    if (side === 'top' || side === 'bottom') {
      // along the bottom or top: by where the line goes along the gap, left to right
      list.sort((p, q) => p.it.far[p.end] - q.it.far[q.end] || otherY(p) - otherY(q));
    } else {
      // on the left or right, top to bottom: lines going up and coming back to this side (nearest first), lines
      // going up and on across the gap (farthest first), then the same going down in mirror order, so that a line
      // never leaves the gap inside another's span on the side it would have to cross
      const mid = p0.y + p0.h / 2;
      const key = (p) => {
        const it = p.it, y = p.end === 'into' ? it.a.y + it.a.h / 2 : it.kind === 'side' ? it.b.y + it.b.h / 2 : it.rowGap;
        const group = y < mid ? (it.kind === 'side' ? 0 : 1) : (it.kind === 'side' ? 3 : 2);
        return [group, group === 0 || group === 3 ? -y : y];
      };
      list.sort((p, q) => { const [g1, y1] = key(p), [g2, y2] = key(q); return g1 - g2 || y1 - y2; });
    }
    list.forEach((p, i) => { p.it[p.end === 'out' ? 'pa' : 'pb'] = portAt(p0, side, i, list.length); });
  }
  // next row in the same lane: the end that is alone on its side moves over to the other one, and the line is straight
  for (const it of items) {
    if (it.kind !== 'tt' || it.a.x !== it.b.x) continue;
    if (sides.get(`${it.e.to}|top`).length === 1) it.pb = { ...it.pb, x: it.pa.x };
    else if (sides.get(`${it.e.from}|bottom`).length === 1) it.pa = { ...it.pa, x: it.pb.x };
  }
}

// The lane gaps: one track (it.vx) per line that runs along one. A pin is where the line leaves the gap (dir -1: to
// the left, 1: to the right); port: at a card's side, rather than into a gap between rows.
function laneTracks(items, lanes) {
  const segs = new Map();
  for (const it of items) {
    if (it.kind === 'tt') continue;
    const y2 = it.kind === 'side' ? it.pb.y : it.rowGap;
    const pinA = { at: it.pa.y, dir: it.out === 'right' ? -1 : 1, port: true };
    const pinB = { at: y2, dir: it.kind === 'side' ? pinA.dir : Math.sign(it.b.x + NODE_W / 2 - gapX(it.gap)), port: it.kind === 'side' };
    if (!segs.has(it.gap)) segs.set(it.gap, []);
    segs.get(it.gap).push({ it, lo: Math.min(it.pa.y, y2), hi: Math.max(it.pa.y, y2), pins: [pinA, pinB] });
  }
  for (const [g, list] of segs) {
    const n = tracks(list), step = Math.min(TRACK_V, gapRoom(g, lanes) / Math.max(n - 1, 1));
    for (const s of list) s.it.vx = gapX(g) + (s.track - (n - 1) / 2) * step;
  }
}

// The gaps between rows: one track (it.hy) per line that runs across one (pins: -1 = up out of the gap, 1 = down).
// Returns row above the gap -> the number of tracks.
function rowTracks(items) {
  const segs = new Map();
  for (const it of items) {
    if (it.kind === 'side') continue;
    const [x1, x2] = it.kind === 'tt' ? [it.pa.x, it.pb.x] : it.into === 'top' ? [it.vx, it.pb.x] : [it.pb.x, it.vx];
    if (Math.abs(x1 - x2) < 1) continue;  // straight down: nothing runs across
    const row = it.kind === 'tt' ? it.a.layer : it.into === 'top' ? it.b.layer - 1 : it.b.layer;
    const pins = [{ at: x1, dir: -1, port: it.kind === 'tt' || it.into === 'bottom' }, { at: x2, dir: 1, port: it.kind === 'tt' || it.into === 'top' }];
    if (!segs.has(row)) segs.set(row, []);
    segs.get(row).push({ it, lo: Math.min(x1, x2), hi: Math.max(x1, x2), pins });
  }
  const counts = new Map();
  for (const [row, list] of segs) {
    const n = tracks(list);
    counts.set(row, n);
    for (const s of list) {
      const p = s.it.kind === 'tt' ? s.it.a : s.it.b, side = s.it.kind === 'tt' ? 'bottom' : s.it.into;
      const room = (side === 'bottom' ? p.below : p.above) - 16;
      s.it.hy = rowGapY(p, side) + (s.track - (n - 1) / 2) * Math.min(TRACK_H, room / Math.max(n - 1, 1));
    }
  }
  return counts;
}

function draw(it) {
  const { pa, pb } = it;
  if (it.kind === 'tt') return it.hy == null ? `M${pa.x},${pa.y} L${pb.x},${pb.y}` : rounded([pa, { x: pa.x, y: it.hy }, { x: pb.x, y: it.hy }, pb]);
  if (it.kind === 'side') return rounded([pa, { x: it.vx, y: pa.y }, { x: it.vx, y: pb.y }, pb]);
  const y = it.hy ?? it.rowGap;
  return rounded([pa, { x: it.vx, y: pa.y }, { x: it.vx, y }, { x: pb.x, y }, pb]);
}

function portAt(p, side, i, n) {
  if (side === 'top' || side === 'bottom') return { x: Math.round(p.x + (p.w * (i + 1)) / (n + 1)), y: side === 'top' ? p.y : p.y + p.h };
  return { x: side === 'left' ? p.x : p.x + p.w, y: p.y + p.h / 2 + (i - (n - 1) / 2) * Math.min(PORT_SIDE, (p.h - 12) / Math.max(n - 1, 1)) };
}

// Give each segment of one channel a track (s.track, 0 = the left or top one); returns how many tracks there are.
// A segment whose line leaves the channel inside another's span, toward one side, should lie on that side of it,
// or the two cross; the order keeps as many of these as it can (greedy, the segment fewest others must precede
// first), and overlapping segments get different tracks in that order.
export function tracks(segs) {
  const inside = (u, s) => u > s.lo + 0.5 && u < s.hi - 0.5;
  const overlap = (p, q) => p.lo < q.hi + 3 && q.lo < p.hi + 3;
  const pref = segs.map((p) => segs.map((q) => {  // > 0: p before (left of / above) q
    if (p === q || !overlap(p, q)) return 0;
    let v = 0;
    for (const pin of p.pins) if (inside(pin.at, q)) v -= pin.dir;
    for (const pin of q.pins) if (inside(pin.at, p)) v += pin.dir;
    // two lines leaving at the same place, one each way: the one leaving up (left) lies above (left of) the other,
    // or the two would run on top of each other there and read as one line (worse than a crossing: it weighs double
    // where both leave at a card's side; going on into a gap, the two get tracks of their own there)
    for (const a of p.pins) for (const b of q.pins) if (Math.abs(a.at - b.at) < 0.5 && a.dir !== b.dir) v -= (a.port && b.port ? 2 : 1) * a.dir;
    return v;
  }));
  const left = segs.map((_, i) => i), order = [];
  while (left.length) {
    const cost = (i) => left.reduce((c, j) => c + Math.max(0, pref[j][i]), 0);
    let best = left[0];
    for (const i of left) if (cost(i) < cost(best) || (cost(i) === cost(best) && segs[i].hi - segs[i].lo < segs[best].hi - segs[best].lo)) best = i;
    order.push(best);
    left.splice(left.indexOf(best), 1);
  }
  let n = 0;
  order.forEach((i, k) => {
    const before = order.slice(0, k).filter((j) => overlap(segs[i], segs[j]));
    let t = Math.max(0, ...before.filter((j) => pref[j][i] > 0).map((j) => segs[j].track + 1));
    while (before.some((j) => segs[j].track === t)) t++;
    segs[i].track = t;
    n = Math.max(n, t + 1);
  });
  return n;
}

function rounded(pts, r = RADIUS) {
  let d = `M${pts[0].x},${pts[0].y}`;
  for (let i = 1; i < pts.length - 1; i++) {
    const [a, p, b] = [pts[i - 1], pts[i], pts[i + 1]];
    const la = Math.hypot(a.x - p.x, a.y - p.y), lb = Math.hypot(b.x - p.x, b.y - p.y);
    if (!la || !lb) continue;
    const ra = Math.min(r, la / 2) / la, rb = Math.min(r, lb / 2) / lb;
    d += ` L${p.x + (a.x - p.x) * ra},${p.y + (a.y - p.y) * ra} Q${p.x},${p.y} ${p.x + (b.x - p.x) * rb},${p.y + (b.y - p.y) * rb}`;
  }
  const last = pts[pts.length - 1];
  return `${d} L${last.x},${last.y}`;
}
