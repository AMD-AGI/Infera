// The board's graph (web/graph.js): where cards go, edge routes that keep to the gaps and do not cross where they need
// not, the edges another path implies, and the gaps between rows that grow for the lines running along them.
//   node --test src/agent_sys_v2/web/test/*.test.js
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { layout, laneAt, redundant, tracks, DRAFT_BAND } from '../graph.js';

const task = (id, started, status = 'done', extra = {}) => ({ id, type: id.split('/')[0], status, started, created: started, ...extra });
const at = (h) => `2026-10-20T${String(h).padStart(2, '0')}:00:00Z`;

// the corners of a path: its start, the corners the rounded turns go round (their Q control points) and its end
function points(d) {
  const all = [...d.matchAll(/([MLQ])(-?[\d.]+),(-?[\d.]+)/g)].map((m) => ({ c: m[1], x: +m[2], y: +m[3] }));
  return [all[0], ...all.filter((p) => p.c === 'Q'), all[all.length - 1]];
}
const segments = (d) => { const p = points(d); return p.slice(1).map((q, i) => [p[i], q]); };
function crosses([a, b], [c, d]) {
  const den = (b.x - a.x) * (d.y - c.y) - (b.y - a.y) * (d.x - c.x);
  if (!den) return false;
  const u = ((c.x - a.x) * (d.y - c.y) - (c.y - a.y) * (d.x - c.x)) / den, v = ((c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x)) / den;
  return u > 0.01 && u < 0.99 && v > 0.01 && v < 0.99;
}
const crossings = (paths) => {
  let n = 0;
  for (let i = 0; i < paths.length; i++) for (let j = i + 1; j < paths.length; j++)
    for (const s of segments(paths[i])) for (const r of segments(paths[j])) n += crosses(s, r);
  return n;
};
// two paths running on top of each other somewhere (they would read as one line)
const overlaps = (paths) => {
  let n = 0;
  for (let i = 0; i < paths.length; i++) for (let j = i + 1; j < paths.length; j++)
    for (const [a, b] of segments(paths[i])) for (const [c, d] of segments(paths[j])) {
      const along = a.y === b.y && c.y === d.y && a.y === c.y ? [a.x, b.x, c.x, d.x] : a.x === b.x && c.x === d.x && a.x === c.x ? [a.y, b.y, c.y, d.y] : null;
      if (along && Math.min(Math.max(along[0], along[1]), Math.max(along[2], along[3])) - Math.max(Math.min(along[0], along[1]), Math.min(along[2], along[3])) > 0.5) n++;
    }
  return n;
};
// every corner of every path is outside every card it does not start or end at
const clear = (edges, paths, pos) => edges.every((e, i) => points(paths[i]).every((p) => [...pos].every(([id, c]) =>
  id === e.from || id === e.to || !(p.x > c.x && p.x < c.x + c.w && p.y > c.y && p.y < c.y + c.h))));

test('edges are orthogonal lines in the gaps, and lines leaving the same side nest instead of crossing', () => {
  // a feeds b (next row, same lane), c (same lane, two rows down), d and e (the next lane, further down)
  const tasks = [task('x/a', at(1)), task('x/b', at(2)), task('x/c', at(3)), task('y/d', at(4)), task('y/e', at(5))];
  const edges = [{ from: 'x/a', to: 'x/b' }, { from: 'x/a', to: 'x/c' }, { from: 'x/a', to: 'y/e' }, { from: 'x/a', to: 'y/d' }];
  const L = layout(tasks, edges, ['x', 'y'], new Set());
  const paths = L.paths;
  for (const d of paths) for (const [p, q] of segments(d)) assert.ok(p.x === q.x || p.y === q.y, `not orthogonal: ${d}`);
  assert.equal(crossings(paths), 0);
  assert.ok(clear(edges, paths, L.pos));
  // b → d now runs across the gap a's lines go down: a → c goes round the other side of the lane, a → e still crosses
  const more = [...edges, { from: 'x/b', to: 'y/d' }];
  const M = layout(tasks, more, ['x', 'y'], new Set());
  const two = M.paths;
  assert.ok(two[1].startsWith(`M${M.pos.get('x/a').x},`), two[1]);  // out of a's left side
  assert.equal(crossings(two), 1);
});

test('the order of the tracks in a gap avoids crossings the old insertion order made', () => {
  // three lines down the same lane gap from cards on the left, each turning right lower down than the one before
  const tasks = [task('x/a', at(1)), task('x/b', at(2)), task('x/c', at(3)), task('y/p', at(4)), task('y/q', at(5)), task('y/r', at(6))];
  const edges = [{ from: 'x/a', to: 'y/p' }, { from: 'x/b', to: 'y/q' }, { from: 'x/c', to: 'y/r' }];
  for (const order of [edges, [...edges].reverse()]) {
    const L = layout(tasks, order, ['x', 'y'], new Set());
    const paths = L.paths;
    assert.equal(crossings(paths), 0, paths.join('\n'));
    assert.ok(clear(order, paths, L.pos));
  }
});

test('tracks: a segment leaving the channel inside another one\'s span goes on that side of it', () => {
  // p runs 0..60, q 20..100
  const p = { lo: 0, hi: 60, pins: [{ at: 0, dir: -1 }, { at: 60, dir: 1 }] };
  const q = { lo: 20, hi: 100, pins: [{ at: 20, dir: -1 }, { at: 100, dir: 1 }] };
  // q's pin at 20 (to the left/up) is inside p: q goes before p; p's pin at 60 (to the right/down) is inside q: p after q
  assert.equal(tracks([p, q]), 2);
  assert.ok(q.track < p.track);
  const far = { lo: 200, hi: 300, pins: [] };
  assert.equal(tracks([p, far]), 1);  // apart: one track for both
});

test('redundant: the edges another path of blocking edges implies; 等子任务, review and subtask edges stay', () => {
  const e = (from, to, kind = 'input') => ({ from, to, kind, blocking: ['input', 'code', 'children'].includes(kind) });  // as the server sends them
  const ab = e('a', 'b'), bc = e('b', 'c'), ac = e('a', 'c'), cd = e('c', 'd'), ad = e('a', 'd');
  assert.deepEqual([...redundant([ab, bc, ac, cd, ad])], [ac, ad]);
  const bd = e('b', 'd');
  assert.deepEqual([...redundant([ab, bc, cd, bd])], [bd]);
  assert.deepEqual([...redundant([ab, e('b', 'a')])], []);         // a cycle: nothing else leads from a to b
  // a ↔ b with c → a and c → b: one of c's lines may go, never both
  const ca = e('c', 'a'), cb = e('c', 'b');
  assert.deepEqual([...redundant([ab, e('b', 'a'), ca, cb])], [ca]);
  assert.deepEqual([...redundant([e('a', 'b', 'code'), bc, e('a', 'c', 'code')])].map((x) => x.kind), ['code']);  // code baselines too
  // 等子任务 counts as a path but is not left out: it also waits for a's subtasks
  assert.deepEqual([...redundant([e('a', 'b', 'children'), bc, ac])], [ac]);
  assert.deepEqual([...redundant([ab, bc, e('a', 'c', 'children')])], []);
  for (const kind of ['review', 'subtask']) {
    assert.deepEqual([...redundant([e('a', 'b', kind), bc, ac])], []);  // no path (they do not order the tasks)
    assert.deepEqual([...redundant([ab, bc, e('a', 'c', kind)])], []);  // nor left out
  }
});

test('layout: the gap between two rows grows when more lines run across it than fit', () => {
  // four cards in a row (lanes a-d), each feeding e/u and e/v in lane e below: a fan of lines across one gap
  const lanes = ['a', 'b', 'c', 'd', 'e'];
  const top = lanes.slice(0, 4).map((l, i) => task(`${l}/t${i}`, at(1)));
  const below = ['e/u', 'e/v', 'e/w', 'e/x'].map((id, i) => task(id, at(2 + i)));
  const edges = top.map((t, i) => ({ from: t.id, to: 'e/u' })).concat(top.map((t) => ({ from: t.id, to: 'e/v' })));
  const L = layout([...top, ...below], edges, lanes, new Set());
  const gap = L.rows[1].y - (L.rows[0].y + L.rows[0].h);
  assert.ok(gap > 34, `gap ${gap}`);
  assert.ok(clear(edges, L.paths, L.pos));
});

test('two lines turning at the same place, one up and one down, do not run on top of each other', () => {
  // lanes a and b, one task a day: a0, b1, b2, a3; a0 → b2 turns down at b's middle, a3 → b1 turns up there
  const day = (d) => `2026-10-2${d}T01:00:00Z`;
  const tasks = [task('a/0', day(0)), task('b/1', day(1)), task('b/2', day(2)), task('a/3', day(3))];
  for (const edges of [[{ from: 'a/0', to: 'b/2' }, { from: 'a/3', to: 'b/1' }], [{ from: 'a/3', to: 'b/1' }, { from: 'a/0', to: 'b/2' }]]) {
    const L = layout(tasks, edges, ['a', 'b'], new Set());
    const verticals = L.paths.map((d) => segments(d).filter(([p, q]) => p.x === q.x && p.y !== q.y)
      .map(([p, q]) => ({ x: p.x, lo: Math.min(p.y, q.y), hi: Math.max(p.y, q.y) })));
    for (const v of verticals[0]) for (const w of verticals[1]) assert.ok(v.x !== w.x || v.hi <= w.lo || w.hi <= v.lo, JSON.stringify([v, w]));
  }
});

test('layout: lanes are columns, start days are rows, drafts go to the band at the bottom', () => {
  const tasks = [task('develop/a', '2026-10-20T01:00:00Z'), task('develop/b', '2026-10-21T01:00:00Z', 'running'),
    task('perf/c', '2026-10-21T02:00:00Z', 'queued'), task('develop/d', '', 'draft', { created: '2026-10-19T00:00:00Z' })];
  const edges = [{ from: 'develop/a', to: 'develop/b', kind: 'input' }, { from: 'develop/b', to: 'perf/c', kind: 'input' }];
  const L = layout(tasks, edges, ['develop', 'perf'], new Set(['develop/a']));
  const pos = (id) => L.pos.get(id);
  assert.equal(pos('develop/a').x, pos('develop/b').x);                 // same lane
  assert.ok(pos('perf/c').x > pos('develop/b').x);                        // next lane to the right
  assert.ok(pos('develop/a').y < pos('develop/b').y);                     // earlier day above
  assert.ok(pos('develop/b').y < pos('perf/c').y);                        // linked tasks never share a row
  assert.equal(pos('develop/a').h, 30);                                   // collapsed
  assert.ok(L.draftTop !== null && pos('develop/d').y > pos('perf/c').y);  // the draft is below every started task
  assert.deepEqual(L.rows.map((r) => r.draft), [false, false, false, true]);
});

test('layout: a template\'s first round runs from the top left to the bottom right, whatever the times say', () => {
  // made at once (one creation time); intake started a few seconds later, the rest wait for it
  const made = '2026-10-26T05:52:34Z';
  const tasks = ['plan/r1', 'profile/r1', 'baseline/r1', 'survey/r1'].map((id) => task(id, '', 'queued', { created: made }))
    .concat(task('intake/r1', '2026-10-26T05:52:37Z', 'needs_you', { created: made }));
  const edges = [['intake/r1', 'survey/r1'], ['intake/r1', 'baseline/r1'], ['baseline/r1', 'profile/r1'], ['profile/r1', 'plan/r1'],
    ['survey/r1', 'plan/r1'], ['intake/r1', 'plan/r1']].map(([from, to]) => ({ from, to, kind: 'input' }));
  const lanes = ['intake', 'survey', 'baseline', 'profile', 'plan'];
  const L = layout(tasks, edges, lanes, new Set());
  const row = (id) => L.pos.get(id).layer;
  assert.deepEqual(lanes.map((l) => row(`${l}/r1`)), [0, 1, 1, 2, 3]);
  assert.ok(lanes.every((l, i) => !i || L.pos.get(`${l}/r1`).x > L.pos.get(`${lanes[i - 1]}/r1`).x));
  // a cycle does not hang the layout
  const loop = layout(tasks, [...edges, { from: 'plan/r1', to: 'intake/r1', kind: 'input' }], lanes, new Set());
  assert.equal(loop.pos.size, 5);
  // a task that started before its upstream (re-run upstream) keeps its own time: the edge points up
  const rerun = [task('a/x', '2026-10-21T01:00:00Z'), task('b/y', '2026-10-20T01:00:00Z')];
  const R = layout(rerun, [{ from: 'a/x', to: 'b/y', kind: 'input' }], ['a', 'b'], new Set());
  assert.ok(R.pos.get('b/y').layer < R.pos.get('a/x').layer);
});

test('layout paths: one SVG path per drawn edge, from the upstream card to the downstream card', () => {
  const tasks = [task('develop/a', '2026-10-20T01:00:00Z'), task('develop/b', '2026-10-21T01:00:00Z', 'running'),
    task('develop/c', '2026-10-22T01:00:00Z', 'queued')];
  const edges = [{ from: 'develop/a', to: 'develop/b', kind: 'input' }, { from: 'develop/a', to: 'develop/c', kind: 'code' }];
  const L = layout(tasks, edges, ['develop'], new Set());
  const [ab, ac] = L.paths;
  const a = L.pos.get('develop/a'), b = L.pos.get('develop/b');
  assert.equal(ab, `M${a.x + a.w / 2},${a.y + a.h} L${a.x + a.w / 2},${b.y}`);  // next row, same lane: straight down
  assert.ok(ac.startsWith(`M${a.x + a.w},`));                                   // same lane, further down: round the right side
  // only the edges drawn get a path; the others still order the rows
  const one = layout(tasks, edges, ['develop'], new Set(), [edges[1]]);
  assert.equal(one.paths.length, 1);
  assert.deepEqual([...one.pos.values()].map((p) => p.layer), [...L.pos.values()].map((p) => p.layer));
});

test('laneAt: where a dragged lane title or card lands', () => {
  const L = layout([task('a/x', '2026-10-20T01:00:00Z'), task('b/y', '2026-10-20T02:00:00Z')], [], ['a', 'b', 'c'], new Set());
  const x0 = L.pos.get('a/x').x, x1 = L.pos.get('b/y').x;
  assert.equal(laneAt(x0 + 124, 3), 0);                    // the middle of a card in the first lane
  assert.equal(laneAt(x1 + 124, 3), 1);
  assert.equal(laneAt(x1 + 124 + 2 * (x1 - x0), 3), 2);    // past the last lane: the last one
  assert.equal(laneAt(-500, 3), 0);
});

test('a line into the first draft row turns in the gap above the drafts\' band, not across its title', () => {
  const tasks = [task('x/a', '2026-10-20T01:00:00Z'), task('x/b', '2026-10-21T01:00:00Z'),
    task('y/d', '', 'draft', { created: '2026-10-22T01:00:00Z' }), task('x/e', '', 'draft', { created: '2026-10-22T02:00:00Z' })];
  const edges = [{ from: 'x/a', to: 'y/d', kind: 'input' }, { from: 'x/b', to: 'x/e', kind: 'input' }];
  const L = layout(tasks, edges, ['x', 'y'], new Set());
  const above = L.rows.findLast((r) => !r.draft);
  assert.equal(L.pos.get('y/d').y - L.draftTop, DRAFT_BAND);
  for (const d of L.paths) {
    const runs = segments(d).filter(([u, v]) => u.y === v.y && u.x !== v.x);  // the last horizontal run goes into the draft's top
    if (!runs.length) continue;  // straight down
    const [p] = runs.at(-1);
    assert.ok(p.y > above.y + above.h && p.y < L.draftTop, `${d}: y ${p.y} not between ${above.y + above.h} and ${L.draftTop}`);
  }
});

test('two lines into one lane gap from the cards on either side, turning opposite ways lower down, do not run as one', () => {
  // b/L and c/R share a row, a lane apart; c/R → a/X turns left, b/L → d/Y right
  const day = (d) => `2026-10-2${d}T01:00:00Z`;
  const tasks = [task('b/L', day(0)), task('c/R', day(0)), task('a/M', day(1)), task('a/X', day(2)), task('d/Y', day(2))];
  const edges = [{ from: 'c/R', to: 'a/X', kind: 'input' }, { from: 'b/L', to: 'd/Y', kind: 'input' }];
  for (const order of [edges, [...edges].reverse()]) {
    const L = layout(tasks, order, ['a', 'b', 'c', 'd'], new Set());
    assert.equal(overlaps(L.paths), 0, L.paths.join('\n'));
  }
});
