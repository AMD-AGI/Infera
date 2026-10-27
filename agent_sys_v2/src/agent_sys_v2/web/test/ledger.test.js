// The ledger's chips on the board's head (ledger.js): what they say of the server's baseline and best.  node --test web/test/*.test.js
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { LedgerChips } from '../ledger.js';
import { setLang } from '../i18n.js';

setLang('zh');

// The text a component shows, its function components rendered on the way.
const text = (v) => (v == null || typeof v === 'boolean' ? '' : Array.isArray(v) ? v.map(text).join('')
  : typeof v !== 'object' ? String(v) : typeof v.type === 'function' ? text(v.type(v.props)) : text(v.props.children));
// one string per chip (the page sets them apart)
const chips = (ledger) => [LedgerChips({ pid: 'p', ledger }).props.children].flat().map(text).filter(Boolean).join(' | ');

test('the chips name the best once, and keep its number apart from its delta', () => {
  const base = { task: 'baseline/r1', file: 'results/c64.json', value: 12300, delta_pct: 0, gain: 0 };
  // a project's first result is its baseline and its best: shown once
  assert.equal(chips({ count: 1, primary: 'tok_s_gpu', baseline: base, best: { ...base } }), 'tok_s_gpu | 基线 12.3K');
  const best = { task: 'experiment/a', file: 'results/c64.json', value: 11500, delta_pct: -6.5, gain: 6.5 };
  assert.equal(chips({ count: 2, primary: 'p90_ttft', baseline: base, best }), 'p90_ttft | 基线 12.3K | 最佳 11.5K −6.50%');
  // at the working point: the series' names
  const wp = { count: 9, primary: 'y', curve_head: true, baseline: { series: 'v0', value: 18000 }, best: { series: 'r1-spf', value: 19800, delta_pct: 10, gain: 10 } };
  assert.equal(chips(wp), 'y | 基线 v0 18K | 最佳 r1-spf 19.8K +10.0%');
});
