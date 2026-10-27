// The English dictionary against the texts the UI uses: every t('…') key has English with the same {placeholders},
// and no Chinese is left outside t() (comments and lines marked `// i18n: <why>` aside).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import { EN } from '../i18n_en.js';
import { t, setLang, lang } from '../i18n.js';

const WEB = new URL('../', import.meta.url);
const files = readdirSync(WEB).filter((f) => f.endsWith('.js') && !f.startsWith('i18n'));
// boot.js is a plain script that cannot import: it has its own two texts per message
const SCANNED = files.filter((f) => f !== 'boot.js');
const source = Object.fromEntries(files.map((f) => [f, readFileSync(new URL(f, WEB), 'utf8')]));
const KEY = /\b(?:t|tr)\(\s*'((?:[^'\\]|\\.)*)'/g;  // board.js, run.js and task.js import t as tr (t is a task there)
const keysOf = (text) => [...text.matchAll(KEY)].map((m) => eval(`'${m[1]}'`));  // the literal as JS reads it
const holes = (s) => [...s.matchAll(/\{(\w+)\}/g)].map((m) => m[1]).sort().join(',');
const CJK = /[　-〿一-鿿＀-￯]/;

test('the page is in English unless <html lang> says zh', () => {
  assert.equal(lang, 'en');  // no document here: the default
});

test('every t() key has English, with the same placeholders', () => {
  const missing = [], holesDiffer = [];
  for (const [f, text] of Object.entries(source)) {
    for (const key of keysOf(text)) {
      const en = EN[key];
      if (en === undefined) missing.push(`${f}: ${key}`);
      else if (typeof en === 'string' && holes(en) !== holes(key)) holesDiffer.push(`${f}: ${key} -> ${en}`);
    }
  }
  assert.deepEqual(missing, []);
  assert.deepEqual(holesDiffer, []);
});

test('the dictionary has no entry the UI does not use', () => {
  const used = new Set(Object.values(source).flatMap(keysOf));
  // texts looked up through a variable: labels in constants, translated when shown
  assert.deepEqual(Object.keys(EN).filter((k) => !used.has(k) && !DYNAMIC.has(k)), []);
  for (const k of DYNAMIC) assert.ok(k in EN, k);
});
const DYNAMIC = new Set(['全部日期', '近 3 天', '近 7 天', '近 30 天', '自定义日期…', '正常结束', '已停止', '运行中断', '预算用尽', '转为终端接管', '已退出终端',
  '出错', '预算已用尽', '注意事项', '环境信息', '方法', '已验证无效', '全局经验库', '项目经验库']);

test('no Chinese is left outside t()', () => {
  const left = [];
  for (const f of SCANNED) {
    const code = source[f].replace(/\/\*[\s\S]*?\*\//g, '');
    code.split('\n').forEach((line, i) => {
      if (/\/\/ i18n: /.test(line)) return;  // marked: both languages, not shown, or translated where shown
      const bare = line.replace(KEY, 't(').replace(/(^|[^:'"`])\/\/.*$/, '$1');
      if (CJK.test(bare)) left.push(`${f}:${i + 1}: ${line.trim().slice(0, 100)}`);
    });
  }
  assert.deepEqual(left, []);
});

test('t() fills placeholders, falls back to the Chinese, and takes plural functions', () => {
  setLang('en');
  assert.equal(t('没有这条译文 {x}', { x: 1 }), '没有这条译文 1');
  const fn = Object.entries(EN).find(([, v]) => typeof v === 'function');
  if (fn) assert.equal(typeof t(fn[0], { n: 1 }), 'string');
  setLang('zh');
  assert.equal(t('没有这条译文 {x}', { x: 2 }), '没有这条译文 2');
  setLang('en');
});
