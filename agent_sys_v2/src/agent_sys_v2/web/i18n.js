// The UI's language: agent_sys.yaml `language` (en, the default, or zh), which the server writes into <html lang> when it
// serves the page, so it is known before anything is drawn. Switching it saves the setting and reloads the page.
//
// Texts are written in Chinese, the language agent_sys was first written in, and looked up in EN (i18n_en.js) for
// English: t('等待上游任务：{tasks}', { tasks }). {name} is filled from params. An entry of EN may be a function of
// params, for plurals. A text missing from EN shows in Chinese; web/test/i18n.test.js checks that none is missing.
import { EN } from './i18n_en.js';

export const LANGS = { en: 'English', zh: '中文' };
export let lang = globalThis.document?.documentElement.lang.startsWith('zh') ? 'zh' : 'en';
export function setLang(l) { lang = l === 'zh' ? 'zh' : 'en'; }  // tests

export function t(text, params = {}) {
  const e = lang === 'en' ? EN[text] : undefined;
  const s = (typeof e === 'function' ? e(params) : e ?? text).replace(/\{(\w+)\}/g, (m, k) => (k in params ? String(params[k]) : m));
  // A sentence is often followed by the next piece on a new line of the template, which htm drops; Chinese needs no
  // space there, English does. A space at the end of a block collapses away.
  return e !== undefined && /[.:;?!]$/.test(s) ? `${s} ` : s;
}

// A list in the language's way: A、B (zh), A, B (en).
export const list = (items) => items.join(lang === 'en' ? ', ' : '、');
