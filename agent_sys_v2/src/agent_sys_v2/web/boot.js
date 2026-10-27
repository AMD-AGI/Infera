// Runs before the app, as a plain script that old browsers can parse too: applies the saved theme (dark when none is
// saved; auto: the system's, followed when it changes) before the first paint, and if the app's scripts fail or never
// start (old browser, lost connection), shows what went wrong on the page instead of leaving it blank.
(function () {
  var system = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)');
  var chosen = 'dark';  // storage blocked: the default
  try { chosen = localStorage.getItem('agent_sys-theme') || 'dark'; } catch (e) { /* the default */ }
  // also the theme switch's (app.js)
  window.applyTheme = function (theme) {
    chosen = theme || chosen;
    var dark = chosen === 'auto' ? !system || system.matches : chosen !== 'light';
    document.documentElement.setAttribute('data-theme', dark ? 'dark' : 'light');
  };
  window.applyTheme();
  if (system && system.addEventListener) system.addEventListener('change', function () { window.applyTheme(); });

  function show(text) {
    var box = document.getElementById('boot-error');
    if (!box) {
      if (!document.body) return;
      box = document.createElement('div');
      box.id = 'boot-error';
      box.setAttribute('role', 'alert');
      box.style.cssText = 'position:fixed;z-index:99;left:16px;right:16px;top:16px;padding:12px 16px;border-radius:8px;' +
        'background:#fdecea;color:#7a1a12;border:1px solid #e3a59e;font:14px/1.6 system-ui,sans-serif;white-space:pre-wrap';
      document.body.appendChild(box);
    }
    box.textContent += (box.textContent ? '\n' : '') + text;
  }
  // the page's language (the server writes agent_sys.yaml's into <html lang>); this script cannot import i18n.js
  var zh = /^zh/.test(document.documentElement.getAttribute('lang') || '');
  var hint = zh ? '浏览器：' + navigator.userAgent + '\n需要较新版本的浏览器（Chrome / Edge 80+、Firefox 98+、Safari 15.4+）。如需排查，请提供以上信息。'
    : 'Browser: ' + navigator.userAgent + '\nagent_sys needs a recent browser (Chrome / Edge 80+, Firefox 98+, Safari 15.4+). Include the above when reporting the problem.';
  if (zh) {  // index.html's "loading" line is in English; the app replaces it once it starts
    document.addEventListener('DOMContentLoaded', function () {
      var line = document.querySelector('#app .boot p');
      if (line) line.textContent = 'agent_sys 正在加载…';
    });
  }

  window.addEventListener('error', function (e) {
    var file = e.filename || '';
    if (file && file.indexOf(location.origin) !== 0) return;  // browser extensions
    if (/ResizeObserver/.test(e.message || '')) return;
    var where = file ? file.split('/').pop() + ':' + e.lineno : '';
    show((zh ? '页面脚本出错：' : 'Page script error: ') + (e.message || e) + (where ? (zh ? '（' + where + '）' : ' (' + where + ')') : '') +
      (window.appMounted ? '' : '\n' + hint));
  });
  setTimeout(function () {
    if (!window.appMounted) {
      show((zh ? '页面脚本 15 秒内未能启动（网络过慢或浏览器不支持）。' : 'The page scripts did not start within 15 seconds (slow network or unsupported browser).') + '\n' + hint);
    }
  }, 15000);
})();
