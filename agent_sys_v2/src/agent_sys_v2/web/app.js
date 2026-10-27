// agent_sys web UI: routes, the app bar, the home page (projects, 新建任务 and what runs on top; below, everything that
// needs you, across projects) and the settings / archived-projects pages. The board is in board.js, the ledger in ledger.js, the task drawer in
// task.js, the rules and knowledge pages in knowledge.js.
import { render } from './vendor/preact.module.js';
import { useState, useEffect } from './vendor/hooks.module.js';
import { api } from './api.js';
import { html, Icon, StatusMark, statusLine, useApi, useNow, money, ago, since, clock, ErrorBox, Toasts, ConfirmHost, toast,
  fail, Loading, Holder, refill, parseHash, hrefWith, go, taskHref, ModelSelect, EffortSelect, UltracodeCheck, machinesText,
  parseMachines } from './ui.js';
import { t, lang, LANGS } from './i18n.js';
import { BoardPage, ArchiveButton } from './board.js';
import { TaskDrawer, NewTaskDrawer } from './task.js';
import { NeedsCard, ReviewCard, MarkAllRead, resumeAll, interruptedOf } from './cards.js';
import { RulesPage, KnowledgePage } from './knowledge.js';
import { LedgerPage } from './ledger.js';

// ---------------------------------------------------------------- shell

function useHash() {
  const [h, set] = useState(parseHash());
  useEffect(() => { const f = () => set(parseHash()); addEventListener('hashchange', f); return () => removeEventListener('hashchange', f); }, []);
  return h;
}

// Archived projects are not in home.projects, but their pages still need the title.
const projectOf = (home, pid) => home && [...home.projects, ...home.archived_projects].find((p) => p.id === pid);

function App() {
  const { path, params } = useHash();
  const home = useApi('/api/home');
  const [conn, setConn] = useState(true);
  useEffect(() => api.onChange((c) => { if (c.type === 'conn') setConn(c.state === 'open'); }), []);
  // the tab title counts what needs you: 待确认 + 待审核
  const n = home.data ? home.data.needs_you.length + home.data.in_review.length : 0;
  useEffect(() => { document.title = n ? `(${n}) agent_sys` : 'agent_sys'; }, [n]);
  const m = path.match(/^\/p\/([^/]+)(?:\/(ledger|rules|knowledge|settings))?$/);
  const g = path.match(/^\/(rules|knowledge|settings|archived)$/);
  const pid = m?.[1] || null;
  const page = m ? m[2] || 'board' : g ? g[1] : 'home';
  const [tp, ...rest] = (params.t || '').split('/');
  const openTask = params.t && rest.length === 2 ? { pid: tp, tid: rest.join('/') } : null;
  // ?new on a project's pages, ?new=<project> on the home page (its 新建任务 box): the same drawer
  const newTask = !params.new ? null : pid || (page === 'home' && home.data?.projects.some((p) => p.id === params.new) ? params.new : null);
  useEffect(() => { if (pid) stored.set('agent_sys-project', pid); }, [pid]);
  useEffect(() => {
    const key = (ev) => {
      if (/INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName) || ev.ctrlKey || ev.metaKey || ev.altKey) return;
      if (ev.key === 'n' && pid && !projectOf(home.data, pid)?.archived) { ev.preventDefault(); go(newTaskHref(pid)); }
      const hp = ev.key === 'n' && page === 'home' && home.data && homeProject(home.data.projects);
      if (hp) { ev.preventDefault(); go(newTaskHref(hp)); }
      if (ev.key === 'g') { ev.preventDefault(); go('#/'); }
    };
    addEventListener('keydown', key);
    return () => removeEventListener('keydown', key);
  }, [pid, page, home.data]);
  // the hash is the route: the skip link moves the focus itself (tabIndex: every page's <main> takes it)
  const skip = (e) => { e.preventDefault(); const m = document.getElementById('main'); if (m) { m.tabIndex = -1; m.focus(); } };
  return html`<a class="skip" href="#main" onClick=${skip}>${t('跳转到主要内容')}</a>
    <${AppBar} home=${home.data} pid=${pid} page=${page} conn=${conn} />
    ${!conn && html`<div class="conn-banner" role="status"><${Icon} name="warning" />${t('实时连接已断开，正在重新连接。页面显示的是断开前的数据，重新连接后将自动刷新。')}</div>`}
    <div class=${`layout ${openTask || newTask ? 'with-drawer' : ''}`}>
      ${page === 'home' && html`<${HomePage} home=${home} drawerPid=${newTask} />`}
      ${page === 'board' && html`<${BoardPage} key=${pid} pid=${pid} onNewTask=${() => go(newTaskHref(pid))} />`}
      ${page === 'ledger' && html`<${LedgerPage} key=${pid} pid=${pid} />`}
      ${page === 'rules' && html`<${RulesPage} key=${pid} pid=${pid} />`}
      ${page === 'knowledge' && html`<${KnowledgePage} key=${pid} pid=${pid} />`}
      ${page === 'settings' && html`<${SettingsPage} key=${pid} pid=${pid} project=${projectOf(home.data, pid)} />`}
      ${page === 'archived' && html`<${ArchivedPage} home=${home} />`}
      ${openTask && html`<${TaskDrawer} key=${`${openTask.pid}/${openTask.tid}`} pid=${openTask.pid} tid=${openTask.tid} tab=${params.tab} file=${params.file} />`}
      ${newTask && !openTask && html`<${NewTaskDrawer} key=${newTask} pid=${newTask} />`}
    </div>
    <${Toasts} /><${ConfirmHost} />`;
}

function AppBar({ home, pid, page, conn }) {
  const project = projectOf(home, pid);
  const n = home ? home.needs_you.length + home.in_review.length : 0;
  const nav = (key, label) => html`<a href=${`#/p/${pid}${key === 'board' ? '' : `/${key}`}`} aria-current=${page === key ? 'page' : undefined}>${label}</a>`;
  return html`<header class="appbar">
    <a class="brand" href="#/" aria-label=${t('agent_sys 首页')}><${Icon} name="logo" size=${22} /><span>agent_sys</span></a>
    ${pid && html`<span class="crumb" aria-hidden="true">/</span><a class="crumb-project" href=${`#/p/${pid}`}>${project?.title || pid}</a>
      ${project?.archived && html`<span class="tag">${t('已归档')}</span>`}
      <nav class="tabs-nav" aria-label=${t('项目')}>${nav('board', t('看板'))}${nav('ledger', t('台账'))}${nav('rules', t('规则'))}${nav('knowledge', t('经验'))}${nav('settings', t('设置'))}</nav>`}
    ${!pid && html`<nav class="tabs-nav" aria-label=${t('全局')}><a href="#/" aria-current=${page === 'home' ? 'page' : undefined}>${t('首页')}</a>
      <a href="#/rules" aria-current=${page === 'rules' ? 'page' : undefined}>${t('全局规则')}</a><a href="#/knowledge" aria-current=${page === 'knowledge' ? 'page' : undefined}>${t('全局经验')}</a>
      <a href="#/settings" aria-current=${page === 'settings' ? 'page' : undefined}>${t('设置')}</a></nav>`}
    <span class="spacer"></span>
    ${home && html`<a class=${`needs-pill ${n ? 'on' : ''}`} href="#/" title=${t('待确认和待审核的任务（不含已归档项目）')}><${Icon} name="needs_you" />${n ? t('{n} 项待处理', { n }) : t('暂无待处理事项')}</a>
      <span class="muted small" title=${t('今天开始的运行的花费合计')}>${t('今日花费 {cost}', { cost: money(home.cost_today_usd) })}</span>`}
    <span class=${`conn ${conn ? 'on' : 'off'}`} title=${conn ? t('实时连接正常') : t('实时连接已断开，正在重新连接')}><span class="dot" aria-hidden="true"></span><span class="sr-only">${conn ? t('实时连接正常') : t('实时连接已断开')}</span></span>
    <${LangToggle} />
    <${ThemeToggle} />
  </header>`;
}

// localStorage can be blocked (embedded browsers, strict privacy settings): then the theme is not remembered.
// Dark is the default; 跟随系统 follows the system's.
const stored = { get: (k) => { try { return localStorage.getItem(k); } catch { return null; } }, set: (k, v) => { try { localStorage.setItem(k, v); } catch { /* not remembered */ } } };

// ?new=<project>: the new-task drawer, on the home page and on the project's pages
const newTaskHref = (pid) => hrefWith({ new: pid, t: null, tab: null, file: null });

// The project the home page's 新建任务 starts in: the one opened last (a project's pages, or picked in the box),
// else the first of the list (the only one, often).
const homeProject = (projects) => {
  const ok = projects.filter((p) => !p.error);
  return (ok.find((p) => p.id === stored.get('agent_sys-project')) || ok[0])?.id;
};

function ThemeToggle() {
  const [theme, set] = useState(stored.get('agent_sys-theme') || 'dark');
  useEffect(() => {
    stored.set('agent_sys-theme', theme);
    window.applyTheme(theme);  // boot.js: auto follows the system's
  }, [theme]);
  const next = { auto: 'light', light: 'dark', dark: 'auto' }[theme];
  const label = { auto: t('跟随系统'), light: t('亮色'), dark: t('暗色') };
  return html`<button class="btn icon-only ghost" onClick=${() => set(next)} title=${t('主题：{theme}（点击切换为{next}）', { theme: label[theme], next: label[next] })} aria-label=${t('主题：{theme}', { theme: label[theme] })}>
    <${Icon} name=${{ auto: 'auto', light: 'sun', dark: 'moon' }[theme]} /></button>`;
}

// The language is agent_sys.yaml's (the server's messages follow it too): saved, then the page is loaded again in it.
const switchLang = (to) => api.put('/api/settings', { scope: 'global', values: { language: to } })
  .then(() => location.reload(), fail);

function LangToggle() {
  const other = lang === 'en' ? 'zh' : 'en';
  return html`<button class="btn ghost small lang-toggle" onClick=${() => switchLang(other)} lang=${other}
    title=${other === 'zh' ? '切换为中文（Switch to Chinese）' : 'Switch to English（切换为英文）'}>${LANGS[other]}</button>`;  // i18n: both
}

// ---------------------------------------------------------------- home

// What a project sets for its new tasks, in one line each.
const autorunText = (on) => (on ? t('自动开始：开启') : t('自动开始：关闭'));
const agentText = (a) => t('新建任务默认：模型 {model}，effort {effort}，预算 ${budget}',
  { model: a.model, effort: a.effort + (a.ultracode ? ' + ultracode' : ''), budget: a.budget_usd });

function HomePage({ home, drawerPid }) {
  const { data, error, reload } = home;
  const now = useNow(30000);
  if (error && !data) return html`<main id="main" tabIndex="-1" class="page"><${ErrorBox} what=${t('加载首页')} error=${error} retry=${reload} /></main>`;
  if (!data) return html`<${Loading} />`;
  const interrupted = interruptedOf(data.needs_you);
  const archived = data.archived_projects.length;
  if (!data.projects.length) return html`<main id="main" tabIndex="-1" class="page home">
    <section class="welcome"><h1>${t('暂无项目')}</h1><p>${t('agent_sys 以项目为单位管理任务：每个项目对应 projects/ 下的一个目录，任务按类型分泳道显示。请先新建一个项目。')}</p>
      ${archived > 0 && html`<p class="muted">${t('另有 {n} 个已归档项目，', { n: archived })}<a href="#/archived">${t('查看已归档项目')}</a>${t('。')}</p>`}
      <${NewProject} templates=${data.templates} root=${data.projects_path} first /></section></main>`;
  const queues = [
    { key: 'needs', icon: 'needs_you', label: t('待确认'), n: data.needs_you.length, hint: t('agent 提问、运行中断或预算用尽时，任务会出现在这里。') },
    { key: 'review', icon: 'in_review', label: t('待审核'), n: data.in_review.length, hint: t('设置了"完成后需审核"的任务完成后会出现在这里，审核通过后下游任务才会开始。') },
    { key: 'unread', icon: 'done', label: t('最近完成'), n: data.unread.length, hint: t('完成后还没查看的任务，查看后移出列表。') },
  ];
  const quiet = queues.filter((q) => !q.n);
  // an empty queue is a chip in one line (its hint as the tooltip), not a heading over a sentence
  const chips = html`<span class="quiet-chips">${quiet.map((q) => html`<span class=${`quiet-chip qc-${q.key}`} title=${q.hint}>
    <${Icon} name=${q.icon} />${q.label} <span class="count">0</span></span>`)}</span>`;
  return html`<main id="main" tabIndex="-1" class="page home">
    <h1 class="sr-only">${t('agent_sys 首页')}</h1>
    ${error && html`<${ErrorBox} what=${t('刷新首页')} error=${error} retry=${reload} />`}
    <div class="home-top">
      <${ProjectsCard} data=${data} />
      <${NewTaskCard} projects=${data.projects} drawerPid=${drawerPid} />
      <${RunningSummary} data=${data} now=${now} />
    </div>
    ${data.machines.length > 0 && html`<${HeldMachines} machines=${data.machines} now=${now} />`}
    <div class="home-queues">
      ${quiet.length === queues.length ? html`<div class="queues-quiet" role="status"><${Icon} name="done" className="quiet-ok" />
          <b>${t('暂无待处理事项')}</b><span class="muted small">${t('需要你回复、审核或查看的任务会出现在这里。')}</span>${chips}</div>`
        : html`
        ${data.needs_you.length > 0 && html`<section class="queue" aria-labelledby="h-needs">
          <div class="section-head"><h2 id="h-needs"><${Icon} name="needs_you" />${t('待确认')} <span class="count">${data.needs_you.length}</span></h2>
            ${interrupted.length >= 2 && html`<button class="btn primary" onClick=${() => resumeAll(interrupted)}>
              ${t('继续全部中断任务（{n}）', { n: interrupted.length })}</button>`}</div>
          ${data.needs_you.map((x) => html`<${NeedsCard} key=${x.project + x.id} t=${x} now=${now} />`)}
        </section>`}
        ${data.in_review.length > 0 && html`<section class="queue" aria-labelledby="h-review">
          <div class="section-head"><h2 id="h-review"><${Icon} name="in_review" />${t('待审核')} <span class="count">${data.in_review.length}</span></h2></div>
          ${data.in_review.map((x) => html`<${ReviewCard} key=${x.project + x.id} t=${x} now=${now} />`)}
        </section>`}
        ${data.unread.length > 0 && html`<section class="queue" aria-labelledby="h-unread">
          <div class="section-head"><h2 id="h-unread"><${Icon} name="done" />${t('最近完成')} <span class="count">${data.unread.length}</span><span class="muted small">${t('不阻塞下游，查看后移出列表')}</span></h2>
            <span class="spacer"></span><${MarkAllRead} tasks=${data.unread} /></div>
          <ul class="unread-list">${data.unread.map((x) => html`<li>
            <a href=${taskHref(x.project, x.id, { tab: 'docs' })}><${Icon} name="unread" className="unread-dot" /><b>${x.title}</b></a>
            <span class="muted small">${x.project} · ${ago(x.since, now)}</span><span class="ellipsis small">${x.last_message}</span>
            <a class="btn small" href=${taskHref(x.project, x.id, { tab: 'docs' })}>${t('查看报告')}</a></li>`)}</ul>
        </section>`}
        ${quiet.length > 0 && html`<p class="queues-quiet small"><span class="muted">${t('其他队列暂无事项：')}</span>${chips}</p>`}`}
    </div>
  </main>`;
}

// The projects, one compact line each (scrolls past about six; a filter from nine on). 新建项目 is used seldom: it
// takes the card's place behind a button in the card's head.
function ProjectsCard({ data }) {
  const [adding, setAdding] = useState(false);
  const [q, setQ] = useState('');
  const archived = data.archived_projects.length;
  const shown = q ? data.projects.filter((p) => `${p.title} ${p.id}`.toLowerCase().includes(q.trim().toLowerCase())) : data.projects;
  return html`<section class="side-card home-card projects-card" aria-labelledby="h-projects">
    <div class="card-head"><h2 id="h-projects">${t('项目列表')} <span class="count">${data.projects.length}</span></h2><span class="spacer"></span>
      ${!adding && data.projects.length > 8 && html`<input type="search" class="proj-filter" value=${q} onInput=${(e) => setQ(e.target.value)}
        placeholder=${t('筛选项目')} aria-label=${t('筛选项目')} />`}
      <button class="btn small ghost" aria-expanded=${adding} onClick=${() => setAdding(!adding)}>
        <${Icon} name=${adding ? 'x' : 'plus'} />${adding ? t('取消') : t('新建项目')}</button></div>
    ${adding ? html`<div class="card-body"><${NewProject} templates=${data.templates} root=${data.projects_path} first /></div>` : html`
      <ul class="projects card-body">${shown.map((p) => p.error ? html`<li class="proj"><b class="ellipsis">${p.title}</b>
          <span class="error-text small proj-sub">${p.error}</span></li>` : html`<li class="proj">
        <a class="proj-name" href=${`#/p/${p.id}`}><b class="ellipsis">${p.title}</b>${p.title !== p.id && html`<span class="mono small muted ellipsis">${p.id}</span>`}</a>
        <span class="proj-sub counts small">
          ${p.template_review && html`<a class="st-in_review" href=${`#/p/${p.id}`} title=${t('模板预置的任务等你审核后才开始')}><${Icon} name="in_review" />${t('待审核模板')}</a>`}
          ${p.counts.needs_you > 0 && html`<span class="st-needs_you"><${Icon} name="needs_you" />${t('{n} 项待确认', { n: p.counts.needs_you })}</span>`}
          ${p.counts.in_review > 0 && html`<span class="st-in_review"><${Icon} name="in_review" />${t('{n} 项待审核', { n: p.counts.in_review })}</span>`}
          ${p.counts.running > 0 && html`<span class="st-running"><${Icon} name="running" />${t('{n} 个运行中', { n: p.counts.running })}</span>`}
          <span class="muted">${t('共 {n} 个任务，今日花费 {cost}', { n: p.counts.total, cost: money(p.cost_today_usd) })}</span></span>
        <span class="proj-actions"><${ArchiveButton} project=${p} /><a class="btn small icon-only ghost" href=${newTaskHref(p.id)}
          onClick=${() => stored.set('agent_sys-project', p.id)} title=${t('在 {project} 中新建任务', { project: p.title })}
          aria-label=${t('在 {project} 中新建任务', { project: p.title })}><${Icon} name="plus" /></a></span></li>`)}
        ${!shown.length && html`<li class="empty-line">${t('没有匹配的项目。')}</li>`}</ul>
      ${archived > 0 && html`<a class="archived-link small" href="#/archived"><${Icon} name="archive" />${t('已归档项目（{n}）', { n: archived })}</a>`}`}
  </section>`;
}

// 新建任务 from the home page: pick a project (the one opened last), then its NewTaskDrawer opens over this page.
// While the drawer is open the box shows its project; picking another one moves the drawer there.
function NewTaskCard({ projects, drawerPid }) {
  const ok = projects.filter((p) => !p.error);
  const [pick, setPick] = useState(() => homeProject(projects));
  useEffect(() => { if (drawerPid) setPick(drawerPid); }, [drawerPid]);
  const p = ok.find((x) => x.id === (drawerPid || pick)) || ok.find((x) => x.id === homeProject(projects));
  const choose = (id) => { setPick(id); stored.set('agent_sys-project', id); if (drawerPid) go(newTaskHref(id)); };
  return html`<section class="side-card home-card new-task-card" aria-labelledby="h-new-task">
    <div class="card-head"><h2 id="h-new-task"><${Icon} name="plus" />${t('新建任务')}</h2></div>
    <label class="field">${t('项目')}<select value=${p?.id} onChange=${(e) => choose(e.target.value)} disabled=${!ok.length}>
      ${ok.map((x) => html`<option value=${x.id}>${x.title === x.id ? x.id : `${x.title} (${x.id})`}</option>`)}</select></label>
    ${p && html`<div class="muted small nt-defaults">
      <div class="nt-flags"><span>${p.template ? t('模板 {template}', { template: p.template }) : t('无模板')}</span><span>${autorunText(p.autorun)}</span>
        ${p.template_review && html`<span class="st-in_review">${t('待审核模板')}</span>`}</div>
      <div>${agentText(p.agent)}</div></div>`}
    <a class=${`btn primary nt-go ${p ? '' : 'disabled'}`} href=${p ? newTaskHref(p.id) : undefined} aria-disabled=${!p}
      title=${t('新建任务（快捷键 n）')}><${Icon} name="plus" />${t('新建任务')}<span class="kbd">n</span></a>
    <p class="muted small nt-hint">${t('打开新建任务表单：选类型、写 prompt、选上游任务，也可以先存为草稿。')}</p>
  </section>`;
}

function RunningSummary({ data, now }) {
  const groups = new Map();
  // where today's cost went: the projects that spent most (fills the card while nothing runs)
  const spent = data.projects.filter((p) => p.cost_today_usd > 0).sort((a, b) => b.cost_today_usd - a.cost_today_usd).slice(0, 4);
  for (const x of data.running) { if (!groups.has(x.project)) groups.set(x.project, []); groups.get(x.project).push(x); }
  return html`<section class="side-card home-card running-card" aria-labelledby="h-running">
    <div class="card-head"><h2 id="h-running"><${Icon} name="running" className=${data.running.length ? 'busy' : ''} />${t('运行中')} <span class="count">${data.running.length}</span>
      ${data.queued > 0 && html`<span class="muted small">${t('待开始 {n}', { n: data.queued })}</span>`}</h2></div>
    <div class="card-body">${[...groups].map(([p, list]) => html`<div class="run-group"><a class="small" href=${`#/p/${p}`}>${data.projects.find((x) => x.id === p)?.title || p}</a>
      <ul>${list.map((x) => html`<li><a href=${taskHref(p, x.id, { tab: x.managed ? 'run' : 'docs' })}><${StatusMark} t=${x} short /><span class="ellipsis">${x.title}</span></a>
        <span class="muted small nowrap">${x.managed ? since(x.since, now) : ''}</span>
        <span class="muted small ellipsis">${statusLine(x)}</span></li>`)}</ul></div>`)}
    ${!data.running.length && html`<p class="empty-line">${t('暂无运行中的任务。')}</p>`}</div>
    <div class="muted small cost-line"><div>${t('今日花费')} <b>${money(data.cost_today_usd)}</b></div>
      ${spent.length > 0 && html`<ul class="cost-list">${spent.map((p) => html`<li><a class="ellipsis" href=${`#/p/${p.id}`}>${p.title}</a>
        <span class="nowrap">${money(p.cost_today_usd)}</span></li>`)}</ul>`}</div>
  </section>`;
}

// The machines held now, across projects: by whom, and which holders wait for you meanwhile (闲置占用).
function HeldMachines({ machines, now }) {
  const idle = machines.filter((m) => m.holders.some((h) => h.idle_minutes > 0)).length;
  return html`<section class="side-card held-card" aria-labelledby="h-machines">
    <h2 id="h-machines"><${Icon} name="machine" />${t('机器占用')} <span class="count">${machines.length}</span>
      ${idle > 0 && html`<span class="warn-text small">${t('{n} 台闲置占用', { n: idle })}</span>`}</h2>
    <ul class="held">${machines.flatMap(({ name, holders }) => holders.map((h) => html`<li>
      <span class="mono">${name}</span><a href=${taskHref(h.project, h.task)}><${Holder} h=${h} now=${now} /></a>
      <span class="muted small nowrap">${h.project}</span></li>`))}</ul></section>`;
}

function NewProject({ templates, root, first }) {
  const [tmpl, setTmpl] = useState(templates[0]?.id);
  const [name, setName] = useState('');
  const [busy, setBusy] = useState(false);
  const err = name && !/^[A-Za-z0-9][\w-]*$/.test(name) ? t('只能包含字母、数字、_ 和 -，且须以字母或数字开头') : '';
  const create = (ev) => {
    ev.preventDefault();
    if (!name || err) return;
    setBusy(true);
    api.post('/api/projects', { name, template: tmpl }).then((p) => {
      toast(p.template_review ? t('已创建项目 {id}：模板预置的任务等你审核后再开始', { id: p.id }) : t('已创建项目 {id}', { id: p.id }));
      go(`#/p/${p.id}`);
    }, fail).finally(() => setBusy(false));
  };
  return html`<form class="new-project" onSubmit=${create}>
    <div class="tmpl-list" role="radiogroup" aria-label=${t('模板')}>${templates.map((x) => html`<label class=${`tmpl ${tmpl === x.id ? 'on' : ''}`}>
      <input type="radio" name="tmpl" value=${x.id} checked=${tmpl === x.id} onChange=${() => setTmpl(x.id)} />
      <b>${x.title}</b><span class="small muted">${x.description}</span><span class="small mono">${x.types.join(' · ')}</span>
      ${x.tasks?.length > 0 && html`<span class="small" title=${t('建项目时一起创建，等你审核后点"开始运行"；之后按依赖依次开始')}>${t('自带 {n} 个任务：', { n: x.tasks.length })}<span class="mono">${x.tasks.join(' · ')}</span></span>`}</label>`)}</div>
    <label class="field">${t('项目名称')}<input class="mono" value=${name} onInput=${(e) => setName(e.target.value)} placeholder=${t('例如 infera-perf-q4')} aria-invalid=${!!err} autofocus=${first} />
      ${err && html`<span class="error-text small">${err}</span>`}<span class="muted small">${t('将创建在')} <code>${root}/${name || t('<项目名称>')}/</code></span></label>
    <button class="btn primary" type="submit" disabled=${busy || !name || !!err}>${t('创建项目')}</button>
  </form>`;
}

// ---------------------------------------------------------------- archived projects

function ArchivedPage({ home }) {
  const { data, error, reload } = home;
  if (error && !data) return html`<main id="main" tabIndex="-1" class="page"><${ErrorBox} what=${t('加载已归档项目')} error=${error} retry=${reload} /></main>`;
  if (!data) return html`<${Loading} />`;
  const list = data.archived_projects;
  return html`<main id="main" tabIndex="-1" class="page archived-page">
    <header class="page-head"><div><h1>${t('已归档项目')}</h1>
      <p class="muted">${t('已归档的项目不显示在首页，其中的任务不计入待处理事项，调度器也不会自动开始这些任务。取消归档后恢复正常。')}</p></div></header>
    ${list.length ? html`<ul class="archived-list">${list.map((p) => html`<li>
        <div><a href=${`#/p/${p.id}`}><b>${p.title}</b><span class="mono small muted">${p.id}</span></a>
          <div class="small muted">${t('归档于 {time}；共 {n} 个任务，已完成 {done} 个', { time: clock(p.archived_at), n: p.counts.total, done: p.counts.done })}</div></div>
        <${ArchiveButton} project=${p} /></li>`)}</ul>`
      : html`<p class="empty-line">${t('暂无已归档项目。可在首页的项目列表或项目设置中归档项目。')}</p>`}
  </main>`;
}

// ---------------------------------------------------------------- settings

// The project form as filled from project.yaml's values.
const projectForm = (v) => v && { ...v, goal: v.goal || '', agent: { ...v.agent },
  ledger: { primary: '', higher_is_better: true, ...(v.ledger || {}) }, machinesText: machinesText(v.machines) };

function SettingsPage({ pid, project }) {
  const { data, error, reload } = useApi(`/api/settings${pid ? `?project=${pid}` : ''}`, (c) => !c.task);
  // the page reloads on any change of the project (or after a while hidden): it must not undo what you typed
  const [ps, setPs] = useState(null);
  const [gs, setGs] = useState(null);
  useEffect(() => {
    if (!data) return;
    setPs((s) => refill(s, projectForm(data.project?.values)));
    setGs((s) => refill(s, { ...data.global.values }));
  }, [data]);
  if (error && !data) return html`<main id="main" tabIndex="-1" class="page"><${ErrorBox} what=${t('加载设置')} error=${error} retry=${reload} /></main>`;
  if (!data || !gs) return html`<${Loading} />`;
  const p = ps?.value, g = gs.value;
  const setP = (v) => setPs((s) => ({ ...s, value: v }));
  const setG = (v) => setGs((s) => ({ ...s, value: v }));
  // the file changed while you edited: saving overwrites it
  const stale = (s, path, reset) => s?.stale && html`<div class="notice"><${Icon} name="warning" /><div>
    ${t('{path} 在你编辑期间已被修改，保存会覆盖这些修改。', { path })}
    <button class="btn small" onClick=${reset}>${t('载入新版本')}</button>${t('（放弃你的修改）')}</div></div>`;
  const pick = (v, keys) => Object.fromEntries(keys.map((k) => [k, v[k]]));
  const save = (scope, values) => api.put('/api/settings', { scope, project: pid,
    values: scope === 'project' ? { ...pick(values, ['title', 'goal', 'autorun', 'max_running', 'agent', 'ledger']), machines: parseMachines(values.machinesText) }
      : pick(values, ['language', 'protocol_language', 'max_running', 'claude_config']) })
    .then(() => {
      if (scope === 'global' && values.language !== lang) return location.reload();
      // saved: what the server reads back replaces the form, even where it writes a value its own way
      (scope === 'project' ? setPs : setGs)((s) => ({ ...s, base: s.value, stale: false }));
      reload();
      toast(t('已保存 {path}', { path: scope === 'project' ? data.project.path : data.global.path }));
    }, fail);
  const pa = (k) => (e) => setP({ ...p, agent: { ...p.agent, [k]: k === 'budget_usd' ? +e.target.value : k === 'extra_args' ? e.target.value.split('\n').filter(Boolean) : e.target.value } });
  return html`<main id="main" tabIndex="-1" class="page settings-page">
    <header class="page-head"><div><h1>${t('设置')}</h1><p class="muted">${t('设置直接保存在 yaml 文件中，也可以在终端中直接编辑这些文件。')}</p></div></header>
    <div class=${p || data.projects ? 'two-col' : ''}>
      ${p && html`<div class="settings-col"><section class="form-grid side-card"><h2>${t('项目')} <span class="mono small muted">${data.project.path}</span></h2>
        ${stale(ps, data.project.path, () => setPs(refill(null, projectForm(data.project.values))))}
        <label class="field">${t('标题')}<input value=${p.title} onInput=${(e) => setP({ ...p, title: e.target.value })} /></label>
        <label class="field">${t('目标')}<input value=${p.goal} onInput=${(e) => setP({ ...p, goal: e.target.value })} placeholder=${t('例如 AgentX C64 的 tok/s/GPU 提升 20%，P90 interactivity ≥ 30')} />
          <span class="muted small">${t('显示在看板顶部，也是给 agent 的背景（写进项目规则更完整）')}</span></label>
        <label class="field">${t('机器（每行一台：主机名、GPU 数、备注）')}<textarea class="mono" rows="3" value=${p.machinesText} onInput=${(e) => setP({ ...p, machinesText: e.target.value })} placeholder="smc-a16-19 8 MI300X" />
          <span class="muted small">${t('任务在配置里选用其中的机器；同一台机器同时只给一个任务（运行中和待确认的任务占用它）')}</span></label>
        <div class="field-row"><label class="field">${t('台账主指标')}<input class="mono" value=${p.ledger.primary} onInput=${(e) => setP({ ...p, ledger: { ...p.ledger, primary: e.target.value.trim() } })} placeholder="tok_s_gpu" /></label>
          <label class="check"><input type="checkbox" checked=${p.ledger.higher_is_better !== false} onChange=${(e) => setP({ ...p, ledger: { ...p.ledger, higher_is_better: e.target.checked } })} />${t('越大越好')}</label></div>
        <label class="check"><input type="checkbox" checked=${p.autorun} onChange=${(e) => setP({ ...p, autorun: e.target.checked })} />${t('自动开始：上游任务全部完成后自动开始下游任务（关闭后所有任务均需手动开始）')}</label>
        <label class="field">${t('并发上限（本项目可同时运行的任务数，后台等待和终端接管的任务也计入）')}<input type="number" min="1" value=${p.max_running} onInput=${(e) => setP({ ...p, max_running: +e.target.value })} /></label>
        <h3>${t('新建任务的默认 agent 参数')}</h3>
        <div class="field-row"><label class="field">${t('模型')}<${ModelSelect} value=${p.agent.model} onChange=${(v) => pa('model')({ target: { value: v } })} /></label>
          <${EffortSelect} value=${p.agent.effort} onChange=${(x) => setP({ ...p, agent: { ...p.agent, ...x } })} />
          <label class="field">${t('预算 $')}<input type="number" value=${p.agent.budget_usd} onInput=${pa('budget_usd')} /></label></div>
        <${UltracodeCheck} value=${p.agent.ultracode} onChange=${(x) => setP({ ...p, agent: { ...p.agent, ...x } })} />
        <label class="field">${t('额外的 claude 参数（每行一个）')}<textarea class="mono" rows="2" value=${p.agent.extra_args.join('\n')} onInput=${pa('extra_args')} /></label>
        <div class="row-actions"><button class="btn primary" onClick=${() => save('project', p)}>${t('保存项目设置')}</button></div></section>
        ${project && html`<section class="side-card archive-card"><h2>${project.archived ? t('项目已归档') : t('归档项目')}</h2>
          <p class="muted small">${project.archived ? t('于 {time} 归档。取消归档后，项目重新显示在首页，待开始的任务恢复自动开始。', { time: clock(project.archived_at) })
            : t('归档后，项目不再显示在首页，调度器也不会自动开始其中的任务；项目文件不会改动。已归档的项目可在首页的"已归档项目"中查看和取消归档。')}</p>
          <div class="row-actions"><${ArchiveButton} project=${project} small=${false} /></div></section>`}</div>`}
      <section class="form-grid side-card"><h2>${t('全局')} <span class="mono small muted">${data.global.path}</span></h2>
        ${stale(gs, data.global.path, () => setGs(refill(null, { ...data.global.values })))}
        <label class="field">${t('界面语言')}<select value=${g.language} onChange=${(e) => setG({ ...g, language: e.target.value })}>
          ${Object.entries(LANGS).map(([k, v]) => html`<option value=${k}>${v}</option>`)}</select>
          <span class="muted small">${t('网页界面和 agent_sys 提示信息的语言；保存后页面重新载入')}</span></label>
        <label class="field">${t('agent_sys 约定的语言')}<select value=${g.protocol_language} onChange=${(e) => setG({ ...g, protocol_language: e.target.value })}>
          ${['zh', 'en'].map((k) => html`<option value=${k}>${LANGS[k]}</option>`)}</select>
          <span class="muted small">${t('注入给 agent 的 agent_sys 约定、system.md 的开头和 agent_sys 发给 agent 的消息用这种语言。agent 的工作语言（progress.md、report.md 和回复）跟任务 prompt 的语言一致，不受这项设置影响。')}</span></label>
        <label class="field">${t('agent 的 Claude Code 配置')}<select value=${g.claude_config} onChange=${(e) => setG({ ...g, claude_config: e.target.value })}>
          <option value="isolated">${t('独立（推荐）')}</option><option value="inherit">${t('继承我的 ~/.claude')}</option></select>
          <span class="muted small">${t('独立：agent 用 AGENT_SYS_HOME 下自己的 claude 配置目录，只带登录和模型路由要用的设置，不加载你的 MCP server、插件、CLAUDE.md 和 skills；需要的在任务的额外 claude 参数里显式加（--mcp-config、--plugin-dir）。继承：agent 会话加载你 ~/.claude 里的全部配置。')}</span></label>
        <label class="field">${t('全局并发上限（所有项目合计）')}<input type="number" min="1" value=${g.max_running} onInput=${(e) => setG({ ...g, max_running: +e.target.value })} /></label>
        <div class="row-actions"><button class="btn primary" onClick=${() => save('global', g)}>${t('保存全局设置')}</button></div></section>
      ${!p && data.projects && html`<${ProjectDefaults} projects=${data.projects} />`}
    </div></main>`;
}

// The global page also shows what each project.yaml sets, since the defaults for new tasks live there.
function ProjectDefaults({ projects }) {
  return html`<section class="side-card project-defaults"><h2>${t('各项目的默认设置')}</h2>
    <p class="muted small">${t('自动开始、并发上限和新建任务的默认 agent 参数保存在各项目的 project.yaml 中，点击"编辑"进入该项目的设置页修改。')}</p>
    ${projects.length ? html`<ul>${projects.map(({ id, path, values: v, archived }) => html`<li>
        <div class="entry-head"><b>${v.title}</b>${archived && html`<span class="tag">${t('已归档')}</span>`}<span class="spacer"></span>
          <a class="btn small" href=${`#/p/${id}/settings`}>${t('编辑')}</a></div>
        <div class="small">${t('{auto}；并发上限 {n}', { auto: autorunText(v.autorun), n: v.max_running })}</div>
        <div class="small">${agentText(v.agent)}${v.agent.extra_args.length
          ? t('，额外参数 {args}', { args: v.agent.extra_args.join(' ') }) : ''}</div>
        <div class="mono small muted">${path}</div></li>`)}</ul>`
      : html`<p class="empty-line">${t('暂无项目。')}</p>`}
  </section>`;
}

const root = document.getElementById('app');
root.textContent = '';  // the "loading" line of index.html
render(html`<${App} />`, root);
window.appMounted = true;  // boot.js stops waiting
