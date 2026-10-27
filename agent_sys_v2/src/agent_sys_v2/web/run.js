// Run tab: the events of one run (events.jsonl; while it runs, fetched as the server says there are new ones) folded
// into a readable stream, and the input box. What the agent writes is markdown; each tool call is one line (command,
// time, result). The server sends the events slim (events.slim): a long tool input or result is cut, a thinking block
// is only its length, and an item fetches the whole of its event when you open it. A run opens at its end; earlier
// events load when you scroll up.
import { useState, useEffect, useLayoutEffect, useRef, useMemo } from './vendor/hooks.module.js';
import { api, paths } from './api.js';
import { html, Icon, Markdown, Copy, AutoText, useNow, useId, money, clock, stopwatch, duration, since, toast, fail,
  confirmDialog } from './ui.js';
import { t as tr } from './i18n.js';

// ---------------------------------------------------------------- events -> items

// Folds raw events into display items one event at a time: a run's view keeps one Folder and gives it only the
// events it has not seen yet, so a long run is not folded again for every new event. Items change
// in place (a tool line gets its result later). Also follows the live state: commands in flight, background
// commands, whether the session is idle, and how your messages were delivered. A folder that starts in the middle of
// a run gets `carry`: the session's state and background commands as they were there (read_events).
export class Folder {
  constructor({ partial = false, carry = [] } = {}) {
    this.items = []; this.tools = new Map(); this.byTask = new Map();
    this.bg = []; this.session = 'idle'; this.inited = partial; this.firstUser = !partial;
    this.waiting = [];  // your messages not yet read: [{item, when}]
    for (const e of carry) this.add(e);
    this.count = 0;
  }

  fgInFlight() { return [...this.tools.values()].some((x) => !x.done && !x.background); }

  add(e) {
    this.count += 1;
    const n = this.items.length;
    const fold = { runner: this.runner, system: this.system, tool_progress: this.toolProgress, assistant: this.assistant,
      user: this.user, result: this.result }[e.type];
    fold?.call(this, e, e.replay ? tr('回放') : '');
    for (let i = n; i < this.items.length; i++) this.items[i].key = `${e.seq}.${i - n}`;  // stays when earlier ones load
  }

  // agent_sys's own records: your messages, stop, the acceptance checks, the runner's exit
  runner(e, synthetic) {
    const { items } = this;
    if (e.subtype === 'send_user') {
      const when = !this.firstUser && this.fgInFlight() ? 'after_tool' : !this.firstUser && this.session === 'running' ? 'next_turn' : null;
      const item = { kind: 'user', text: e.text, ts: e.ts, first: this.firstUser, delivery: when ? (when === 'after_tool' ? tr('已送达，将在当前命令结束后读取') : tr('已送达，将在本轮结束后读取')) : '' };
      items.push(item);
      if (when) this.waiting.push({ item, when });
      this.firstUser = false;
    } else if (e.subtype === 'inbox' && ['stop', 'interrupt'].includes(e.cmd?.type)) {  // the other command is your message (send_user)
      items.push({ kind: 'note', icon: 'stop', ts: e.ts, text: e.cmd.reason === 'takeover' ? tr('停止 runner，转为在终端中接管') : tr('已发送停止指令（interrupt）') });
    } else if (e.subtype === 'checks') {
      items.push({ kind: 'note', ts: e.ts, icon: e.phase === 'start' ? 'tool' : e.stopped ? 'stop' : e.ok ? 'done' : 'warning',
        text: e.phase === 'start' ? tr('agent 标记了完成，开始验收检查（{total} 条）', e) : e.stopped ? tr('验收检查随停止中断，本次不计')
          : e.ok ? tr('验收检查全部通过（{passed}/{total}）', e) : tr('验收检查未通过（{passed}/{total}，第 {attempt} 次）', e) });
    } else if (e.subtype === 'check' && e.phase === 'end') {
      items.push({ kind: 'check', ts: e.ts, n: e.n, total: e.total, name: e.name, run: e.run, ok: e.ok, exit: e.exit,
        timed_out: e.timed_out, seconds: e.seconds, output: e.output || '' });
    } else if (e.subtype === 'exit') {
      const why = { idle: tr('本轮结束，无后台等待，runner 已退出'), stopped: tr('已停止，runner 已退出'), budget: tr('预算已用尽，runner 已退出'),
        takeover: tr('已转为终端接管'), error: e.detail ? tr('runner 出错退出：{detail}', { detail: e.detail }) : tr('runner 出错退出') }[e.reason] || tr('runner 已退出：{reason}', { reason: e.reason });
      items.push({ kind: 'note', icon: e.reason === 'idle' ? 'done' : 'stop', text: why, ts: e.ts, synthetic });
    }
  }

  // the session: its start, idle or running, its background commands
  system(e, synthetic) {
    const { tools, byTask } = this;
    if (e.subtype === 'init' && !this.inited) {
      this.inited = true;
      this.items.push({ kind: 'note', icon: 'dot', text: `${tr('会话开始')} · ${e.model || ''} · CLI ${e.claude_code_version || ''} · ${e.permissionMode || ''}`, ts: e.ts, synthetic });
    } else if (e.subtype === 'session_state_changed') {
      this.session = e.state;
      if (e.state === 'idle') for (const w of this.waiting.splice(0)) w.item.delivery = tr('已读取');
    } else if (e.subtype === 'background_tasks_changed') {
      this.bg = e.tasks || [];
    } else if (e.subtype === 'task_started') {
      const tool = tools.get(e.tool_use_id);
      if (tool) { tool.background = e.is_backgrounded; byTask.set(e.task_id, tool); tool.taskId = e.task_id; }
    } else if (e.subtype === 'task_notification' || e.subtype === 'task_updated') {
      const tool = tools.get(e.tool_use_id) || byTask.get(e.task_id);
      const status = e.status || e.patch?.status;
      if (tool && status && status !== 'running') {
        Object.assign(tool, { bgStatus: status, bgEnded: e.ts, outputFile: e.output_file || tool.outputFile });
        if (tool.background && e.subtype === 'task_notification') this.items.push({ kind: 'note', icon: status === 'completed' ? 'done' : 'stop', ts: e.ts, synthetic,
          text: `${status === 'completed' ? tr('后台命令已结束：{cmd}', { cmd: tool.summary }) : tr('后台命令已终止：{cmd}', { cmd: tool.summary })}${e.summary && /time limit/.test(e.summary) ? tr('（超过后台运行时限）') : ''}` });
      }
    }
  }

  toolProgress(e) {
    const tool = this.tools.get(e.parent_tool_use_id);
    if (tool) tool.elapsed = Math.max(tool.elapsed || 0, e.elapsed_time_seconds || 0);
  }

  // what the agent writes: text, thinking, tool calls
  assistant(e, synthetic) {
    for (const [at, c] of (e.message?.content || []).entries()) {
      const whole = c.clipped ? [e.seq, at] : null;  // where the rest of a cut block is
      if (c.type === 'text' && c.text.trim()) this.items.push({ kind: 'text', text: c.text, ts: e.ts, synthetic });
      else if (c.type === 'thinking' && (c.thinking || c.chars)) this.items.push({ kind: 'thinking', text: c.thinking, chars: c.chars ?? c.thinking.length, whole, ts: e.ts });
      else if (c.type === 'tool_use') {
        const tool = { kind: 'tool', id: c.id, name: c.name, input: c.input || {}, wholeIn: whole, ts: e.ts, background: !!c.input?.run_in_background,
          summary: toolSummary(c.name, c.input || {}), synthetic };
        this.tools.set(c.id, tool);
        this.items.push(tool);
      }
    }
  }

  // what comes back to the agent: tool results, an interrupt
  user(e) {
    const content = e.message?.content;
    for (const [at, c] of (Array.isArray(content) ? content : []).entries()) {
      if (c.type === 'tool_result') {
        const tool = this.tools.get(c.tool_use_id);
        if (tool) {
          Object.assign(tool, { output: resultText(c.content), wholeOut: c.clipped ? [e.seq, at] : null, error: !!c.is_error, ended: e.ts });
          if (!tool.background) tool.done = true;
          else tool.started = true;
        }
        if (!this.fgInFlight()) {
          for (const w of this.waiting.filter((x) => x.when === 'after_tool')) { w.item.delivery = tr('已读取'); this.waiting.splice(this.waiting.indexOf(w), 1); }
        }
      } else if (c.type === 'text') {
        if (/^\[Request interrupted/.test(c.text)) this.items.push({ kind: 'note', icon: 'stop', text: tr('已中断：正在执行的命令已终止'), ts: e.ts });
        else if (e.isSynthetic) this.items.push({ kind: 'note', icon: 'dot', text: tr('Claude Code 要求模型补充一条可见回复'), ts: e.ts });
      }
    }
  }

  // the end of a round
  result(e, synthetic) {
    const cost = typeof e.total_cost_usd === 'number' ? ` · ${tr('会话累计 {cost}', { cost: money(e.total_cost_usd) })}` : '';
    const text = e.subtype === 'success' ? `${tr('本轮结束')}${e.origin?.kind === 'task-notification' ? tr('（由后台命令结束唤醒）') : ''}${cost}`
      : e.subtype === 'error_max_budget_usd' ? `${tr('预算已用尽，本轮中止')}${cost}`
        : e.terminal_reason === 'aborted_tools' ? tr('本轮已停止（手动停止，并非运行失败）') : tr('本轮出错：{subtype}', { subtype: e.subtype });
    this.items.push({ kind: 'note', icon: e.is_error ? 'stop' : 'done', text, ts: e.ts, synthetic, round: true });
  }

  view() {
    const inFlight = [...this.tools.values()].filter((x) => !x.done && !x.background);
    return { items: this.items, inFlight, bg: this.bg, session: this.session };
  }
}

// All of a run's events at once (tests, and anything that does not follow a live run).
export function fold(events) {
  const f = new Folder();
  for (const e of events) f.add(e);
  return f.view();
}

function toolSummary(name, i) {
  if (name === 'Bash') return i.command || i.description || '';
  if (['Read', 'Write', 'Edit', 'NotebookEdit'].includes(name)) return i.file_path || i.notebook_path || '';
  if (name === 'Grep' || name === 'Glob') return `${i.pattern || ''}${i.path ? ` · ${i.path}` : ''}`;
  if (name === 'WebFetch' || name === 'WebSearch') return i.url || i.query || '';
  if (name === 'Task' || name === 'Agent') return i.description || '';
  return JSON.stringify(i).slice(0, 160);
}

function resultText(c) {
  if (typeof c === 'string') return c;
  if (Array.isArray(c)) return c.map((x) => x.text || '').join('\n');
  return JSON.stringify(c);
}

// The rest of an opened item that the server cut: load(seq) gives an event whole.
async function fillIn(it, load) {
  const block = ([seq, at]) => load(seq).then((e) => e.message.content[at]);
  if (it.whole) it.text = (await block(it.whole)).thinking;
  if (it.wholeIn) it.input = (await block(it.wholeIn)).input || {};
  if (it.wholeOut) it.output = resultText((await block(it.wholeOut)).content);
  Object.assign(it, { whole: null, wholeIn: null, wholeOut: null });
}

// ---------------------------------------------------------------- the tab

export function RunTab({ pid, t }) {
  const [chosen, setChosen] = useState(null);
  const latest = t.runs[0];
  const run = (chosen && t.runs.find((r) => r.id === chosen)) || latest;
  // a new run (a reply resumed the session) takes over the view
  useEffect(() => { if (latest?.live) setChosen(null); }, [latest?.id]);
  return html`<div class="run-tab">
    ${t.runs.length > 0 && html`<div class="run-picker">
      <label>${tr('运行')} <select value=${run?.id} onChange=${(e) => setChosen(e.target.value)}>
        ${t.runs.map((r, i) => html`<option value=${r.id}>${runLabel(r, i === 0)}</option>`)}</select></label>
      <span class="muted small">${tr('共 {n} 次运行，累计 {cost}', { n: t.runs.length, cost: money(t.cost_usd) })}</span></div>`}
    ${run ? html`<${RunView} key=${run.id} pid=${pid} t=${t} run=${run} />` : html`<${NoRun} t=${t} />`}
    <${Composer} pid=${pid} t=${t} run=${run} />
  </div>`;
}

// translated when shown, not when the module loads (the language can be set after, in tests)
const EXIT_ZH = { idle: '正常结束', stopped: '已停止', interrupted: '运行中断', budget: '预算用尽', takeover: '转为终端接管',  // i18n: translated where shown
  terminal_exit: '已退出终端', error: '出错' };  // i18n: translated where shown

function runLabel(r, latest) {
  const kind = r.kind === 'terminal' ? tr('终端接管') : r.kind === 'replay' ? tr('回放') : r.resume ? 'resume' : tr('首次运行');
  const state = r.live ? tr('进行中') : EXIT_ZH[r.exit] ? tr(EXIT_ZH[r.exit]) : r.exit || '';
  return `${latest ? `${tr('最新')} · ` : ''}${clock(r.started)} · ${kind} · ${state}${r.cost_usd ? ` · ${money(r.cost_usd)}` : ''}`;
}

function NoRun({ t }) {
  const text = !t.managed ? tr('该任务未托管，agent_sys 中没有其运行记录。工作区文件可在"工作区文件"中查看；如需交由 agent_sys 管理，请点击上方的"纳入托管"。')
    : t.detail?.kind === 'joined' ? tr('已纳入托管，尚未运行。回复后，agent_sys 将开启新会话，首条消息为你的回复以及"工作区中已有进展，请先阅读 progress.md"。')
      : t.status === 'queued' ? tr('尚未开始运行。上游任务全部完成且有运行名额时将自动开始，也可点击上方的"开始"。')
        : t.status === 'draft' ? tr('草稿不会运行。编写完成后，请点击上方的"纳入托管"。') : tr('暂无运行记录。');
  return html`<div class="empty-state">${text}</div>`;
}

const PAGE = 400;  // events a run view loads at first, and per step back (runs.TAIL)

// The events of a run as far as they are loaded: {page: {events, carry, first} (from seq `first` to the end), error,
// earlier: loading earlier ones, loadEarlier(), load(seq): one event whole}. First the run's end, then what is new
// whenever the server says the run has more (or the connection came back), one request at a time.
function useRunEvents(url, run, at) {
  const [page, setPage] = useState(null);
  const [error, setError] = useState(null);
  const [earlier, setEarlier] = useState(false);
  const pull = useRef(() => {});
  const wasLive = useRef(run.live);
  useEffect(() => {
    let alive = true, busy = false, again = false, upto = null;
    const get = () => {
      if (busy) { again = true; return; }
      busy = true;
      api.get(upto == null ? `${url}?tail=${PAGE}` : `${url}?after=${upto}`).then((r) => {
        if (!alive) return;
        if (upto == null) { setPage({ events: r.events, carry: r.carry, first: r.first }); setError(null); }  // a retry worked
        else if (r.events.length) setPage((p) => ({ ...p, events: [...p.events, ...r.events] }));
        upto = r.upto;
      }, (e) => alive && upto == null && setError(e)).finally(() => {
        busy = false;
        if (again && alive) { again = false; get(); }
      });
    };
    pull.current = (seq) => { if (seq == null || upto == null || seq > upto) get(); };
    get();
    const off = api.onChange((c) => {
      if (c.type === 'reconnected' || (c.type === 'run' && at(c) && c.run === run.id)) pull.current(c.seq);
    });
    return () => { alive = false; off(); };
  }, [run.id]);
  useEffect(() => {  // the run ended: its last events
    if (wasLive.current && !run.live) pull.current();
    wasLive.current = run.live;
  }, [run.live]);
  const load = useMemo(() => {
    const got = new Map();
    return (seq) => {
      if (!got.has(seq)) got.set(seq, api.get(`${url}/${seq}`).catch((e) => { got.delete(seq); throw e; }));
      return got.get(seq);
    };
  }, [run.id]);
  // before: called with the response, before the events are put above (to keep the scroll position)
  const loadEarlier = (before) => {
    if (earlier || !page?.first) return;
    setEarlier(true);
    api.get(`${url}?before=${page.first}&tail=${PAGE}`).then((r) => {
      before(r);
      setPage((p) => ({ ...p, events: [...r.events, ...p.events], carry: r.carry, first: r.first }));
    }, (e) => toast(tr('加载更早的输出失败：{error}', { error: e.message }), { error: true })).finally(() => setEarlier(false));
  };
  return { page, error, earlier, loadEarlier, load };
}

function RunView({ pid, t, run }) {
  const base = paths.task(pid, t.id);
  const { page, error, earlier, loadEarlier, load } = useRunEvents(`${base}/runs/${run.id}/events`, run,
    (c) => c.project === pid && c.task === t.id);
  const listRef = useRef();
  const stick = useRef(true);
  const anchor = useRef(null);  // distance from the bottom to keep while earlier events are put above
  const folder = useRef(null);
  const [newBelow, setNewBelow] = useState(false);
  const earlierAbove = () => loadEarlier((r) => {
    const el = listRef.current;
    if (el && r.events.length) anchor.current = el.scrollHeight - el.scrollTop;
  });
  // only the events the folder has not seen; earlier events loaded fold the page again
  const view = useMemo(() => {
    if (!page) return null;
    let f = folder.current;
    if (!f || f.from !== page.first || page.events.length < f.count) {
      f = folder.current = Object.assign(new Folder({ partial: page.first > 0, carry: page.carry }), { from: page.first });
    }
    for (let i = f.count; i < page.events.length; i++) f.add(page.events[i]);
    return f.view();
  }, [page]);
  // follow the bottom while you are at the bottom; earlier events go above what you are reading
  useLayoutEffect(() => {
    const el = listRef.current;
    if (!el) return;
    if (anchor.current != null) { el.scrollTop = el.scrollHeight - anchor.current; anchor.current = null; return; }
    if (stick.current) el.scrollTop = el.scrollHeight; else setNewBelow(true);
  }, [page?.events.length]);
  const onScroll = () => {
    const el = listRef.current;
    stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
    if (stick.current) setNewBelow(false);
    if (el.scrollTop < 300) earlierAbove();
  };
  if (run.kind === 'terminal') return html`<${TerminalPanel} pid=${pid} t=${t} run=${run} />`;
  if (error) return html`<div class="error-box">${tr('加载运行输出失败：{error}', { error: error.message })}</div>`;
  if (!view) return html`<div class="loading">${tr('正在加载运行输出…')}</div>`;
  const { items, inFlight, bg, session } = view;
  const last = page.events[page.events.length - 1];
  const killed = !run.live && run.exit === 'interrupted';
  return html`<div class="run-view">
    ${run.kind === 'replay' && html`<div class="mock-note small">${tr('回放模式：本次运行回放录好的事件，没有调用模型，内容与本任务无关（录制文件见 launch.md）。')}</div>`}
    <${LaunchPanel} base=${base} run=${run} />
    <${LiveStrip} run=${run} inFlight=${inFlight} bg=${bg} session=${session} killed=${killed} t=${t} />
    <div class="stream" ref=${listRef} onScroll=${onScroll} aria-live="polite" aria-label=${tr('运行输出')}>
      ${page.first > 0 && html`<button class="btn ghost small load-earlier" onClick=${earlierAbove} disabled=${earlier}>${earlier ? tr('正在加载…') : tr('加载更早的输出')}</button>`}
      ${items.map((it, i) => html`<${Item} key=${it.key ?? i} it=${it} load=${load} runEnded=${!run.live} />`)}
      ${killed && last && html`<div class="item note warn"><${Icon} name="warning" /><span>${tr('运行中断：runner 进程已退出，最后一条事件的时间为 {time}。正在执行的命令和后台命令均已终止。', { time: clock(new Date(last.ts * 1000).toISOString()) })}</span></div>`}
    </div>
    ${newBelow && html`<button class="btn small new-below" onClick=${() => { stick.current = true; listRef.current.scrollTop = listRef.current.scrollHeight; setNewBelow(false); }}>↓ ${tr('有新输出')}</button>`}
  </div>`;
}

// A running clock: the only parts of the run view that change every second.
function Ticker({ from, min = 0, prefix = '' }) {
  const now = useNow(1000);
  return `${prefix}${stopwatch(Math.max(now - from, min))}`;
}

// What the run is doing right now: the command in flight and its clock, or what it waits for.
export function LiveStrip({ run, inFlight, bg, session, killed, t }) {
  if (!run.live) {
    const took = run.ended && run.started ? duration(Date.parse(run.ended) - Date.parse(run.started)) : '';
    const why = { ...EXIT_ZH, stopped: '已手动停止', interrupted: '运行中断（runner 进程已退出）', budget: '预算已用尽' }[run.exit];  // i18n: translated where shown
    const whyText = why ? tr(why) : run.exit;
    return html`<div class="live-strip ended"><${Icon} name=${killed || run.exit === 'error' ? 'warning' : 'done'} /><span>${tr('本次运行已结束：{why}', { why: whyText })}${run.exit_detail ? tr('（{detail}）', { detail: run.exit_detail }) : ''}${took ? tr('，用时 {took}', { took }) : ''}${tr('，花费 {cost}', { cost: money(run.cost_usd) })}</span></div>`;
  }
  if (run.host) return html`<div class="live-strip working"><${Icon} name="running" /><span>${tr('该运行在 {host} 上，本机的 agent_sys 只显示，不能停止或接管', { host: run.host })}</span></div>`;
  const fg = inFlight[0];
  if (fg) {
    return html`<div class="live-strip working"><${Icon} name="running" className="busy" />
      <span>${tr('正在执行')} <b>${fg.name}</b> <code class="ellipsis">${fg.summary}</code></span><span class="clock"><${Ticker} from=${fg.ts * 1000} min=${(fg.elapsed || 0) * 1000} prefix=${tr('已运行 ')} /></span></div>`;
  }
  const check = run.current?.tool === 'Check' && run.current;  // the server's: the session is idle meanwhile
  if (check) {
    return html`<div class="live-strip working"><${Icon} name="running" className="busy" />
      <span>${tr('正在运行验收检查')} <code class="ellipsis">${check.text}</code></span><span class="clock"><${Ticker} from=${Date.parse(check.since)} prefix=${tr('已运行 ')} /></span></div>`;
  }
  if (session === 'idle' && bg.length) {
    const cmd = bg[0].description, next = t.progress?.next;
    return html`<div class="live-strip waiting"><${Icon} name="waiting" />
      <span>${tr('等待中：')}${next || html`${tr('后台命令')} <code class="ellipsis" title=${cmd}>${cmd}</code>`}<span class="muted">${next ? tr('（后台命令：{cmd}；结束后自动开始新一轮，等待期间不消耗 token）', { cmd }) : tr('（结束后自动开始新一轮，等待期间不消耗 token）')}</span></span>
      <span class="clock"><${Ticker} from=${Date.parse(t.since) || Date.now()} prefix=${tr('已等待 ')} /></span></div>`;
  }
  return html`<div class="live-strip working"><${Icon} name="running" className="busy" /><span>${tr('agent 正在思考或输出')}</span>
    <span class="clock"><${Ticker} from=${Date.parse(run.started)} prefix=${tr('本次运行 ')} /></span></div>`;
}

// runEnded: a tool that never got its result was cut short with the run.
function Item({ it, load, runEnded }) {
  const [open, setOpen] = useState(false);
  const [, redraw] = useState(0);
  const [failed, setFailed] = useState(null);
  const clipped = !!(it.whole || it.wholeIn || it.wholeOut);  // the server cut it
  // it: earlier events loaded fold the run again, into new items under the same keys
  useEffect(() => { if (open && clipped && load) fillIn(it, load).then(() => redraw((n) => n + 1), setFailed); }, [open, clipped, it]);
  const more = clipped && open && html`<div class="small muted">${failed ? tr('加载全文失败：{error}', { error: failed.message }) : tr('正在加载全文…')}</div>`;
  const tag = it.synthetic ? html`<span class="tag synthetic" title=${tr('回放模式给出的说明，没有调用模型')}>${it.synthetic}</span>` : '';
  if (it.kind === 'user') {
    const long = it.first && it.text.length > 400;
    return html`<div class="item user"><div class="who">${tr('你')}${it.first ? html`<span class="muted small">${tr('（首条消息）')}</span>` : ''}</div>
      <div class="bubble"><div class=${`pre-wrap ${long && !open ? 'clamp' : ''}`}>${it.text}</div>
        ${long && html`<button class="btn ghost small" onClick=${() => setOpen(!open)}>${open ? tr('收起') : tr('展开全文')}</button>`}
        ${it.delivery && html`<div class=${`delivery ${it.delivery === tr('已读取') ? 'read' : ''}`}>${it.delivery}</div>`}</div></div>`;
  }
  if (it.kind === 'text') return html`<div class="item text">${tag}<${Markdown} text=${it.text} /></div>`;
  if (it.kind === 'thinking') {
    return html`<div class="item thinking"><button class="line-btn" aria-expanded=${open} onClick=${() => setOpen(!open)}>
      <${Icon} name="chevron" className=${open ? 'rot90' : ''} />${tr('思考过程')} <span class="muted">${tr('{n} 字', { n: it.chars ?? it.text.length })}</span></button>
      ${open && (clipped ? more : html`<div class="pre-wrap muted small">${it.text}</div>`)}</div>`;
  }
  if (it.kind === 'tool') {
    const cut = runEnded && !it.done && !it.bgStatus;
    const running = !it.done && !it.background && !cut;
    const bgRunning = it.background && !it.bgStatus && !cut;
    const status = running ? 'running' : bgRunning ? 'bg' : cut || it.error || it.bgStatus === 'stopped' || it.bgStatus === 'killed' ? 'stop' : 'done';
    const time = cut ? tr('因运行中断而终止')
      : it.background ? (it.bgEnded ? tr('后台运行 {time}', { time: duration((it.bgEnded - it.ts) * 1000) }) : html`<${Ticker} from=${it.ts * 1000} prefix=${tr('后台已运行 ')} />`)
        : running ? html`<${Ticker} from=${it.ts * 1000} min=${(it.elapsed || 0) * 1000} prefix=${tr('已运行 ')} />` : duration((it.ended - it.ts) * 1000);
    return html`<div class=${`item tool ${it.error ? 'error' : ''}`}>
      <button class="line-btn" aria-expanded=${open} onClick=${() => setOpen(!open)}>
        <${Icon} name="chevron" className=${open ? 'rot90' : ''} /><${Icon} name=${status} className=${status === 'running' ? 'busy' : ''} />
        <b>${it.name}</b>${it.background ? html`<span class="tag">${tr('后台')}</span>` : ''}<code class="ellipsis">${it.summary}</code>
        <span class="muted nowrap">${time}</span>${tag}</button>
      ${open && html`<div class="tool-body">${more}
        <div class="label">${tr('输入')}</div><pre>${it.name === 'Bash' ? it.input.command : JSON.stringify(it.input, null, 2)}</pre>
        ${it.output != null && html`<div class="label">${it.error ? tr('结果（出错）') : tr('结果')}</div><${Output} text=${it.output} />`}
        ${it.outputFile && html`<div class="small muted">${tr('后台输出文件：')}<code>${it.outputFile}</code></div>`}</div>`}</div>`;
  }
  if (it.kind === 'check') {
    return html`<div class=${`item tool check ${it.ok ? '' : 'error'}`}>
      <button class="line-btn" aria-expanded=${open} onClick=${() => setOpen(!open)}>
        <${Icon} name="chevron" className=${open ? 'rot90' : ''} /><span class=${it.ok ? 'ok-text' : 'warn-text'}>${it.ok ? '✓' : '✗'}</span>
        <b>${tr('验收 {n}/{total}', it)}</b><code class="ellipsis">${it.name}</code>
        <span class="muted nowrap">${it.timed_out ? tr('超时') : `exit ${it.exit}`} · ${it.seconds}s</span></button>
      ${open && html`<div class="tool-body"><div class="label">${tr('命令')}</div><pre>${it.run}</pre>
        <div class="label">${tr('输出（末尾）')}</div><${Output} text=${it.output || tr('（无输出）')} /></div>`}</div>`;
  }
  return html`<div class=${`item note ${it.round ? 'round' : ''}`}><${Icon} name=${it.icon} /><span>${it.text}</span>${tag}
    <span class="muted small nowrap">${clock(new Date(it.ts * 1000).toISOString()).slice(6)}</span></div>`;
}

function Output({ text }) {
  const [all, setAll] = useState(false);
  const lines = text.split('\n');
  const shown = all ? lines : lines.slice(0, 200);
  return html`<pre>${shown.join('\n')}</pre>${lines.length > 200 && !all && html`<button class="btn ghost small" onClick=${() => setAll(true)}>${tr('展开全部 {n} 行', { n: lines.length })}</button>`}`;
}

function LaunchPanel({ base, run }) {
  const [md, setMd] = useState(null);
  const load = (ev) => { if (ev.target.open && md == null) api.get(`${base}/runs/${run.id}/launch`).then((r) => setMd(r.markdown), (e) => setMd(tr('加载失败：{error}', { error: e.message }))); };
  return html`<details class="launch" onToggle=${load}><summary>${tr('本次注入内容与等价命令（launch.md）')}</summary>
    ${md == null ? html`<div class="loading">${tr('正在加载…')}</div>` : html`<${Markdown} text=${md} className="small" />`}</details>`;
}

function TerminalPanel({ pid, t, run }) {
  const now = useNow(1000);
  if (!run.live) {
    return html`<div class="terminal-panel ended"><p>${tr('终端接管')} ${clock(run.started)} – ${clock(run.ended)}${tr('（{time}）', { time: duration(Date.parse(run.ended) - Date.parse(run.started)) })} · ${money(run.cost_usd)}</p>
      <p class="muted small">${tr('接管期间的对话保存在会话记录中，不在 agent_sys 的事件流里；退出后已根据 progress.md 更新任务状态。')}</p></div>`;
  }
  // claude has exited (结束接管, or in the terminal); its wrapper runs the checks
  if (t.detail?.current) {
    return html`<div class="terminal-panel"><p><${Icon} name="running" className="busy" /> ${tr('验收检查运行中，完成后自动更新任务状态：')}<code>${t.detail.current}</code></p></div>`;
  }
  return html`<div class="terminal-panel">
    <p><${Icon} name="terminal" /> <b>${tr('该任务正在终端中接管')}</b>${tr('（已 {time}）。输出显示在终端中，此处不显示；网页输入框暂停使用。', { time: since(run.started, now) })}</p>
    <${AttachCommand} attach=${run.attach} />
    <p class="muted small">${tr('退出 claude 后，agent_sys 将根据 progress.md 自动判断状态（已完成 → 已完成，否则 → 待确认），无需手动标记；直接关闭 tmux 窗口效果相同。')}</p>
    <button class="btn" onClick=${() => endTakeover(pid, t)}>${tr('结束接管')}</button>
  </div>`;
}

export function AttachCommand({ attach }) {
  const host = location.hostname === 'localhost' || location.hostname === '127.0.0.1' ? tr('<agent_sys 所在主机>') : location.hostname;
  const ssh = `ssh -t ${host} ${attach}`;
  return html`<div class="attach">
    <div class="cmd"><code>${attach}</code><${Copy} text=${attach} /></div>
    <div class="cmd"><code>${ssh}</code><${Copy} text=${ssh} label=${tr('复制 ssh 命令')} /></div></div>`;
}

export async function endTakeover(pid, t) {
  const ok = await confirmDialog({ title: tr('确认结束接管？'), ok: tr('结束接管'), body: html`<p>${tr('将结束终端中的 claude 进程，会话记录保留；随后根据 progress.md 判断任务状态。')}</p>` });
  if (!ok) return;
  return api.post(`${paths.task(pid, t.id)}/actions/end_takeover`).then((r) => toast(endTakeoverText(r.task)), fail);
}

// The task after 结束接管: what it became, or (still running) that its checks run
export const endTakeoverText = (task) => (task.status === 'running' ? tr('已结束接管：验收检查运行中，完成后自动更新任务状态')
  : tr('已结束接管，任务状态：{status}', { status: { done: tr('已完成'), needs_you: tr('待确认'), in_review: tr('待审核') }[task.status] || task.status }));

// 停止 (the drawer's head and the input box)
export async function stopRun(pid, t) {
  if (!await confirmDialog({ title: tr('确认停止本次运行？'), ok: tr('停止'), danger: true,
    body: html`<p>${tr('正在执行的命令将立即终止，任务转为"待确认"。会话保留，回复后可继续运行。')}</p>` })) return;
  return api.post(`${paths.task(pid, t.id)}/actions/stop`).then(() => toast(tr('已停止，任务转为待确认')), fail);
}

// What you send a task -> [path, body]: your words as a message; 继续 without words is the resume action, which the
// server words in the agent's language (agent_sys.yaml protocol_language).
export const messageRequest = (base, text) => (text.trim() ? [`${base}/messages`, { text }] : [`${base}/actions/resume`, {}]);
export const sendMessage = (pid, tid, text) => api.post(...messageRequest(paths.task(pid, tid), text));

// ---------------------------------------------------------------- input box

function composerState(t, run) {
  if (!t.managed) return { off: tr('该任务未托管，agent_sys 无法访问其会话。请在终端中回复，或先点击上方的"纳入托管"。') };
  if (['queued', 'draft'].includes(t.status)) return { off: tr('任务尚未开始，暂不能发送消息。') };
  if (t.status === 'cancelled') return { off: tr('任务已取消。') };
  if (t.status === 'running' && t.detail?.kind === 'terminal') return { off: tr('该任务正在终端中接管，请在终端中回复。') };
  if (t.detail?.kind === 'budget' && t.cost_usd >= t.budget_usd) return { off: tr('预算已用尽（{cost} / {budget}），请先在上方追加预算。', { cost: money(t.cost_usd), budget: money(t.budget_usd) }) };
  if (t.status === 'running') {
    const a = run?.activity;
    if (a === 'waiting' || t.detail?.kind === 'waiting') return { hint: tr('消息将立即触发新一轮回复；后台命令不受影响，结束后照常唤醒。'), placeholder: tr('提问或补充要求') };
    if (run?.current || t.detail?.current) return { hint: tr('消息将在当前命令结束后、于本轮内被读取。前台命令可能耗时较长，如需立即调整，请先停止再发送。'), placeholder: tr('补充要求或调整范围') };
    return { hint: tr('消息将在本轮结束后，于新一轮中处理。'), placeholder: tr('补充要求或调整范围') };
  }
  if (t.status === 'in_review') return { hint: tr('反馈将在同一会话中处理，修改完成后重新进入待审核；如无问题，请点击上方的"审核通过"。'), placeholder: tr('对报告的修改意见') };
  if (t.status === 'done') return { hint: tr('追问将在同一会话中开始新一轮；若 report.md 未变化，任务保持已完成。'), placeholder: tr('追问内容') };
  if (t.detail?.kind === 'joined') return { hint: tr('回复将与"工作区中已有进展，请先阅读 progress.md"一并作为新会话的首条消息。请先确认终端中的会话已结束。'), placeholder: tr('回复') };
  return { hint: tr('消息将在原会话中继续（resume），任务恢复运行。'), placeholder: t.detail?.kind === 'interrupted' || t.detail?.kind === 'wait_interrupted' ? tr('继续') : tr('回复') };
}

function Composer({ pid, t, run }) {
  const [text, setText] = useState('');
  const [busy, setBusy] = useState(false);
  const s = composerState(t, run);
  const resume = /interrupted/.test(t.detail?.kind || '');  // 继续 with nothing typed
  const send = () => {
    if ((!text.trim() && !resume) || busy) return;
    setBusy(true);
    sendMessage(pid, t.id, text).then((r) => {
      setText('');
      toast({ after_tool: tr('已送达，将在当前命令结束后读取'), next_turn: tr('已送达，将在本轮结束后读取'), now: tr('已送达，agent 将立即回复'), resume: tr('已发送，任务恢复运行') }[r.delivery] || tr('已发送'));
    }, fail).finally(() => setBusy(false));
  };
  const live = t.status === 'running' && run?.live && run.kind !== 'terminal';
  return html`<div class="composer">
    ${s.off ? html`<div class="composer-off"><${Icon} name="dot" />${s.off}</div>` : html`
      <label class="sr-only" for=${`msg-${t.id}`}>${tr('发送消息给 agent')}</label>
      <${AutoText} id=${`msg-${t.id}`} value=${text} onInput=${setText} onSubmit=${send} placeholder=${s.placeholder} rows=${2} />
      <div class="composer-row"><span class="muted small">${s.hint}</span>
        ${live && html`<button class="btn danger-ghost" onClick=${() => stopRun(pid, t)}><${Icon} name="stop" />${tr('停止')}</button>`}
        <button class="btn primary" onClick=${send} disabled=${busy || (!text.trim() && !resume)}>${text.trim() || !resume ? tr('发送') : tr('继续')} <span class="kbd">Ctrl+Enter</span></button></div>`}
  </div>`;
}

// 追加预算并继续: $100 unless you change the amount (it goes into the task's agent.budget_usd).
export function BudgetButton({ pid, t }) {
  const [usd, setUsd] = useState('100');
  const [busy, setBusy] = useState(false);
  const ok = Number(usd) > 0;
  const add = () => {
    setBusy(true);
    api.post(`${paths.task(pid, t.id)}/actions/add_budget`, { usd: Number(usd) })
      .then((r) => toast(tr('预算已追加 ${usd}（共 {total}），任务继续运行', { usd: Number(usd), total: money(r.budget_usd) })), fail)
      .finally(() => setBusy(false));
  };
  const id = useId('budget');
  return html`<span class="budget-add"><label class="sr-only" for=${id}>${tr('追加金额（美元）')}</label>
    <span class="unit" aria-hidden="true">$</span><input id=${id} type="number" min="1" step="10" value=${usd}
      onInput=${(e) => setUsd(e.target.value)} onKeyDown=${(e) => e.key === 'Enter' && ok && add()} />
    <button class="btn primary small" disabled=${busy || !ok} onClick=${add}>${tr('追加预算并继续')}</button></span>`;
}
