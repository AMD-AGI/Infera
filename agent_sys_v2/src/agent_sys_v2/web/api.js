// The UI's only data layer. Every call is (method, path, body) -> JSON, answered by the agent_sys server: REST over
// fetch, change notifications over SSE (one stream per page: a run view is told its run has new events and fetches
// them).
const http = {
  async request(method, path, body) {
    const res = await fetch(path, {
      method,
      headers: body ? { 'Content-Type': 'application/json' } : {},
      body: body ? JSON.stringify(body) : undefined,
    });
    const data = (res.headers.get('content-type') || '').includes('json') ? await res.json() : await res.text();
    if (!res.ok) {
      throw Object.assign(new Error(data?.error || `${res.status} ${res.statusText}`), { status: res.status, data });
    }
    return data;
  },
  // The stream is closed while the page is hidden: a browser gives all its tabs together 6 connections to one
  // host:port, and each open stream holds one for good. Back in view, it opens again and everything reloads.
  subscribe(fn) {
    let es = null, lost = false, timer = 0, paused = false;
    const open = () => {
      es = new EventSource('/api/events');
      es.addEventListener('changed', (ev) => fn({ type: 'changed', ...JSON.parse(ev.data) }));
      es.addEventListener('run', (ev) => fn({ type: 'run', ...JSON.parse(ev.data) }));
      es.onerror = () => { if (!lost || paused) { lost = true; paused = false; fn({ type: 'conn', state: 'closed' }); } };
      es.onopen = () => { if (lost) { lost = false; paused = false; fn({ type: 'conn', state: 'open' }); fn({ type: 'reconnected' }); } };
    };
    const visibility = () => {
      clearTimeout(timer);
      if (document.hidden) {  // not for a glance at another tab
        timer = setTimeout(() => { es?.close(); es = null; lost = paused = true; }, HIDDEN_CLOSE_MS);  // not shown as lost
      } else if (!es) open();
    };
    open();
    visibility();  // a tab opened in the background
    document.addEventListener('visibilitychange', visibility);
    return () => { clearTimeout(timer); es?.close(); document.removeEventListener('visibilitychange', visibility); };
  },
  rawUrl: (project, task, path) => `/api/projects/${project}/tasks/${task}/raw?path=${encodeURIComponent(path)}`,
};

const HIDDEN_CLOSE_MS = 10000;

// One shared change subscription, opened by the first handler; components register through api.onChange.
const handlers = new Set();
let subscribed = false;
function subscribe() {
  if (subscribed) return;
  subscribed = true;
  http.subscribe((c) => { for (const fn of handlers) fn(c); });
}

export const api = {
  get: (path) => http.request('GET', path),
  post: (path, body) => http.request('POST', path, body || {}),
  put: (path, body) => http.request('PUT', path, body || {}),
  // fn(change): {type: 'changed', project, task} (task null: the project), {type: 'run', project, task, run, seq}
  // (the run has events up to seq), {type: 'conn', state}, {type: 'reconnected'} (reload what you show)
  onChange(fn) { subscribe(); handlers.add(fn); return () => handlers.delete(fn); },
  rawUrl: http.rawUrl,
};

// Paths used by more than one page.
export const paths = {
  task: (p, id) => `/api/projects/${p}/tasks/${id}`,
  board: (p) => `/api/projects/${p}/board`,
};
