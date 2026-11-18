// Small wrapper around the server's JSON API. Errors carry the server's message.
async function request(method, url, body) {
  const opts = { method, headers: {} };
  if (body !== undefined) {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(url, opts);
  const text = await res.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = text; }
  if (!res.ok) {
    const detail = data && data.detail ? data.detail : `${res.status} ${res.statusText}`;
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
  }
  return data;
}

export const api = {
  system: () => request('GET', '/api/system'),
  catalog: () => request('GET', '/api/catalog'),
  addCard: (kind, card) => request('POST', `/api/${kind}`, card),
  deleteCard: (kind, id) => request('DELETE', `/api/${kind}/${encodeURIComponent(id)}`),
  parseLink: (link) => request('POST', '/api/parse-link', { link }),
  inspect: (source, format = 'auto') => request('POST', '/api/inspect', { source, format }),
  runs: () => request('GET', '/api/runs'),
  run: (id) => request('GET', `/api/runs/${id}`),
  startRun: (body) => request('POST', '/api/runs', body),
  cancelRun: (id) => request('POST', `/api/runs/${id}/cancel`),
  deleteRun: (id) => request('DELETE', `/api/runs/${id}`),
  frame: (id, params) => request('GET', `/api/runs/${id}/frame?` + new URLSearchParams(params)),
};
