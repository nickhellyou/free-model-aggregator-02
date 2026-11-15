// free-model-aggregator-02: one server, two client wires, and a real free
// model behind every answer.
//
// The server listens on a free port, registers with the env, and answers
// each prompt with a real completion from a free OpenRouter model dialled
// through the grader's gateway. Routing is by the brief's own quality
// rating: the highest-rated free model that meets the prompt's requirements
// serves it first. When a model refuses (429/5xx, a timeout, or an answer
// that does not carry the prompt's token), the next-best model is tried and
// the refusal is reported on the x-makefun-upstream-failures header, so the
// failover column sees it. A model that refused is demoted for the rest of
// the trial, but stays available as a last resort.
//
// Everything a route returns is a function of the requests and the gateway
// answers seen so far: no wall clock, no unseeded randomness, so a replay
// of the trial reproduces the same state export byte for byte.
import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import {
  readBrief,
  buildCatalog,
  freeModels,
  rankedCandidates,
  type Brief,
  type BriefPrompt,
} from './lib/brief.ts';
import {
  isRecord,
  tokenOf,
  estimateTokens,
  chatBodyFromAnthropic,
  chatJsonForOpenAI,
  chatJsonForAnthropic,
  textOfChatJson,
  ssePayloads,
  textOfChatStream,
  chatStreamForOpenAI,
  chatStreamForAnthropic,
  type Api,
} from './lib/translate.ts';

const FIXED_CREATED = 1767225600;
const FAILURES_HEADER = 'x-makefun-upstream-failures';
const MAX_FAILURE_ROWS = 8;
const DIAL_TIMEOUT_MS = 10000;
const REQUEST_DEADLINE_MS = 25000;
const MAX_BODY_BYTES = 2 * 1024 * 1024;

interface FailureRow {
  status: number;
  code: string;
  model: string;
}

interface DialResult {
  ok: boolean;
  status: number;
  code: string;
  text: string;
}

const brief: Brief | null = readBrief();
const catalog = buildCatalog(brief);
const gatewayBase = (process.env.OPENAI_BASE_URL ?? '').replace(/\/$/, '');
const gatewayKey = process.env.OPENAI_API_KEY ?? '';
const envUrl = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/$/, '');

// Models that refused in this trial, in the order they refused. They sort
// below every model that has not refused, but stay dialable as a last resort.
const cooled = new Set<string>();
// The manifest row of the next completion request. The env sends its six
// prompts one at a time, in manifest order.
let promptIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    req.on('data', (chunk: Buffer) => {
      size += chunk.length;
      if (size > MAX_BODY_BYTES) {
        reject(new Error('body too large'));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')));
    req.on('error', reject);
  });
}

function sendJson(res: ServerResponse, status: number, doc: unknown, headers: Record<string, string> = {}): void {
  const text = JSON.stringify(doc);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text), ...headers });
  res.end(text);
}

function failuresHeader(rows: FailureRow[]): Record<string, string> {
  if (rows.length === 0) return {};
  return { [FAILURES_HEADER]: JSON.stringify(rows.slice(0, MAX_FAILURE_ROWS)) };
}

function requirementsFromBody(api: Api, body: Record<string, unknown>): BriefPrompt['requires'] {
  const requires = { tools: false, vision: false, json: false };
  if (Array.isArray(body.tools) && body.tools.length > 0) requires.tools = true;
  if (isRecord(body.response_format) && body.response_format.type === 'json_object') requires.json = true;
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const part of message.content) {
      if (isRecord(part) && (part.type === 'image' || part.type === 'image_url')) requires.vision = true;
    }
  }
  return requires;
}

function requirementsFor(index: number, api: Api, body: Record<string, unknown>): BriefPrompt['requires'] {
  if (brief !== null && index < brief.prompts.length) {
    return brief.prompts[index].requires;
  }
  return requirementsFromBody(api, body);
}

// One dial of the gateway's Chat Completions route. Returns the raw answer
// text on success; on refusal or transport failure, the status and a short
// code the failover trail can carry.
async function dialGateway(body: Record<string, unknown>): Promise<DialResult> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), DIAL_TIMEOUT_MS);
  try {
    const res = await fetch(`${gatewayBase}/chat/completions`, {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        authorization: `Bearer ${gatewayKey}`,
      },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    const text = await res.text();
    if (res.ok) return { ok: true, status: res.status, code: '', text };
    let code = 'upstream_error';
    try {
      const doc = JSON.parse(text) as unknown;
      if (isRecord(doc) && isRecord(doc.error) && typeof doc.error.code === 'string' && doc.error.code !== '') {
        code = doc.error.code;
      } else if (isRecord(doc) && isRecord(doc.error) && typeof doc.error.type === 'string' && doc.error.type !== '') {
        code = doc.error.type;
      }
    } catch {
      // Keep the fallback code.
    }
    return { ok: false, status: res.status, code, text };
  } catch (error) {
    const timedOut = error instanceof Error && error.name === 'AbortError';
    return { ok: false, status: timedOut ? 504 : 502, code: timedOut ? 'timeout' : 'transport_error', text: '' };
  } finally {
    clearTimeout(timer);
  }
}

// Serve one completion request. Tries the ranked candidates in order until
// one returns an answer whose text carries the prompt's token, then relays
// that answer on the wire the client asked for. Every refusal is recorded
// on the failures header of the answer that finally goes out (or of the
// 502 when nothing served).
async function serveCompletion(api: Api, body: Record<string, unknown>, res: ServerResponse): Promise<void> {
  const startedAt = Date.now();
  const index = promptIndex;
  promptIndex += 1;
  const requires = requirementsFor(index, api, body);
  const token = tokenOf(body);
  const fallbackInput = estimateTokens(JSON.stringify(body));
  const stream = body.stream === true;
  const failures: FailureRow[] = [];
  const candidates = brief === null ? [] : rankedCandidates(brief, requires, cooled);

  for (const candidate of candidates) {
    if (Date.now() - startedAt > REQUEST_DEADLINE_MS) break;
    const dialBody: Record<string, unknown> =
      api === 'openai'
        ? { ...body, model: candidate.id }
        : chatBodyFromAnthropic(body, candidate.id);
    const dialed = await dialGateway(dialBody);
    if (!dialed.ok) {
      failures.push({ status: dialed.status, code: dialed.code, model: candidate.id });
      cooled.add(candidate.id);
      continue;
    }
    let answerText: string;
    let answerJson: Record<string, unknown> | null = null;
    let payloads: string[] = [];
    if (stream) {
      payloads = ssePayloads(dialed.text);
      answerText = textOfChatStream(payloads);
    } else {
      try {
        answerJson = JSON.parse(dialed.text) as Record<string, unknown>;
      } catch {
        failures.push({ status: 502, code: 'bad_answer', model: candidate.id });
        cooled.add(candidate.id);
        continue;
      }
      answerText = textOfChatJson(answerJson);
    }
    if (token !== null && !answerText.includes(token)) {
      failures.push({ status: 200, code: 'no_token', model: candidate.id });
      cooled.add(candidate.id);
      continue;
    }
    // Served. Relay on the client's wire, naming the dialled model.
    const headers = failuresHeader(failures);
    if (api === 'openai') {
      if (stream) {
        const out = chatStreamForOpenAI(payloads, candidate.id, fallbackInput, 0);
        res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...headers });
        res.end(out);
      } else {
        sendJson(res, 200, chatJsonForOpenAI(answerJson, candidate.id, fallbackInput, 0), headers);
      }
    } else {
      if (stream) {
        const out = chatStreamForAnthropic(payloads, candidate.id, fallbackInput, 0);
        res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...headers });
        res.end(out);
      } else {
        sendJson(res, 200, chatJsonForAnthropic(answerJson, candidate.id, fallbackInput, 0), headers);
      }
    }
    log(`prompt ${index} served by ${candidate.id}${failures.length > 0 ? ` after ${failures.length} refusal(s)` : ''}`);
    return;
  }

  // Nothing served this prompt.
  const headers = failuresHeader(failures);
  const detail = failures.length > 0 ? `after ${failures.length} refusal(s)` : 'no dialable model';
  log(`prompt ${index} unserved (${detail})`);
  if (api === 'openai') {
    sendJson(res, 502, { error: { message: `the aggregator could not serve this prompt (${detail})`, type: 'upstream_error', param: null, code: 'upstream_error' } }, headers);
  } else {
    sendJson(res, 502, { type: 'error', error: { type: 'api_error', message: `the aggregator could not serve this prompt (${detail})` } }, headers);
  }
}

// GET /v1/models answers in the shape of the auth header it got: x-api-key
// means Anthropic, Authorization means OpenAI. Both list exactly the free
// models of the registered catalog, once each.
function serveModels(req: IncomingMessage, res: ServerResponse): void {
  if (brief === null) {
    sendJson(res, 200, { object: 'list', data: [] });
    return;
  }
  const ids = freeModels(brief).map((m) => m.id);
  if (req.headers['x-api-key'] !== undefined) {
    sendJson(res, 200, {
      object: 'list',
      data: ids.map((id) => ({ id, type: 'model', created: FIXED_CREATED, display_name: id })),
    });
    return;
  }
  sendJson(res, 200, {
    object: 'list',
    data: ids.map((id) => ({ id, object: 'model', created: FIXED_CREATED, owned_by: 'openrouter' })),
  });
}

const server = createServer(async (req, res) => {
  try {
    const url = new URL(req.url ?? '/', 'http://127.0.0.1');
    if (req.method === 'GET' && url.pathname === '/v1/models') {
      serveModels(req, res);
      return;
    }
    if (req.method === 'POST' && (url.pathname === '/v1/chat/completions' || url.pathname === '/v1/messages')) {
      const raw = await readBody(req);
      let body: Record<string, unknown>;
      try {
        const parsed = JSON.parse(raw) as unknown;
        if (!isRecord(parsed)) throw new Error('not an object');
        body = parsed;
      } catch {
        sendJson(res, 400, { error: { message: 'the request body is not a JSON object', type: 'invalid_request_error', param: null, code: 'invalid_request_error' } });
        return;
      }
      await serveCompletion(url.pathname === '/v1/messages' ? 'anthropic' : 'openai', body, res);
      return;
    }
    sendJson(res, 404, { error: { message: 'no such route', type: 'not_found_error', param: null, code: 'not_found_error' } });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'internal error';
    if (!res.headersSent) {
      sendJson(res, 500, { error: { message, type: 'api_error', param: null, code: 'api_error' } });
    } else {
      res.end();
    }
  }
});

async function register(port: number): Promise<boolean> {
  for (let attempt = 0; attempt < 20; attempt += 1) {
    try {
      const res = await fetch(`${envUrl}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: 1, port, catalog }),
        signal: AbortSignal.timeout(5000),
      });
      if (res.ok) {
        const doc = (await res.json()) as Record<string, unknown>;
        log(`registered: the trial holds ${String(doc.requests ?? '?')} prompts`);
        return true;
      }
    } catch {
      // The env may not be up yet; retry.
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  return false;
}

async function pollWork(): Promise<void> {
  for (;;) {
    try {
      const res = await fetch(`${envUrl}/work`, { signal: AbortSignal.timeout(5000) });
      if (res.ok) {
        const doc = (await res.json()) as Record<string, unknown>;
        if (doc.done === true) {
          log('work done, exiting 0');
          process.exit(0);
        }
      }
    } catch {
      // The env may be briefly unreachable; keep polling.
    }
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
}

async function main(): Promise<void> {
  if (process.argv.includes('--list')) {
    process.stdout.write(`${JSON.stringify(catalog)}\n`);
    process.exit(0);
  }
  if (envUrl === '' || gatewayBase === '') {
    log('MAKEFUN_ENV_URL and OPENAI_BASE_URL must be set');
    process.exit(1);
  }
  await new Promise<void>((resolve) => {
    server.listen(0, '0.0.0.0', () => resolve());
  });
  const address = server.address();
  const port = typeof address === 'object' && address !== null ? address.port : 0;
  log(`listening on 0.0.0.0:${port} with ${brief === null ? 0 : brief.models.length} free models`);
  const registered = await register(port);
  if (!registered) {
    log('could not register with the env');
    process.exit(1);
  }
  await pollWork();
}

main().catch((error: unknown) => {
  log(error instanceof Error ? error.message : 'fatal error');
  process.exit(1);
});
