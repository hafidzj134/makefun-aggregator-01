// free-model-aggregator-01: a deterministic free-model routing server.
//
// The server reads its brief (ten simulated providers, a sealed quality
// table, and the twenty-request workload manifest), probes every provider's
// catalog route, registers its port and catalog with the env, then serves
// the client's requests one at a time. Each request is answered by exactly
// one successful provider call, on a free model that meets the request's
// requirements, translated between the client's wire and the provider's.
//
// Routing is planned, not hopeful. The manifest is known before the first
// request arrives, and the requests arrive heaviest first, so the router
// walks a candidate list ordered by the quality table's rating and spends
// each provider's per-trial budget deliberately: a conservative planning cap
// per provider (the low end of its budget range) keeps early light requests
// from draining the providers the later ones still need, and a refusal —
// a scheduled failure, an outage, or a spent budget — moves the request to
// the next-best model on a different provider instead of sinking it. A
// provider that refuses at or beyond its planning cap, or three times in
// all, is treated as spent for the rest of the trial.
//
// Every choice derives from the brief, the requests and the provider
// answers seen so far. No wall clock, no randomness: a replay of the same
// trial makes the same calls and leaves the same state.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  dialBase,
  isRecord,
  probeProvider,
  ratingRow,
  readBrief,
  rowMeets,
  type Brief,
  type Catalog,
  type Json,
  type ProbeResult,
  type ProviderEntry,
  type RatingRow,
  type Requirement,
} from './lib/providers.ts';
import {
  answerToClient,
  anthropicErrorBody,
  anthropicModelList,
  anthropicStreamAsOpenAI,
  clientJsonMode,
  geminiStreamAsAnthropic,
  geminiStreamAsOpenAI,
  openAIErrorBody,
  openAIModelList,
  openAIStreamAsAnthropic,
  toAnthropicRequest,
  toGeminiRequest,
  toOpenAIRequest,
  type ClientApi,
  type ServedModel,
} from './lib/wire.ts';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const CALL_TIMEOUT_MS = 20000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;

/**
 * Conservative per-provider planning caps: the low end of each provider's
 * per-trial call budget range, as the kit's provider table publishes it.
 * Budgets are drawn per trial and never disclosed, so the router plans
 * against the low end and treats anything above it as failover headroom.
 */
const PLANNED_CAP: Record<string, number> = {
  groq: 9,
  cerebras: 5,
  gemini: 6,
  zai: 4,
  openrouter: 10,
  sambanova: 4,
  pollinations: 10,
  cloudflare: 7,
  ovh: 5,
  kilol: 12,
};
const DEFAULT_PLANNED_CAP = 8;
/** Refusals after which a provider is treated as spent even below its cap. */
const MAX_REFUSALS = 3;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief: Brief = readBrief(PROMPT_FILE);
let catalog: Catalog = { providers: [] };
/** Models each provider's catalog listed (empty when its probe failed). */
const listedByProvider = new Map<string, Set<string>>();
/** How many requests of the manifest have arrived; they arrive in order. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  const probes: ProbeResult[] =
    ENV_URL === ''
      ? brief.providers.map((provider) => ({ provider, ok: false, models: [] }))
      : await Promise.all(brief.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) {
    listedByProvider.set(probe.provider.id, new Set(probe.models.map((m) => m.id)));
  }
  catalog = buildCatalog(brief, probes);
}

// ---------------------------------------------------------------------------
// The router
// ---------------------------------------------------------------------------

interface Candidate {
  provider: ProviderEntry;
  model: string;
  rating: number;
  row: RatingRow;
}

/** Every free model the trial offers, with its quality row, in brief order. */
function allCandidates(): Candidate[] {
  const out: Candidate[] = [];
  for (const provider of brief.providers) {
    const listed = listedByProvider.get(provider.id);
    for (const row of brief.ratings) {
      if (row.provider !== provider.id || !row.free) continue;
      // A model the provider's own catalog does not list cannot be called;
      // when the probe failed there is no listing to check against.
      if (listed !== undefined && listed.size > 0 && !listed.has(row.model)) continue;
      out.push({ provider, model: row.model, rating: row.rating_milli, row });
    }
  }
  return out;
}

const CANDIDATES: Candidate[] = [];

/** Counted provider calls so far (every inference call names a free model). */
const countedCalls = new Map<string, number>();
/** Refusals seen per provider this trial. */
const refusalCounts = new Map<string, number>();
/** Providers treated as spent for the rest of the trial. */
const spentProviders = new Set<string>();

function plannedCap(providerId: string): number {
  return PLANNED_CAP[providerId] ?? DEFAULT_PLANNED_CAP;
}

/**
 * The candidates for one request, best first: rating, then the provider that
 * has spent fewest calls, then provider and model ids — a total order, so
 * the pick never depends on iteration luck. A request the client put in JSON
 * mode stays off the Anthropic wire, which cannot carry JSON mode.
 */
function rankedCandidates(requires: Requirement, api: ClientApi, body: Json): Candidate[] {
  const jsonMode = api === 'openai' && clientJsonMode(body);
  const eligible = CANDIDATES.filter((candidate) => {
    if (!rowMeets(candidate.row, requires)) return false;
    if (jsonMode && candidate.provider.wire === 'anthropic') return false;
    return true;
  });
  eligible.sort((a, b) => {
    if (a.rating !== b.rating) return b.rating - a.rating;
    const spentDiff = (countedCalls.get(a.provider.id) ?? 0) - (countedCalls.get(b.provider.id) ?? 0);
    if (spentDiff !== 0) return spentDiff;
    if (a.provider.id !== b.provider.id) return a.provider.id < b.provider.id ? -1 : 1;
    return a.model < b.model ? -1 : a.model > b.model ? 1 : 0;
  });
  return eligible;
}

/**
 * The next candidate to try for a request: first the unspent providers under
 * their planning cap, then any unspent provider, then — when nothing else is
 * left — a spent one, because a served request on a long shot still beats an
 * unserved one. A provider is never tried twice for the same request.
 */
function nextCandidate(ranked: Candidate[], tried: Set<string>): Candidate | null {
  for (const candidate of ranked) {
    if (tried.has(candidate.provider.id)) continue;
    if (spentProviders.has(candidate.provider.id)) continue;
    if ((countedCalls.get(candidate.provider.id) ?? 0) >= plannedCap(candidate.provider.id)) continue;
    return candidate;
  }
  for (const candidate of ranked) {
    if (tried.has(candidate.provider.id)) continue;
    if (spentProviders.has(candidate.provider.id)) continue;
    return candidate;
  }
  for (const candidate of ranked) {
    if (tried.has(candidate.provider.id)) continue;
    return candidate;
  }
  return null;
}

function noteRefusal(providerId: string): void {
  const refusals = (refusalCounts.get(providerId) ?? 0) + 1;
  refusalCounts.set(providerId, refusals);
  if ((countedCalls.get(providerId) ?? 0) >= plannedCap(providerId) || refusals >= MAX_REFUSALS) {
    spentProviders.add(providerId);
  }
}

// ---------------------------------------------------------------------------
// Provider calls
// ---------------------------------------------------------------------------

interface UpstreamOk {
  ok: true;
  response: Response;
}
interface UpstreamFail {
  ok: false;
  status: number;
}

/** One inference call to a candidate's provider, in the provider's own wire. */
async function callProvider(candidate: Candidate, api: ClientApi, body: Json, stream: boolean): Promise<UpstreamOk | UpstreamFail> {
  const provider = candidate.provider;
  const base = dialBase(ENV_URL, provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider) };
  let url: string;
  let payload: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    payload = toOpenAIRequest(body, api, candidate.model);
    if (stream) headers.accept = 'text/event-stream';
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    payload = toAnthropicRequest(body, api, candidate.model);
    if (stream) headers.accept = 'text/event-stream';
  } else {
    const verb = stream ? 'streamGenerateContent?alt=sse' : 'generateContent';
    url = `${base}/v1beta/models/${encodeURIComponent(candidate.model)}:${verb}`;
    payload = toGeminiRequest(body, api);
    if (stream) headers.accept = 'text/event-stream';
  }
  countedCalls.set(provider.id, (countedCalls.get(provider.id) ?? 0) + 1);
  let response: Response;
  try {
    response = await fetch(url, { method: 'POST', headers, body: JSON.stringify(payload), signal: AbortSignal.timeout(CALL_TIMEOUT_MS) });
  } catch {
    return { ok: false, status: 0 };
  }
  if (!response.ok) {
    // Drain the refusal so the connection is settled, then let the router move on.
    try {
      await response.text();
    } catch {
      // Nothing to drain: the refusal stands.
    }
    return { ok: false, status: response.status };
  }
  if (stream && response.body === null) return { ok: false, status: 0 };
  return { ok: true, response };
}

/** Copy a provider's event stream to the client untouched (same wire both sides). */
async function pipeRaw(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

async function pipeTranslated(events: AsyncGenerator<string>, res: ServerResponse): Promise<void> {
  for await (const event of events) res.write(event);
  res.end();
}

/** Send a successful provider stream to the client in the client's wire. */
async function sendStream(candidate: Candidate, api: ClientApi, response: Response, res: ServerResponse): Promise<void> {
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  const body = response.body;
  if (body === null) {
    res.end();
    return;
  }
  const wire = candidate.provider.wire;
  if (wire === 'openai' && api === 'openai') return pipeRaw(body, res);
  if (wire === 'anthropic' && api === 'anthropic') return pipeRaw(body, res);
  if (wire === 'openai' && api === 'anthropic') return pipeTranslated(openAIStreamAsAnthropic(body, candidate.model), res);
  if (wire === 'anthropic' && api === 'openai') return pipeTranslated(anthropicStreamAsOpenAI(body, candidate.model), res);
  if (wire === 'gemini' && api === 'openai') return pipeTranslated(geminiStreamAsOpenAI(body, candidate.model), res);
  return pipeTranslated(geminiStreamAsAnthropic(body, candidate.model), res);
}

// ---------------------------------------------------------------------------
// Client routes
// ---------------------------------------------------------------------------

async function readJsonBody(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendClientError(res: ServerResponse, api: ClientApi, status: number, message: string): void {
  if (api === 'anthropic') sendJson(res, status, anthropicErrorBody('api_error', message));
  else sendJson(res, status, openAIErrorBody(message, 'api_error', null));
}

/** Requirements from the request body itself, when the manifest has no row left for it. */
function requirementsFromBody(api: ClientApi, body: Json): Requirement {
  let vision = false;
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (!isRecord(block)) continue;
      if (block.type === 'image' || block.type === 'image_url') vision = true;
    }
  }
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: api === 'openai' && clientJsonMode(body),
  };
}

/** Serve one client request: plan, call, fail over, translate, answer. */
async function serve(api: ClientApi, req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJsonBody(req);
  if (body === null) return sendClientError(res, api, 400, 'the request body is not one JSON object');
  const entry = brief.manifest[manifestIndex];
  manifestIndex += 1;
  const requires = entry !== undefined ? entry.requires : requirementsFromBody(api, body);
  const stream = body.stream === true;
  const ranked = rankedCandidates(requires, api, body);
  const tried = new Set<string>();
  for (;;) {
    const candidate = nextCandidate(ranked, tried);
    if (candidate === null) return sendClientError(res, api, 502, 'no free model could serve this request');
    tried.add(candidate.provider.id);
    const attempt = await callProvider(candidate, api, body, stream);
    if (!attempt.ok) {
      noteRefusal(candidate.provider.id);
      continue;
    }
    if (stream) {
      await sendStream(candidate, api, attempt.response, res);
      return;
    }
    let answer: unknown;
    try {
      answer = await attempt.response.json();
    } catch {
      noteRefusal(candidate.provider.id);
      continue;
    }
    if (!isRecord(answer)) {
      noteRefusal(candidate.provider.id);
      continue;
    }
    sendJson(res, 200, answerToClient(candidate.provider.wire, answer, api, candidate.model));
    return;
  }
}

/** The free models of the registered catalog, in catalog order: GET /v1/models' rows. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const provider of catalog.providers) {
    for (const model of provider.models) {
      if (model.free) out.push({ id: model.id, provider: provider.id });
    }
  }
  return out;
}

function handleModels(req: IncomingMessage, res: ServerResponse): void {
  const models = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(models));
  else sendJson(res, 200, openAIModelList(models));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') return handleModels(req, res);
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') return serve('openai', req, res);
  if (req.method === 'POST' && url.pathname === '/v1/messages') return serve('anthropic', req, res);
  return sendClientError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, `no route ${req.method ?? ''} ${url.pathname}`);
}

// ---------------------------------------------------------------------------
// Trial plumbing: register, poll, exit
// ---------------------------------------------------------------------------

function listen(server: Server): Promise<number> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => {
      const address = server.address();
      if (address === null || typeof address === 'string') reject(new Error('the server has no port'));
      else resolve(address.port);
    });
  });
}

async function register(port: number): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work; the moment the env says the workload is done, exit 0. */
function pollWork(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status === 200) {
          const work: unknown = await res.json();
          if (isRecord(work) && work.done === true) {
            clearInterval(timer);
            server.close();
            process.exit(0);
          }
        }
      } catch {
        // The env is not reachable yet: the next tick retries.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  CANDIDATES.push(...allCandidates());
  if (process.argv.includes('--list')) {
    // The register body's catalog, byte for byte: print the one built object and stop.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        sendClientError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'the server failed to answer');
      } else {
        res.end();
      }
    });
  });
  const port = await listen(server);
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} providers and ${servedModels().length} free models`);
  await register(port);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  pollWork(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
