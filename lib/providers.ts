// The brief, the provider directory, the quality table and the catalog.
//
// One trial hands this server a single brief file (its path is in
// $MAKEFUN_PROMPT_FILE). Inside the prose sit three JSON documents, found by
// their `schema` member:
//
//   - free-model-aggregator-01/providers/v1: the ten providers of the trial,
//     each with an id, a wire (openai, anthropic or gemini), an auth scheme,
//     the environment variable a real deployment would read its key from, the
//     key itself, and the base path its routes hang off under $MAKEFUN_ENV_URL.
//   - free-model-aggregator-01/ratings/v1: one row per model with its rating
//     in milli-points, its context length, the capabilities it offers (tools,
//     vision, JSON mode) and whether it is free. The brief's copy lists the
//     free models; the providers' own listings complete the picture.
//   - free-model-aggregator-01/workload/v1: the twenty requests the client
//     will send, in order, each with its API, stream flag, weight and the
//     requirements a model must meet to serve it.
//
// The catalog the server registers is built from the providers' own catalog
// routes, probed in each provider's wire, corrected by the quality table
// (whose free flag is authoritative where it has a row). A provider whose
// probe fails is rebuilt from the quality table alone, so one bad listing
// never costs the whole catalog entry.
//
// Everything in this file is a pure function of the brief and the listings:
// no clock, no randomness, no iteration over unordered structures without an
// explicit sort, so a replay rebuilds the identical catalog.

import { readFileSync } from 'node:fs';

export type Wire = 'openai' | 'anthropic' | 'gemini';
export type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

export const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
export const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
export const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

export type Json = Record<string, unknown>;

export const isRecord = (value: unknown): value is Json =>
  value !== null && typeof value === 'object' && !Array.isArray(value);

export interface ProviderEntry {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  /** The base path (`/p/<id>`) or absolute base URL from the brief. */
  base: string;
  key_env: string;
  key: string;
}

export interface RatingRow {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

export interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

export interface WorkloadRequest {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

export interface Brief {
  providers: ProviderEntry[];
  ratings: RatingRow[];
  /** The best rating per request id, as the brief's table states it. */
  best: Map<string, number>;
  manifest: WorkloadRequest[];
  error: string | null;
}

export interface ListedModel {
  id: string;
  free: boolean | null;
  context_length: number | null;
}

export interface ProbeResult {
  provider: ProviderEntry;
  ok: boolean;
  models: ListedModel[];
}

export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

export interface Catalog {
  providers: CatalogProvider[];
}

/** Join a provider's base path (or absolute base URL) to the env's origin. */
export function dialBase(envUrl: string, base: string): string {
  const trimmed = base.trim();
  if (/^https?:\/\//i.test(trimmed)) return trimmed.replace(/\/+$/, '');
  const origin = envUrl.replace(/\/+$/, '');
  if (trimmed === '') return origin;
  return `${origin}${trimmed.startsWith('/') ? '' : '/'}${trimmed.replace(/\/+$/, '')}`;
}

/** The header set a provider's auth scheme asks for (empty for `none`). */
export function authHeaders(provider: ProviderEntry): Record<string, string> {
  if (provider.key === '') return {};
  if (provider.auth === 'bearer') return { authorization: `Bearer ${provider.key}` };
  if (provider.auth === 'x-api-key') return { 'x-api-key': provider.key };
  if (provider.auth === 'x-goog-api-key') return { 'x-goog-api-key': provider.key };
  return {};
}

function asString(value: unknown): string {
  return typeof value === 'string' ? value : '';
}

function asFiniteNumber(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

/** The end index of the JSON object opening at `start`, or -1. Strings are skipped. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let quoted = false;
  for (let i = start; i < text.length; i += 1) {
    const ch = text[i];
    if (quoted) {
      if (ch === '\\') i += 1;
      else if (ch === '"') quoted = false;
    } else if (ch === '"') quoted = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** Every JSON object in `text` carrying a string `schema`, in file order. */
function schemaDocuments(text: string): Json[] {
  const found: Json[] = [];
  for (let at = text.indexOf('{'); at !== -1; at = text.indexOf('{', at + 1)) {
    const end = objectEnd(text, at);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(at, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') found.push(value);
    } catch {
      // Not a standalone JSON object: keep scanning.
    }
  }
  return found;
}

function parseProvider(value: unknown): ProviderEntry | null {
  if (!isRecord(value)) return null;
  const id = asString(value.id);
  const wire = value.wire === 'openai' || value.wire === 'anthropic' || value.wire === 'gemini' ? value.wire : null;
  if (id === '' || wire === null) return null;
  const auth =
    value.auth === 'bearer' || value.auth === 'x-api-key' || value.auth === 'x-goog-api-key' || value.auth === 'none'
      ? value.auth
      : 'bearer';
  const base =
    asString(value.path) !== ''
      ? asString(value.path)
      : asString(value.base_url) !== ''
        ? asString(value.base_url)
        : asString(value.base_path) !== ''
          ? asString(value.base_path)
          : asString(value.base);
  return {
    id,
    name: asString(value.name) !== '' ? asString(value.name) : id,
    wire,
    auth,
    base,
    key_env: asString(value.key_env),
    key: asString(value.key),
  };
}

function parseRating(value: unknown): RatingRow | null {
  if (!isRecord(value)) return null;
  const model = asString(value.model);
  const provider = asString(value.provider);
  if (model === '' || provider === '') return null;
  return {
    model,
    provider,
    rating_milli: asFiniteNumber(value.rating_milli) ?? 0,
    context_length: asFiniteNumber(value.context_length),
    tools: value.tools === true,
    vision: value.vision === true,
    json: value.json === true,
    free: value.free === true,
  };
}

function parseWorkloadRequest(value: unknown): WorkloadRequest | null {
  if (!isRecord(value)) return null;
  const id = asString(value.id);
  if (id === '') return null;
  const requires = isRecord(value.requires) ? value.requires : {};
  const minContext = asFiniteNumber(requires.min_context);
  return {
    id,
    api: value.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: value.stream === true,
    weight: value.weight === 2 ? 2 : 1,
    requires: {
      tools: requires.tools === true,
      vision: requires.vision === true,
      json: requires.json === true,
      ...(minContext !== null ? { min_context: minContext } : {}),
    },
  };
}

/** Read the brief at `path`; on failure the brief is empty and `error` says why. */
export function readBrief(path: string): Brief {
  const empty: Brief = { providers: [], ratings: [], best: new Map(), manifest: [], error: null };
  if (path === '') return { ...empty, error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { ...empty, error: `cannot read the brief: ${(err as Error).message}` };
  }
  const docs = schemaDocuments(text);
  const directory = docs.find((d) => d.schema === PROVIDERS_SCHEMA && Array.isArray(d.providers));
  if (directory === undefined) return { ...empty, error: `the brief holds no ${PROVIDERS_SCHEMA} document` };
  const providers = (directory.providers as unknown[]).flatMap((v) => {
    const parsed = parseProvider(v);
    return parsed === null ? [] : [parsed];
  });
  const ratingsDoc = docs.find((d) => d.schema === RATINGS_SCHEMA && Array.isArray(d.ratings));
  const ratings =
    ratingsDoc === undefined
      ? []
      : (ratingsDoc.ratings as unknown[]).flatMap((v) => {
          const parsed = parseRating(v);
          return parsed === null ? [] : [parsed];
        });
  const best = new Map<string, number>();
  if (ratingsDoc !== undefined && Array.isArray(ratingsDoc.best)) {
    for (const entry of ratingsDoc.best as unknown[]) {
      if (!isRecord(entry)) continue;
      const id = asString(entry.id);
      const rating = asFiniteNumber(entry.rating_milli);
      if (id !== '' && rating !== null) best.set(id, rating);
    }
  }
  const workloadDoc = docs.find((d) => d.schema === WORKLOAD_SCHEMA && Array.isArray(d.manifest));
  const manifest =
    workloadDoc === undefined
      ? []
      : (workloadDoc.manifest as unknown[]).flatMap((v) => {
          const parsed = parseWorkloadRequest(v);
          return parsed === null ? [] : [parsed];
        });
  return { providers, ratings, best, manifest, error: null };
}

/** The quality table's row for one provider's model, or null. */
export function ratingRow(brief: Brief, providerId: string, modelId: string): RatingRow | null {
  return brief.ratings.find((r) => r.provider === providerId && r.model === modelId) ?? null;
}

/**
 * Whether a model is free: the quality table decides where it has a row (it
 * is the sealed truth), the provider's own listing where it does not, and a
 * model neither source calls free is treated as paid and never routed.
 */
export function isFreeModel(row: RatingRow | null, listed: ListedModel | null): boolean {
  if (row !== null) return row.free;
  if (listed !== null) return listed.free ?? false;
  return false;
}

/** Whether a quality row meets one request's requirements. */
export function rowMeets(row: RatingRow, requires: Requirement): boolean {
  if (requires.tools === true && !row.tools) return false;
  if (requires.vision === true && !row.vision) return false;
  if (requires.json === true && !row.json) return false;
  if (requires.min_context !== undefined && (row.context_length === null || row.context_length < requires.min_context)) return false;
  return true;
}

function listedModel(wire: Wire, value: unknown): ListedModel | null {
  if (!isRecord(value)) return null;
  const id = wire === 'gemini' ? asString(value.name).replace(/^models\//, '') : asString(value.id);
  if (id === '') return null;
  return {
    id,
    free: typeof value.free === 'boolean' ? value.free : null,
    context_length: asFiniteNumber(value.context_length) ?? asFiniteNumber(value.context_window) ?? asFiniteNumber(value.inputTokenLimit),
  };
}

/** Probe one provider's catalog route in its own wire; failures come back as `ok: false`. */
export async function probeProvider(envUrl: string, provider: ProviderEntry, timeoutMs: number): Promise<ProbeResult> {
  const base = dialBase(envUrl, provider.base);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = { accept: 'application/json', ...authHeaders(provider) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  try {
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(timeoutMs) });
    if (!res.ok) return { provider, ok: false, models: [] };
    const body: unknown = await res.json();
    if (!isRecord(body)) return { provider, ok: false, models: [] };
    const list = provider.wire === 'gemini' ? body.models : body.data;
    if (!Array.isArray(list)) return { provider, ok: false, models: [] };
    const models = list.flatMap((v) => {
      const parsed = listedModel(provider.wire, v);
      return parsed === null ? [] : [parsed];
    });
    return { provider, ok: true, models };
  } catch {
    return { provider, ok: false, models: [] };
  }
}

const byModelId = (a: { id: string }, b: { id: string }): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/**
 * The one catalog document: what `--list` prints and what the register body
 * carries, byte for byte. Providers appear in the brief's order; models are
 * sorted by id. A provider whose probe failed is rebuilt from the quality
 * table's free rows for it, so its free set is still exact.
 */
export function buildCatalog(brief: Brief, probes: readonly ProbeResult[]): Catalog {
  const providers: CatalogProvider[] = [];
  for (const probe of probes) {
    const models = new Map<string, CatalogModel>();
    if (probe.ok) {
      for (const listed of probe.models) {
        const row = ratingRow(brief, probe.provider.id, listed.id);
        models.set(listed.id, {
          id: listed.id,
          free: isFreeModel(row, listed),
          context_length: row?.context_length ?? listed.context_length ?? null,
        });
      }
    }
    if (models.size === 0) {
      for (const row of brief.ratings) {
        if (row.provider !== probe.provider.id) continue;
        models.set(row.model, { id: row.model, free: row.free, context_length: row.context_length });
      }
    }
    providers.push({
      id: probe.provider.id,
      base_url: probe.provider.base,
      key_env: probe.provider.key_env,
      wire: probe.provider.wire,
      models: [...models.values()].sort(byModelId),
    });
  }
  return { providers };
}
