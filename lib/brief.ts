// The brief: the trial's prompt file, three JSON documents wrapped in prose.
//
// The file the trial hands the entry (at $MAKEFUN_PROMPT_FILE) is prose with
// three JSON documents embedded in it, found here by their `schema` member:
//
//   - the provider directory, `free-model-aggregator-01/providers/v1`: the ten
//     providers, each with an id, a wire (openai, anthropic or gemini), an auth
//     scheme, the key variable a real deployment would read, the key the
//     simulated service takes, and the base path to join to $MAKEFUN_ENV_URL.
//   - the quality table, `free-model-aggregator-01/ratings/v1`: every free
//     model's rating in milli-points, what it can do (tools, vision, JSON,
//     context length) and the best rating per request. The table is the sealed
//     truth about capabilities and free flags; the scorer ranks an answer by
//     its rating, and a paid model is never a legal pick.
//   - the workload manifest, `free-model-aggregator-01/workload/v1`: the
//     twenty requests the client will send, in order, each with its api
//     (openai or anthropic), whether it streams, its weight, and the
//     requirements a model must meet to serve it.
//
// Nothing here reads the wall clock or any randomness: the brief is read once,
// and every later choice is a function of it and of the providers' answers.

import { readFileSync } from 'node:fs';

export type Wire = 'openai' | 'anthropic' | 'gemini';
export type AuthKind = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

/** A JSON object, the only shape this file treats as structured data. */
export type Json = Record<string, unknown>;

export function isRecord(value: unknown): value is Json {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function asString(value: unknown): string {
  return typeof value === 'string' ? value : '';
}

function asNumberOrNull(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

/** One provider as the brief's directory lists it. */
export interface DirectoryProvider {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthKind;
  /** The base path (or absolute URL) the provider's routes hang off. */
  base: string;
  keyEnv: string;
  key: string;
}

/** One row of the sealed quality table: a free model's public standing. */
export interface RatingRow {
  provider: string;
  model: string;
  rating: number;
  context: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

/** What one request needs from the model that serves it. */
export interface Needs {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  minContext?: number;
}

/** One request of the workload manifest, in the order the client sends them. */
export interface WorkloadItem {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: number;
  needs: Needs;
}

/** The brief as the entry reads it. `error` is set when it could not be read. */
export interface Brief {
  providers: DirectoryProvider[];
  ratings: RatingRow[];
  /** The best free rating per request id, for the quality column's bound. */
  best: Map<string, number>;
  manifest: WorkloadItem[];
  error: string | null;
}

const DIRECTORY_SCHEMA = 'free-model-aggregator-01/providers/v1';
const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

/**
 * The index of the `}` that closes the object opening at `start`, or -1.
 * String contents are skipped, so a brace inside a string never counts.
 */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i += 1) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** Every JSON object in `text` that carries a string `schema`, in file order. */
function documentsIn(text: string): Json[] {
  const found: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') found.push(value);
    } catch {
      // Not a document on its own: look on.
    }
  }
  return found;
}

function wireOf(value: unknown): Wire | null {
  return value === 'openai' || value === 'anthropic' || value === 'gemini' ? value : null;
}

function authOf(value: unknown): AuthKind {
  if (value === 'bearer' || value === 'x-api-key' || value === 'x-goog-api-key' || value === 'none') return value;
  return 'bearer';
}

function providerOf(value: unknown): DirectoryProvider | null {
  if (!isRecord(value)) return null;
  const id = asString(value.id);
  const wire = wireOf(value.wire);
  if (id === '' || wire === null) return null;
  // The directory writes `path` (/p/<id>, joined to $MAKEFUN_ENV_URL); a
  // directory that spells an absolute base instead may use base_path,
  // base_url or base. Take the first non-empty one, `path` first.
  const base =
    asString(value.path) !== ''
      ? asString(value.path)
      : asString(value.base_path) !== ''
        ? asString(value.base_path)
        : asString(value.base_url) !== ''
          ? asString(value.base_url)
          : asString(value.base);
  return {
    id,
    name: asString(value.name) !== '' ? asString(value.name) : id,
    wire,
    auth: authOf(value.auth),
    base,
    keyEnv: asString(value.key_env),
    key: asString(value.key),
  };
}

function ratingOf(value: unknown): RatingRow | null {
  if (!isRecord(value)) return null;
  const model = asString(value.model);
  const provider = asString(value.provider);
  if (model === '' || provider === '') return null;
  return {
    provider,
    model,
    rating: typeof value.rating_milli === 'number' && Number.isFinite(value.rating_milli) ? value.rating_milli : 0,
    context: asNumberOrNull(value.context_length),
    tools: value.tools === true,
    vision: value.vision === true,
    json: value.json === true,
    free: value.free === true,
  };
}

function needsOf(value: unknown): Needs {
  if (!isRecord(value)) return {};
  const needs: Needs = {};
  if (value.tools === true) needs.tools = true;
  if (value.vision === true) needs.vision = true;
  if (value.json === true) needs.json = true;
  if (typeof value.min_context === 'number' && Number.isFinite(value.min_context)) needs.minContext = value.min_context;
  return needs;
}

function workloadItemOf(value: unknown): WorkloadItem | null {
  if (!isRecord(value)) return null;
  const id = asString(value.id);
  if (id === '') return null;
  return {
    id,
    api: value.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: value.stream === true,
    weight: value.weight === 2 ? 2 : 1,
    needs: needsOf(value.requires),
  };
}

/** The brief at `path`, its three documents found by schema. */
export function readBrief(path: string): Brief {
  const empty: Brief = { providers: [], ratings: [], best: new Map(), manifest: [], error: null };
  if (path === '') return { ...empty, error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { ...empty, error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  const directoryDoc = docs.find((d) => d.schema === DIRECTORY_SCHEMA && Array.isArray(d.providers)) ?? null;
  if (directoryDoc === null) {
    return { ...empty, error: `the brief at ${path} holds no ${DIRECTORY_SCHEMA} document` };
  }
  const providers = (directoryDoc.providers as unknown[]).flatMap((v) => {
    const p = providerOf(v);
    return p === null ? [] : [p];
  });
  const ratingsDoc = docs.find((d) => d.schema === RATINGS_SCHEMA && Array.isArray(d.ratings)) ?? null;
  const ratings: RatingRow[] =
    ratingsDoc === null
      ? []
      : (ratingsDoc.ratings as unknown[]).flatMap((v) => {
          const r = ratingOf(v);
          return r === null ? [] : [r];
        });
  const best = new Map<string, number>();
  if (ratingsDoc !== null && Array.isArray(ratingsDoc.best)) {
    for (const v of ratingsDoc.best as unknown[]) {
      if (isRecord(v) && asString(v.id) !== '') best.set(asString(v.id), asNumberOrNull(v.rating_milli) ?? 0);
    }
  }
  const workloadDoc = docs.find((d) => d.schema === WORKLOAD_SCHEMA && Array.isArray(d.manifest)) ?? null;
  const manifest: WorkloadItem[] =
    workloadDoc === null
      ? []
      : (workloadDoc.manifest as unknown[]).flatMap((v) => {
          const w = workloadItemOf(v);
          return w === null ? [] : [w];
        });
  return { providers, ratings, best, manifest, error: null };
}

/** The quality table's row for one provider's model, when it has one. */
export function ratingRowOf(brief: Brief, providerId: string, modelId: string): RatingRow | null {
  return brief.ratings.find((r) => r.provider === providerId && r.model === modelId) ?? null;
}

/** The best free rating the table gives one request, or 0 when it names none. */
export function bestRatingOf(brief: Brief, requestId: string): number {
  return brief.best.get(requestId) ?? 0;
}
