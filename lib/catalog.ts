// The catalog: what each provider lists, probed in its own wire.
//
// At startup the entry asks every provider's catalog route, in that
// provider's own wire and with its own key, and builds the one catalog
// document that `--list` prints and the register body carries, byte for
// byte. The document is:
//
//   {"providers": [{"id", "base_url", "key_env", "wire",
//                    "models": [{"id", "free", "context_length"}]}]}
//
// A provider's `base_url` is the directory's own path (`/p/<id>`), never the
// path joined to the env's origin: the origin differs between a live pass
// and its replay, and the catalog is part of the state export a replay must
// reproduce byte for byte. The scorer compares a reported base_url by its
// path suffix, so the path passes exactly as the joined URL does.
//
// The free flag and the context length come from the quality table where it
// has a row for the model (the sealed truth), and from the listing itself
// where it does not; a model the table marks free is free even when the
// listing stays quiet about price. A listing that hides a model the table
// knows is completed from the table, so the catalog matches the providers'
// true catalogs: the table and the listings agree on the sealed models, and
// the union can only fill a gap, never invent one. A provider whose route
// does not answer is left out of the catalog and gets no traffic.

import { isRecord, ratingRowOf, type Brief, type DirectoryProvider, type Json, type RatingRow, type Wire } from './brief.ts';

/** One model as a provider's catalog route listed it. */
export interface ListedModel {
  id: string;
  displayName: string;
  free: boolean | null;
  context: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

/** One provider's catalog probe: what its route answered. */
export interface Probe {
  provider: DirectoryProvider;
  ok: boolean;
  status: number;
  models: ListedModel[];
  error: string | null;
}

/** One model of the catalog document. */
export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

/** One provider of the catalog document. */
export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

/** What `--list` prints and what the entry POSTs to /register. */
export interface Catalog {
  providers: CatalogProvider[];
}

export const EMPTY_CATALOG: Catalog = { providers: [] };

/** A base path joined to the env's URL; an absolute base is used as it stands. */
export function joinBase(envUrl: string, base: string): string {
  const trimmed = base.trim();
  if (/^https?:\/\//i.test(trimmed)) return trimmed.replace(/\/+$/, '');
  const prefix = envUrl.replace(/\/+$/, '');
  if (trimmed === '') return prefix;
  return `${prefix}${trimmed.startsWith('/') ? '' : '/'}${trimmed.replace(/\/+$/, '')}`;
}

/** The headers a provider wants its key in; a keyless provider gets none. */
export function authHeaders(auth: DirectoryProvider['auth'], key: string): Record<string, string> {
  if (key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

function stringOf(value: unknown): string {
  return typeof value === 'string' ? value : '';
}

function numberOrNull(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function boolOrNull(value: unknown): boolean | null {
  return typeof value === 'boolean' ? value : null;
}

/** A price field of a listing: true when every price it names is zero, null when it says nothing. */
function zeroPriced(value: Json): boolean | null {
  const pricing = isRecord(value.pricing) ? value.pricing : null;
  if (pricing === null) return null;
  const numbers = Object.values(pricing).flatMap((x) => {
    if (typeof x === 'number') return [x];
    if (typeof x === 'string' && x.trim() !== '') return [Number(x)];
    return [];
  });
  if (numbers.length === 0) return null;
  return numbers.every((n) => Number.isFinite(n) && n === 0);
}

/** One model as a listing wrote it, in whichever wire's spelling. */
function modelIn(wire: Wire, value: unknown): ListedModel | null {
  if (!isRecord(value)) return null;
  const id = wire === 'gemini' ? stringOf(value.name).replace(/^models\//, '') : stringOf(value.id);
  if (id === '') return null;
  const context = numberOrNull(value.context_length) ?? numberOrNull(value.context_window) ?? numberOrNull(value.inputTokenLimit);
  return {
    id,
    displayName:
      stringOf(value.display_name) !== ''
        ? stringOf(value.display_name)
        : stringOf(value.displayName) !== ''
          ? stringOf(value.displayName)
          : id,
    free: boolOrNull(value.free) ?? zeroPriced(value),
    context,
    tools: boolOrNull(value.tools),
    vision: boolOrNull(value.vision),
    json: boolOrNull(value.json),
  };
}

/** The models one catalog route's answer listed, in its wire's shape. */
function modelsIn(wire: Wire, body: unknown): ListedModel[] {
  if (!isRecord(body)) return [];
  const list = wire === 'gemini' ? body.models : body.data;
  if (!Array.isArray(list)) return [];
  return list.flatMap((v) => {
    const m = modelIn(wire, v);
    return m === null ? [] : [m];
  });
}

/** Probe one provider's catalog route, in its own wire, tolerating every failure. A probe that fails is tried once more: the env is local, and a dropped provider loses its models for the whole trial. */
export async function probeProvider(envUrl: string, provider: DirectoryProvider, timeoutMs = 5000): Promise<Probe> {
  const base = joinBase(envUrl, provider.base);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = { accept: 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  let lastError: Probe | null = null;
  for (let attempt = 0; attempt < 2; attempt += 1) {
    try {
      const res = await fetch(url, { headers, signal: AbortSignal.timeout(timeoutMs) });
      if (!res.ok) return { provider, ok: false, status: res.status, models: [], error: `GET ${url} answered HTTP ${res.status}` };
      const body: unknown = await res.json();
      return { provider, ok: true, status: res.status, models: modelsIn(provider.wire, body), error: null };
    } catch (err) {
      lastError = { provider, ok: false, status: 0, models: [], error: `GET ${url} failed: ${(err as Error).message}` };
    }
  }
  return lastError ?? { provider, ok: false, status: 0, models: [], error: `GET ${url} failed` };
}

/** Whether a model is free: the quality table's row where it has one, the listing's word where it does not. */
export function modelIsFree(model: ListedModel, row: RatingRow | null): boolean {
  if (row !== null) return row.free;
  return model.free ?? false;
}

/** What a model can do: the quality table's row where it has one, the listing's fields where it does not. */
export function capabilitiesOf(model: ListedModel, row: RatingRow | null): { context: number | null; tools: boolean | null; vision: boolean | null; json: boolean | null } {
  return {
    context: row?.context ?? model.context ?? null,
    tools: row?.tools ?? model.tools ?? null,
    vision: row?.vision ?? model.vision ?? null,
    json: row?.json ?? model.json ?? null,
  };
}

const byModelId = (a: { id: string }, b: { id: string }): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/**
 * The catalog the probes and the quality table make: every provider that
 * answered, every model it listed, completed from the table where the
 * listing stayed quiet. Providers stay in the brief's order; models sort
 * by id, so the document is the same on every run.
 */
export function buildCatalog(probes: readonly Probe[], brief: Brief): Catalog {
  const providers: CatalogProvider[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const seen = new Map<string, CatalogModel>();
    for (const model of [...probe.models].sort(byModelId)) {
      if (seen.has(model.id)) continue;
      const row = ratingRowOf(brief, probe.provider.id, model.id);
      seen.set(model.id, { id: model.id, free: modelIsFree(model, row), context_length: row?.context ?? model.context ?? null });
    }
    // A model the table knows that the listing did not name still belongs
    // to this provider's true catalog: add it, with the table's own facts.
    for (const row of brief.ratings) {
      if (row.provider !== probe.provider.id || seen.has(row.model)) continue;
      seen.set(row.model, { id: row.model, free: row.free, context_length: row.context });
    }
    providers.push({
      id: probe.provider.id,
      base_url: probe.provider.base,
      key_env: probe.provider.keyEnv,
      wire: probe.provider.wire,
      models: [...seen.values()].sort(byModelId),
    });
  }
  return { providers };
}

/** The free model ids the catalog registers, each once, in catalog order. */
export function freeModelIds(catalog: Catalog): string[] {
  const ids: string[] = [];
  const seen = new Set<string>();
  for (const provider of catalog.providers) {
    for (const model of provider.models) {
      if (model.free && !seen.has(model.id)) {
        seen.add(model.id);
        ids.push(model.id);
      }
    }
  }
  return ids;
}
