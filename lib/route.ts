// Routing: which free model serves a request, and what to do when it refuses.
//
// The entry knows the whole manifest in advance, but it never knows a
// provider's per-trial budget, its failure schedule or its outage: those are
// drawn per trial and never published. So routing is greedy by rating with
// reactive failover, and every choice is a function of the brief and of the
// providers' answers so far:
//
//   - the candidates for a request are every free model that meets its
//     requirements, best rating first; ties prefer the provider whose wire
//     matches the client's (a call that needs no translation cannot lose a
//     field to one), then the provider id, then the model id.
//   - a JSON-mode request never goes to an Anthropic-wire provider: that
//     wire cannot carry response_format, and the provider would echo JSON
//     mode absent. OpenAI and Gemini wires can carry it.
//   - a provider that refuses is counted, and the request is tried on the
//     next candidate. A provider that has refused twice running is skipped
//     on the first pass for later requests: a budget spent or an outage
//     never recovers, and two refusals in a row are far likelier to be that
//     than two scheduled failures. Skipped providers are still tried on a
//     second pass, so a request is only lost when every candidate refuses.
//   - the first success ends the request: exactly one provider call may
//     answer it, and the answer is the one that counts.
//
// One provider serves one call at a time here because the client sends one
// request at a time and each request's calls run in sequence, so the
// concurrent_requests refusal never arises from this entry.

import {
  anthropicJsonAnswer,
  anthropicStreamAnswer,
  anthropicToGemini,
  anthropicToOpenAI,
  geminiJsonAnswer,
  geminiStreamAnswer,
  openAIJsonAnswer,
  openAIStreamAnswer,
  openAIToAnthropicRequest,
  openAIToGemini,
  type ProviderAnswer,
} from './wire.ts';
import {
  authHeaders,
  capabilitiesOf,
  joinBase,
  modelIsFree,
  type Probe,
} from './catalog.ts';
import { isRecord, ratingRowOf, type Brief, type Json, type Needs } from './brief.ts';

/** How long one provider call may take, in milliseconds. */
const CALL_TIMEOUT_MS = 10000;
/** How many refusals in a row mark a provider skipped on the first pass. */
const SKIP_AFTER_FAILURES = 2;

/** One model a request may be served by, with its provider and rating. */
export interface Candidate {
  providerId: string;
  modelId: string;
  wire: 'openai' | 'anthropic' | 'gemini';
  rating: number;
}

/** Whether a model's capabilities meet one request's requirements. */
function meetsNeeds(needs: Needs, caps: { context: number | null; tools: boolean | null; vision: boolean | null; json: boolean | null }): boolean {
  if (needs.tools === true && caps.tools !== true) return false;
  if (needs.vision === true && caps.vision !== true) return false;
  if (needs.json === true && caps.json !== true) return false;
  if (needs.minContext !== undefined && (caps.context === null || caps.context < needs.minContext)) return false;
  return true;
}

/**
 * The candidates for one request, best first. A JSON-mode OpenAI request
 * drops Anthropic-wire providers; every other request keeps all three.
 */
export function candidatesFor(brief: Brief, probes: readonly Probe[], clientApi: 'openai' | 'anthropic', needs: Needs, jsonMode: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (jsonMode && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingRowOf(brief, probe.provider.id, model.id);
      if (!modelIsFree(model, row)) continue;
      if (!meetsNeeds(needs, capabilitiesOf(model, row))) continue;
      out.push({ providerId: probe.provider.id, modelId: model.id, wire: probe.provider.wire, rating: row?.rating ?? 0 });
    }
    // A model the table knows that the listing did not name is still a
    // legal pick: the table is the sealed truth about what the provider
    // serves, and the env answers it.
    for (const row of brief.ratings) {
      if (row.provider !== probe.provider.id || !row.free) continue;
      if (probe.models.some((m) => m.id === row.model)) continue;
      if (!meetsNeeds(needs, { context: row.context, tools: row.tools, vision: row.vision, json: row.json })) continue;
      out.push({ providerId: probe.provider.id, modelId: row.model, wire: probe.provider.wire, rating: row.rating });
    }
  }
  const wireRank = (wire: Candidate['wire']): number => (clientApi === 'openai' ? (wire === 'openai' ? 0 : wire === 'gemini' ? 1 : 2) : wire === 'anthropic' ? 0 : wire === 'openai' ? 1 : 2);
  out.sort((a, b) => {
    if (a.rating !== b.rating) return b.rating - a.rating;
    const wa = wireRank(a.wire);
    const wb = wireRank(b.wire);
    if (wa !== wb) return wa - wb;
    if (a.providerId !== b.providerId) return a.providerId < b.providerId ? -1 : 1;
    return a.modelId < b.modelId ? -1 : a.modelId > b.modelId ? 1 : 0;
  });
  return out;
}

/** What one provider call came back with. */
export type CallResult =
  | { ok: true; answer: ProviderAnswer }
  | { ok: false; status: number; code: string };

function failureOf(status: number, bodyText: string): { status: number; code: string } {
  let code = '';
  try {
    const parsed: unknown = JSON.parse(bodyText);
    if (isRecord(parsed) && isRecord(parsed.error)) {
      if (typeof parsed.error.code === 'string') code = parsed.error.code;
      else if (typeof parsed.error.type === 'string') code = parsed.error.type;
      else if (typeof parsed.error.status === 'string') code = parsed.error.status;
    }
  } catch {
    // Not JSON: the status alone says it.
  }
  return { status, code };
}

/**
 * One call to one provider's inference route, in its own wire, returning
 * the normalised answer or the refusal. The body is the client's request
 * rewritten for that wire; a streamed call asks for server-sent events.
 */
export async function callProvider(
  envUrl: string,
  probes: readonly Probe[],
  candidate: Candidate,
  clientApi: 'openai' | 'anthropic',
  body: Json,
  stream: boolean,
): Promise<CallResult> {
  const probe = probes.find((p) => p.provider.id === candidate.providerId);
  if (probe === undefined) return { ok: false, status: 0, code: 'no_provider' };
  const base = joinBase(envUrl, probe.provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(probe.provider.auth, probe.provider.key) };
  let url: string;
  let upstreamBody: Json;
  if (candidate.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    upstreamBody = clientApi === 'openai' ? { ...body, model: candidate.modelId } : anthropicToOpenAI(body, candidate.modelId);
    if (stream) {
      headers.accept = 'text/event-stream';
      upstreamBody = { ...upstreamBody, stream: true, stream_options: { include_usage: true } };
    }
  } else if (candidate.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    upstreamBody = clientApi === 'anthropic' ? { ...body, model: candidate.modelId } : openAIToAnthropicRequest(body, candidate.modelId);
    if (stream) headers.accept = 'text/event-stream';
  } else {
    const path = stream ? `:streamGenerateContent?alt=sse` : ':generateContent';
    url = `${base}/v1beta/models/${encodeURIComponent(candidate.modelId)}${path}`;
    upstreamBody = clientApi === 'openai' ? openAIToGemini(body) : anthropicToGemini(body);
    if (stream) headers.accept = 'text/event-stream';
  }
  let res: Response;
  try {
    res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(upstreamBody), signal: AbortSignal.timeout(CALL_TIMEOUT_MS) });
  } catch {
    return { ok: false, status: 0, code: 'unreachable' };
  }
  const text = await res.text();
  if (!res.ok) return { ok: false, ...failureOf(res.status, text) };
  const answer = stream
    ? candidate.wire === 'openai'
      ? openAIStreamAnswer(text)
      : candidate.wire === 'anthropic'
        ? anthropicStreamAnswer(text)
        : geminiStreamAnswer(text)
    : candidate.wire === 'openai'
      ? openAIJsonAnswer(parseJson(text))
      : candidate.wire === 'anthropic'
        ? anthropicJsonAnswer(parseJson(text))
        : geminiJsonAnswer(parseJson(text));
  if (answer === null) return { ok: false, status: res.status, code: 'bad_answer' };
  return { ok: true, answer };
}

function parseJson(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

/** The router's memory: how many times each provider has refused in a row. */
export interface RouterState {
  consecutiveFailures: Map<string, number>;
}

export function newRouterState(): RouterState {
  return { consecutiveFailures: new Map() };
}

/**
 * Serve one client request: walk its candidates until one answers, counting
 * refusals as the walk goes. Providers that refused twice running are
 * skipped on the first pass and tried only if nothing else answers.
 */
export async function serveRequest(
  brief: Brief,
  envUrl: string,
  probes: readonly Probe[],
  state: RouterState,
  clientApi: 'openai' | 'anthropic',
  body: Json,
  stream: boolean,
  needs: Needs,
  jsonMode: boolean,
): Promise<{ candidate: Candidate; answer: ProviderAnswer } | null> {
  const ranked = candidatesFor(brief, probes, clientApi, needs, jsonMode);
  const firstPass = ranked.filter((c) => (state.consecutiveFailures.get(c.providerId) ?? 0) < SKIP_AFTER_FAILURES);
  const secondPass = ranked.filter((c) => (state.consecutiveFailures.get(c.providerId) ?? 0) >= SKIP_AFTER_FAILURES);
  for (const candidate of [...firstPass, ...secondPass]) {
    const result = await callProvider(envUrl, probes, candidate, clientApi, body, stream);
    if (result.ok) {
      state.consecutiveFailures.set(candidate.providerId, 0);
      return { candidate, answer: result.answer };
    }
    state.consecutiveFailures.set(candidate.providerId, (state.consecutiveFailures.get(candidate.providerId) ?? 0) + 1);
  }
  return null;
}
