// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// The entry serves the OpenAI Chat Completions API and the Anthropic
// Messages API, each in JSON and in server-sent events, over the ten
// providers the trial's brief lists. For each request it picks the best
// free model that meets the request's requirements, speaks that provider's
// wire, fails over to another provider when one refuses, and returns the
// answer in the client's wire with the provider's text whole, its usage
// counts as that wire reports them, and its stop reason mapped.
//
// A trial runs it as `node /bot/main.ts` with Node 22's standard library
// alone. It reads the brief at $MAKEFUN_PROMPT_FILE, probes each provider's
// catalog route, then:
//   1. listens on 0.0.0.0 at a free port (never 8080: the env itself is
//      there on one layout);
//   2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>,
//      "catalog": <catalog>};
//   3. serves the client's requests: GET /v1/models (in the shape of the
//      auth header it got: x-api-key means Anthropic, Authorization means
//      OpenAI), POST /v1/chat/completions and POST /v1/messages, each with
//      a JSON body or with server-sent events;
//   4. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env
//      says {"done": true}. A process still running when the wall ends
//      scores zero, so it never waits for anything else.
// `node main.ts --list` prints the same catalog JSON and exits 0 before any
// server starts; the register body's catalog is that same document, byte
// for byte, built once and used for both.
//
// Every choice is a function of the brief, the requests and the providers'
// answers: no wall clock and no unseeded randomness anywhere, so a live
// pass and its replay leave the same state export byte for byte. The only
// randomness the entry ever needs is none at all; the one tie-break that
// could have used it (equal ratings) is settled by wire, then provider id,
// then model id.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { buildCatalog, freeModelIds, probeProvider, type Catalog, type Probe } from './lib/catalog.ts';
import { isRecord, readBrief, type Brief, type Json, type Needs } from './lib/brief.ts';
import { newRouterState, serveRequest } from './lib/route.ts';
import {
  answerToAnthropicJson,
  answerToAnthropicStream,
  answerToOpenAIJson,
  answerToOpenAIStream,
  anthropicError,
  anthropicModelList,
  openAIError,
  openAIModelList,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** How many times a failed register is retried before the trial gives up. */
const REGISTER_RETRIES = 5;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief: Brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: Catalog = { providers: [] };
/** The router's memory: which providers have refused, and how often in a row. */
const router = newRouterState();
/** How many client API requests have arrived: the manifest is in arrival order, one at a time. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(probes, brief);
}

/** The free models the catalog registers, as GET /v1/models lists them. */
function servedModels(): { id: string; provider: string; displayName: string }[] {
  const ids = freeModelIds(catalog);
  const out: { id: string; provider: string; displayName: string }[] = [];
  for (const id of ids) {
    let provider = '';
    let displayName = id;
    // The catalog names the provider for every registered model, including
    // one the listing stayed quiet about and the table completed.
    for (const entry of catalog.providers) {
      if (entry.models.some((m) => m.id === id)) {
        provider = entry.id;
        break;
      }
    }
    for (const probe of probes) {
      const found = probe.models.find((m) => m.id === id);
      if (found !== undefined) {
        displayName = found.displayName;
        break;
      }
    }
    out.push({ id, provider, displayName });
  }
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json, clientApi: 'openai' | 'anthropic'): Needs {
  const manifest = brief.manifest;
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.needs;
  return requirementsFromBody(body, clientApi);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json, clientApi: 'openai' | 'anthropic'): Needs {
  let vision = false;
  if (clientApi === 'openai') {
    const messages = Array.isArray(body.messages) ? body.messages : [];
    for (const message of messages) {
      if (!isRecord(message) || !Array.isArray(message.content)) continue;
      for (const block of message.content) {
        if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
      }
    }
  } else {
    const messages = Array.isArray(body.messages) ? body.messages : [];
    for (const message of messages) {
      if (!isRecord(message) || !Array.isArray(message.content)) continue;
      for (const block of message.content) {
        if (isRecord(block) && block.type === 'image') vision = true;
      }
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  const tools = Array.isArray(body.tools) && body.tools.length > 0;
  return { tools, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** Whether this client request asks for JSON mode (only the OpenAI wire can). */
function jsonModeOf(body: Json, clientApi: 'openai' | 'anthropic'): boolean {
  if (clientApi !== 'openai') return false;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return format === 'json_object' || format === 'json_schema';
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
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

function sendText(res: ServerResponse, status: number, text: string, contentType: string): void {
  res.writeHead(status, { 'content-type': contentType, 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/**
 * One client request, served by the best free model that meets it: the
 * provider's answer is rendered back in the client's wire, streamed or
 * not as it asked.
 */
async function serve(req: IncomingMessage, res: ServerResponse, clientApi: 'openai' | 'anthropic'): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    sendError(res, clientApi, 400, 'invalid_request_error', 'the request body is not one JSON object');
    return;
  }
  const stream = body.stream === true;
  const needs = requirementsFor(body, clientApi);
  const served = await serveRequest(brief, ENV_URL, probes, router, clientApi, body, stream, needs, jsonModeOf(body, clientApi));
  if (served === null) {
    sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
    return;
  }
  const { answer, candidate } = served;
  if (clientApi === 'openai') {
    if (stream) {
      sendText(res, 200, answerToOpenAIStream(answer, candidate.modelId), 'text/event-stream; charset=utf-8');
    } else {
      sendJson(res, 200, answerToOpenAIJson(answer, candidate.modelId));
    }
  } else if (stream) {
    sendText(res, 200, answerToAnthropicStream(answer, candidate.modelId), 'text/event-stream; charset=utf-8');
  } else {
    sendJson(res, 200, answerToAnthropicJson(answer, candidate.modelId));
  }
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await serve(req, res, 'openai');
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await serve(req, res, 'anthropic');
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: Catalog): Promise<void> {
  let lastError: Error | null = null;
  for (let attempt = 0; attempt < REGISTER_RETRIES; attempt += 1) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
      const reply: unknown = await res.json();
      if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
      return;
    } catch (err) {
      lastError = err as Error;
      await new Promise((resolve) => setTimeout(resolve, 200));
    }
  }
  throw lastError ?? new Error('register failed');
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
