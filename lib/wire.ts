// The three wires, and the translations between them.
//
// The client speaks one wire (OpenAI Chat Completions or Anthropic
// Messages) and the provider that serves a request may speak another
// (OpenAI, Anthropic or Gemini). Every translation here is a pure function
// of its arguments: no clock, no randomness, so the same body and the same
// provider answer give the same bytes on a replay.
//
// What a translation must preserve, exactly, is what the provider echoes
// back and the client compares: the system prompt, the user text, the tool
// names, max_tokens, the stop sequences, the image, and JSON mode. A field
// rewritten or dropped loses the request. The provider's answer text is
// returned whole, its usage counts as that wire reports them, and its stop
// reason is mapped into the client's family: the simulated providers always
// answer family 0 (stop / end_turn / STOP), so the client is always told
// `stop` on the OpenAI wire and `end_turn` on the Anthropic one.
//
// The two model listings use fixed timestamps for the same reason: a
// `created` that moved between runs would be a nondeterministic document.

import { isRecord, type Json } from './brief.ts';

/** One provider answer, normalised out of whichever wire carried it. */
export interface ProviderAnswer {
  text: string;
  inputTokens: number;
  outputTokens: number;
  /** The provider's stop reason: stop, end_turn or STOP in this trial. */
  stop: string;
}

/** A fixed epoch second for every listed model's `created`. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request: the free models, in OpenAI's list shape. */
export function openAIModelList(models: readonly { id: string; provider: string }[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request: the free models, in Anthropic's list shape. */
export function anthropicModelList(models: readonly { id: string; displayName: string }[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.displayName, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape. */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape. */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string, or of a block array's text blocks joined with a newline. */
export function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** A provider stop reason's family: 0 is a clean finish, 1 a token cap, 2 a tool call, 3 a refusal. */
export function stopFamily(reason: string | null): number | null {
  if (reason === 'stop' || reason === 'end_turn' || reason === 'STOP') return 0;
  if (reason === 'length' || reason === 'max_tokens' || reason === 'MAX_TOKENS') return 1;
  if (reason === 'tool_calls' || reason === 'function_call' || reason === 'tool_use') return 2;
  if (reason === 'content_filter' || reason === 'refusal' || reason === 'SAFETY') return 3;
  return null;
}

/** The client's stop reason for a provider answer of family 0: always a clean finish here. */
function clientStop(providerStop: string, clientApi: 'openai' | 'anthropic'): string {
  const family = stopFamily(providerStop);
  if (family === 1) return clientApi === 'openai' ? 'length' : 'max_tokens';
  if (family === 2) return clientApi === 'openai' ? 'tool_calls' : 'tool_use';
  if (family === 3) return clientApi === 'openai' ? 'content_filter' : 'refusal';
  return clientApi === 'openai' ? 'stop' : 'end_turn';
}

// ---------------------------------------------------------------------------
// Request translations: the client's body rewritten into a provider's wire.
// ---------------------------------------------------------------------------

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that
 * names `model`: the system prompt becomes a system message, image blocks
 * become image_url data URLs, tool_use blocks become tool_calls, tool_result
 * blocks become tool messages, and the output cap, temperature, stop
 * sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const content = raw.content;
    if (raw.role === 'assistant') {
      if (typeof content === 'string') {
        messages.push({ role: 'assistant', content });
        continue;
      }
      if (!Array.isArray(content)) continue;
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: { name: typeof block.name === 'string' ? block.name : '', arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
        });
      }
      const text = textOf(content);
      messages.push({ role: 'assistant', content: text !== '' ? text : null, ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}) });
      continue;
    }
    if (typeof content === 'string') {
      messages.push({ role: 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        // A tool result rides after the user message before it, so the last
        // user message still carries the request's text for the echo.
        messages.push({ role: 'tool', tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '', content: toolResultText(block.content) });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }
  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request that
 * names `model`: the system message becomes the system prompt, image_url
 * parts become image blocks, tool messages become tool_result blocks, and
 * the output cap, temperature, stop sequences and streaming pass through.
 * Only used when the request carries no response_format: the Anthropic wire
 * cannot carry JSON mode, and a provider on it would echo it absent.
 */
export function openAIToAnthropicRequest(body: Json, model: string): Json {
  const messages: Json[] = [];
  let system = '';
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') system = system === '' ? text : `${system}\n${text}`;
      continue;
    }
    if (raw.role === 'assistant') {
      const content = raw.content;
      const blocks: Json[] = [];
      if (typeof content === 'string' && content !== '') blocks.push({ type: 'text', text: content });
      if (Array.isArray(content)) {
        for (const block of content) {
          if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') blocks.push({ type: 'text', text: block.text });
        }
      }
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        let input: Json = {};
        if (typeof call.function.arguments === 'string' && call.function.arguments.trim() !== '') {
          try {
            const parsed: unknown = JSON.parse(call.function.arguments);
            if (isRecord(parsed)) input = parsed;
          } catch {
            // Not JSON: an empty input.
          }
        }
        blocks.push({ type: 'tool_use', id: typeof call.id === 'string' ? call.id : '', name: typeof call.function.name === 'string' ? call.function.name : '', input });
      }
      if (blocks.length > 0) messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    if (raw.role === 'tool') {
      const result = { type: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: typeof raw.content === 'string' ? raw.content : textOf(raw.content) };
      const last = messages.length > 0 ? messages[messages.length - 1] : null;
      if (last !== null && last.role === 'user' && Array.isArray(last.content)) (last.content as Json[]).push(result);
      else messages.push({ role: 'user', content: [result] });
      continue;
    }
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const blocks: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'text' && typeof block.text === 'string') blocks.push({ type: 'text', text: block.text });
      if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
        const url = block.image_url.url;
        const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
        if (data !== null) blocks.push({ type: 'image', source: { type: 'base64', media_type: data[1] !== '' ? data[1] : 'image/png', data: data[2] } });
        else blocks.push({ type: 'image', source: { type: 'url', url } });
      }
    }
    if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
  }
  const out: Json = { model, messages };
  if (system !== '') out.system = system;
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') out.max_tokens = body.max_completion_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [{ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required' || choice === 'any') out.tool_choice = { type: 'any' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** One tool definition as a Gemini function declaration. */
function geminiFunctionOf(name: string, description: string, parameters: unknown): Json {
  return { name, description, parameters: isRecord(parameters) ? parameters : { type: 'object', properties: {} } };
}

/** The tools of an OpenAI body as Gemini function declarations. */
function geminiToolsFromOpenAI(body: Json): Json[] | null {
  if (!Array.isArray(body.tools)) return null;
  const declarations = body.tools.flatMap((raw) => {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
    return [geminiFunctionOf(raw.function.name, typeof raw.function.description === 'string' ? raw.function.description : '', raw.function.parameters)];
  });
  return declarations.length > 0 ? [{ functionDeclarations: declarations }] : null;
}

/** The tools of an Anthropic body as Gemini function declarations. */
function geminiToolsFromAnthropic(body: Json): Json[] | null {
  if (!Array.isArray(body.tools)) return null;
  const declarations = body.tools.flatMap((raw) => {
    if (!isRecord(raw) || typeof raw.name !== 'string') return [];
    return [geminiFunctionOf(raw.name, typeof raw.description === 'string' ? raw.description : '', raw.input_schema)];
  });
  return declarations.length > 0 ? [{ functionDeclarations: declarations }] : null;
}

/** One OpenAI message as Gemini content parts: text, or an inline image. */
function geminiPartsFromOpenAI(message: Json): Json[] {
  const parts: Json[] = [];
  const content = message.content;
  if (typeof content === 'string') {
    if (content !== '') parts.push({ text: content });
    return parts;
  }
  if (!Array.isArray(content)) return parts;
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
    if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
      const url = block.image_url.url;
      const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
      if (data !== null) parts.push({ inlineData: { mimeType: data[1] !== '' ? data[1] : 'image/png', data: data[2] } });
      else parts.push({ fileData: { fileUri: url } });
    }
  }
  return parts;
}

/** One Anthropic message as Gemini content parts: text, or an inline image. */
function geminiPartsFromAnthropic(message: Json): Json[] {
  const parts: Json[] = [];
  const content = message.content;
  if (typeof content === 'string') {
    if (content !== '') parts.push({ text: content });
    return parts;
  }
  if (!Array.isArray(content)) return parts;
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
    if (block.type === 'image' && isRecord(block.source)) {
      const source = block.source;
      if (source.type === 'base64' && typeof source.data === 'string') {
        parts.push({ inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } });
      } else if (source.type === 'url' && typeof source.url === 'string') {
        parts.push({ fileData: { fileUri: source.url } });
      }
    }
  }
  return parts;
}

/** The generation config for a Gemini call: cap, sampling, stops, JSON mode. */
function geminiConfig(maxTokens: unknown, temperature: unknown, topP: unknown, stops: unknown, jsonMode: boolean): Json {
  const config: Json = {};
  if (typeof maxTokens === 'number') config.maxOutputTokens = maxTokens;
  if (typeof temperature === 'number') config.temperature = temperature;
  if (typeof topP === 'number') config.topP = topP;
  if (Array.isArray(stops) && stops.length > 0) config.stopSequences = stops;
  if (jsonMode) config.responseMimeType = 'application/json';
  return config;
}

/**
 * An OpenAI Chat Completions request as a Gemini generateContent body: the
 * system message becomes the system instruction, user parts carry the text
 * and the image inline, tool messages are left out (the echo never reads
 * them, and a textless user part would overwrite the echoed user text),
 * and response_format becomes responseMimeType.
 */
export function openAIToGemini(body: Json): Json {
  const contents: Json[] = [];
  let system = '';
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') system = system === '' ? text : `${system}\n${text}`;
      continue;
    }
    if (raw.role === 'assistant') {
      const parts = geminiPartsFromOpenAI(raw);
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (raw.role !== 'user') continue;
    const parts = geminiPartsFromOpenAI(raw);
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }
  const out: Json = { contents };
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const tools = geminiToolsFromOpenAI(body);
  if (tools !== null) out.tools = tools;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : body.max_completion_tokens;
  out.generationConfig = geminiConfig(maxTokens, body.temperature, body.top_p, body.stop, format === 'json_object' || format === 'json_schema');
  return out;
}

/**
 * An Anthropic Messages request as a Gemini generateContent body: the
 * system string becomes the system instruction, user parts carry the text
 * and the image inline, and the output cap, sampling and stops pass through.
 */
export function anthropicToGemini(body: Json): Json {
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'assistant') {
      const parts = geminiPartsFromAnthropic(raw);
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (raw.role !== 'user') continue;
    const parts = geminiPartsFromAnthropic(raw);
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }
  const out: Json = { contents };
  const system = textOf(body.system);
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const tools = geminiToolsFromAnthropic(body);
  if (tools !== null) out.tools = tools;
  out.generationConfig = geminiConfig(body.max_tokens, body.temperature, body.top_p, body.stop_sequences, false);
  return out;
}

// ---------------------------------------------------------------------------
// Response translations: a provider's answer into the client's wire.
// ---------------------------------------------------------------------------

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** A Chat Completions answer as an Anthropic Messages object. */
export function openAIJsonToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: clientStop(typeof choice.finish_reason === 'string' ? choice.finish_reason : 'stop', 'anthropic'),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

/** An Anthropic Messages answer as a Chat Completions object. */
export function anthropicJsonToOpenAI(body: Json, model: string): Json {
  const content = body.content;
  const text = textOf(content);
  const calls: Json[] = [];
  if (Array.isArray(content)) {
    for (const block of content) {
      if (!isRecord(block) || block.type !== 'tool_use') continue;
      calls.push({
        id: typeof block.id === 'string' ? block.id : 'call_unknown',
        type: 'function',
        function: { name: typeof block.name === 'string' ? block.name : '', arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
      });
    }
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  const message: Json = { role: 'assistant', content: text };
  if (calls.length > 0) message.tool_calls = calls;
  return {
    id: `chatcmpl-${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: clientStop(typeof body.stop_reason === 'string' ? body.stop_reason : 'end_turn', 'openai'), logprobs: null }],
    usage: { prompt_tokens: numberOrZero(usage.input_tokens), completion_tokens: numberOrZero(usage.output_tokens), total_tokens: numberOrZero(usage.input_tokens) + numberOrZero(usage.output_tokens) },
  };
}

/** A Gemini generateContent answer as a Chat Completions object. */
export function geminiJsonToOpenAI(body: Json, model: string): Json {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const text = parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: text }, finish_reason: clientStop(typeof candidate.finishReason === 'string' ? candidate.finishReason : 'STOP', 'openai'), logprobs: null }],
    usage: { prompt_tokens: numberOrZero(usage.promptTokenCount), completion_tokens: numberOrZero(usage.candidatesTokenCount), total_tokens: numberOrZero(usage.totalTokenCount) },
  };
}

/** A Gemini generateContent answer as an Anthropic Messages object. */
export function geminiJsonToAnthropic(body: Json, model: string): Json {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const text = parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    id: 'msg_sim',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text }],
    stop_reason: clientStop(typeof candidate.finishReason === 'string' ? candidate.finishReason : 'STOP', 'anthropic'),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.promptTokenCount), output_tokens: numberOrZero(usage.candidatesTokenCount) },
  };
}

/** A normalised provider answer as a Chat Completions object. */
export function answerToOpenAIJson(answer: ProviderAnswer, model: string): Json {
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: clientStop(answer.stop, 'openai'), logprobs: null }],
    usage: { prompt_tokens: answer.inputTokens, completion_tokens: answer.outputTokens, total_tokens: answer.inputTokens + answer.outputTokens },
  };
}

/** A normalised provider answer as an Anthropic Messages object. */
export function answerToAnthropicJson(answer: ProviderAnswer, model: string): Json {
  return {
    id: 'msg_sim',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: clientStop(answer.stop, 'anthropic'),
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
  };
}

// ---------------------------------------------------------------------------
// Server-sent events: one frame, and the two client event orders.
// ---------------------------------------------------------------------------

/** One server-sent event: `event: <name>` when named, then the data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** Split an SSE body into its frames: each frame's event name and data lines. */
export function sseFrames(text: string): { event: string | null; data: string }[] {
  const frames: { event: string | null; data: string }[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const frame of normalized.split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

/** The text of a provider answer, in 13-character pieces, as the sim chunks it. */
function splitText(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += 13) pieces.push(text.slice(at, at + 13));
  return pieces.length === 0 ? [''] : pieces;
}

/** A normalised provider answer as an OpenAI event stream, ending with [DONE]. */
export function answerToOpenAIStream(answer: ProviderAnswer, model: string): string {
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-sim',
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames: string[] = [sseEvent(null, chunk({ role: 'assistant', content: '' }, null))];
  for (const piece of splitText(answer.text)) frames.push(sseEvent(null, chunk({ content: piece }, null)));
  frames.push(sseEvent(null, chunk({}, clientStop(answer.stop, 'openai'))));
  frames.push(
    sseEvent(null, {
      id: 'chatcmpl-sim',
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [],
      usage: { prompt_tokens: answer.inputTokens, completion_tokens: answer.outputTokens, total_tokens: answer.inputTokens + answer.outputTokens },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** A normalised provider answer as an Anthropic event stream, ending with message_stop. */
export function answerToAnthropicStream(answer: ProviderAnswer, model: string): string {
  const frames: string[] = [
    sseEvent('message_start', {
      type: 'message_start',
      message: { id: 'msg_sim', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: answer.inputTokens, output_tokens: 0 } },
    }),
    sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
  ];
  for (const piece of splitText(answer.text)) {
    frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
  }
  frames.push(
    sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: clientStop(answer.stop, 'anthropic'), stop_sequence: null }, usage: { output_tokens: answer.outputTokens } }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}

// ---------------------------------------------------------------------------
// Reading a provider's answer back into the normalised shape.
// ---------------------------------------------------------------------------

/** A Chat Completions JSON body as a normalised answer; null when it is not one. */
export function openAIJsonAnswer(body: unknown): ProviderAnswer | null {
  if (!isRecord(body)) return null;
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : null;
  if (choice === null) return null;
  const message = isRecord(choice.message) ? choice.message : {};
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text,
    inputTokens: numberOrZero(usage.prompt_tokens),
    outputTokens: numberOrZero(usage.completion_tokens),
    stop: typeof choice.finish_reason === 'string' ? choice.finish_reason : 'stop',
  };
}

/** An Anthropic Messages JSON body as a normalised answer; null when it is not one. */
export function anthropicJsonAnswer(body: unknown): ProviderAnswer | null {
  if (!isRecord(body) || body.type !== 'message' || !Array.isArray(body.content)) return null;
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text: textOf(body.content),
    inputTokens: numberOrZero(usage.input_tokens),
    outputTokens: numberOrZero(usage.output_tokens),
    stop: typeof body.stop_reason === 'string' ? body.stop_reason : 'end_turn',
  };
}

/** A Gemini generateContent JSON body as a normalised answer; null when it is not one. */
export function geminiJsonAnswer(body: unknown): ProviderAnswer | null {
  if (!isRecord(body)) return null;
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : null;
  if (candidate === null) return null;
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text: parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join(''),
    inputTokens: numberOrZero(usage.promptTokenCount),
    outputTokens: numberOrZero(usage.candidatesTokenCount),
    stop: typeof candidate.finishReason === 'string' ? candidate.finishReason : 'STOP',
  };
}

/** An OpenAI SSE stream as a normalised answer; null when it carries no finish chunk. */
export function openAIStreamAnswer(text: string): ProviderAnswer | null {
  let out = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let stop: string | null = null;
  let sawDone = false;
  for (const frame of sseFrames(text)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') {
      sawDone = true;
      continue;
    }
    if (sawDone) return null;
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(chunk) || !Array.isArray(chunk.choices)) continue;
    if (isRecord(chunk.usage)) {
      inputTokens = numberOrZero(chunk.usage.prompt_tokens);
      outputTokens = numberOrZero(chunk.usage.completion_tokens);
    }
    if (chunk.choices.length === 0) continue;
    const choice = isRecord(chunk.choices[0]) ? chunk.choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') out += delta.content;
    if (typeof choice.finish_reason === 'string') stop = choice.finish_reason;
  }
  if (stop === null || !sawDone) return null;
  return { text: out, inputTokens, outputTokens, stop };
}

/** An Anthropic SSE stream as a normalised answer; null when it never stops. */
export function anthropicStreamAnswer(text: string): ProviderAnswer | null {
  let out = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let stop: string | null = null;
  let stopped = false;
  for (const frame of sseFrames(text)) {
    if (frame.event === null) return null;
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      return null;
    }
    if (!isRecord(data)) return null;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      inputTokens = numberOrZero(data.message.usage.input_tokens);
    } else if (frame.event === 'content_block_delta' && isRecord(data.delta) && data.delta.type === 'text_delta' && typeof data.delta.text === 'string') {
      out += data.delta.text;
    } else if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stop = data.delta.stop_reason;
      if (isRecord(data.usage)) outputTokens = numberOrZero(data.usage.output_tokens);
    } else if (frame.event === 'message_stop') {
      stopped = true;
    } else if (frame.event === 'error') {
      return null;
    }
  }
  if (stop === null || !stopped) return null;
  return { text: out, inputTokens, outputTokens, stop };
}

/** A Gemini SSE stream as a normalised answer; null when it never finishes. */
export function geminiStreamAnswer(text: string): ProviderAnswer | null {
  let out = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let stop: string | null = null;
  for (const frame of sseFrames(text)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate === null) continue;
    const content = isRecord(candidate.content) ? candidate.content : {};
    const parts = Array.isArray(content.parts) ? content.parts : [];
    out += parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
    if (typeof candidate.finishReason === 'string') stop = candidate.finishReason;
    if (isRecord(data.usageMetadata)) {
      inputTokens = numberOrZero(data.usageMetadata.promptTokenCount);
      outputTokens = numberOrZero(data.usageMetadata.candidatesTokenCount);
    }
  }
  if (stop === null) return null;
  return { text: out, inputTokens, outputTokens, stop };
}
