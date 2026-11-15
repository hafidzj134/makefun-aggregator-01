// Wire translation between the client's API and each provider's API.
//
// The client speaks OpenAI Chat Completions or Anthropic Messages; a provider
// speaks one of those two or Gemini's generateContent. The providers answer
// with an echo of what they parsed (system prompt, user text, tool names,
// max_tokens, stop sequences, image hash, JSON mode), and the client compares
// that echo with what it sent, so every converter here is written against the
// parsers' exact semantics:
//
//   - system text is the system messages' (or field's) text blocks joined
//     with "\n"; an absent system stays absent, an empty one stays empty;
//   - user text is the LAST user message's text blocks joined with "\n";
//   - an image is carried as the same bytes on every wire (data URL, base64
//     block or inline data), so its hash survives the crossing;
//   - tool names keep their order; max_tokens and stop sequences are copied,
//     never invented or dropped;
//   - JSON mode exists on the OpenAI wire (response_format) and the Gemini
//     wire (responseMimeType) but not on the Anthropic wire, whose echo
//     always reports it absent — the router keeps such requests off it.
//
// Responses travel back with their text whole, their usage counts re-labelled
// into the client's wire, and their stop reason mapped inside its family
// (end_turn <-> stop, max_tokens <-> length, tool_use <-> tool_calls,
// refusal <-> content_filter). Streaming answers are re-framed event by
// event in the client's own event order. Nothing here reads a clock or a
// random source: the same request and the same provider answer always give
// the same bytes, which the replay check depends on.

import { isRecord, type Json } from './providers.ts';

export type ClientApi = 'openai' | 'anthropic';

/** One free model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
}

/** Fixed listing timestamps: a `created` read from a clock would differ between a trial and its replay. */
const FIXED_CREATED = 1767225600;
const FIXED_CREATED_AT = '2026-01-01T00:00:00Z';

export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_CREATED, owned_by: m.provider })),
  };
}

export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.id, created_at: FIXED_CREATED_AT })),
    has_more: false,
  };
}

export function openAIErrorBody(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

export function anthropicErrorBody(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

// ---------------------------------------------------------------------------
// Shared reading helpers (the parsers' semantics, mirrored)
// ---------------------------------------------------------------------------

/** A content value's text: the string itself, or its text blocks joined with "\n". */
export function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  }
  return parts.join('\n');
}

/** A stop field as a list: a lone string is one stop, an array keeps its order. */
export function stopList(value: unknown): string[] {
  if (typeof value === 'string') return [value];
  if (!Array.isArray(value)) return [];
  return value.filter((v): v is string => typeof v === 'string');
}

/** The output cap a client body carries, under either OpenAI spelling. */
function clientMaxTokens(body: Json): number | null {
  if (typeof body.max_tokens === 'number' && Number.isFinite(body.max_tokens)) return body.max_tokens;
  if (typeof body.max_completion_tokens === 'number' && Number.isFinite(body.max_completion_tokens)) {
    return body.max_completion_tokens;
  }
  return null;
}

/** Whether the client asked for JSON mode (an OpenAI-wire notion). */
export function clientJsonMode(body: Json): boolean {
  if (!isRecord(body.response_format)) return false;
  return body.response_format.type === 'json_object' || body.response_format.type === 'json_schema';
}

/** The system text of an OpenAI-shaped body: its system messages joined in order. */
function openAISystemText(body: Json): string {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const parts: string[] = [];
  for (const message of messages) {
    if (!isRecord(message) || message.role !== 'system') continue;
    const text = blocksText(message.content);
    if (text !== '') parts.push(text);
  }
  return parts.join('\n');
}

interface ImageRef {
  mediaType: string;
  data: string | null;
  url: string | null;
}

/** An OpenAI image_url part as an image reference. */
function imageFromOpenAI(part: Json): ImageRef | null {
  if (!isRecord(part.image_url) || typeof part.image_url.url !== 'string') return null;
  const url = part.image_url.url;
  const data = /^data:([^;,]+);base64,(.*)$/s.exec(url);
  if (data !== null) return { mediaType: data[1], data: data[2], url: null };
  return { mediaType: 'image/png', data: null, url };
}

/** An Anthropic image block as an image reference. */
function imageFromAnthropic(block: Json): ImageRef | null {
  if (!isRecord(block.source)) return null;
  const source = block.source;
  if (source.type === 'base64' && typeof source.data === 'string') {
    return { mediaType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data, url: null };
  }
  if (source.type === 'url' && typeof source.url === 'string') return { mediaType: 'image/png', data: null, url: source.url };
  return null;
}

function imageAsOpenAIPart(image: ImageRef): Json {
  const url = image.data !== null ? `data:${image.mediaType};base64,${image.data}` : (image.url ?? '');
  return { type: 'image_url', image_url: { url } };
}

function imageAsAnthropicBlock(image: ImageRef): Json {
  if (image.data !== null) {
    return { type: 'image', source: { type: 'base64', media_type: image.mediaType, data: image.data } };
  }
  return { type: 'image', source: { type: 'url', url: image.url ?? '' } };
}

function imageAsGeminiPart(image: ImageRef): Json {
  if (image.data !== null) return { inlineData: { mimeType: image.mediaType, data: image.data } };
  return { fileData: { fileUri: image.url ?? '' } };
}

/** A tool call's arguments as an object; a non-JSON string becomes an empty input. */
function argumentsObject(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: fall through to the empty object.
    }
  }
  return {};
}

/** A tool_result's content as plain text. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = blocksText(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

// ---------------------------------------------------------------------------
// Request converters: client wire -> provider wire
// ---------------------------------------------------------------------------

interface ToolDef {
  name: string;
  description: string;
  parameters: Json;
}

function toolsFromOpenAI(body: Json): ToolDef[] {
  if (!Array.isArray(body.tools)) return [];
  const out: ToolDef[] = [];
  for (const tool of body.tools) {
    if (!isRecord(tool) || !isRecord(tool.function) || typeof tool.function.name !== 'string') continue;
    out.push({
      name: tool.function.name,
      description: typeof tool.function.description === 'string' ? tool.function.description : '',
      parameters: isRecord(tool.function.parameters) ? tool.function.parameters : { type: 'object', properties: {} },
    });
  }
  return out;
}

function toolsFromAnthropic(body: Json): ToolDef[] {
  if (!Array.isArray(body.tools)) return [];
  const out: ToolDef[] = [];
  for (const tool of body.tools) {
    if (!isRecord(tool) || typeof tool.name !== 'string') continue;
    out.push({
      name: tool.name,
      description: typeof tool.description === 'string' ? tool.description : '',
      parameters: isRecord(tool.input_schema) ? tool.input_schema : { type: 'object', properties: {} },
    });
  }
  return out;
}

/** An OpenAI Chat Completions body for `model`, from either client wire. */
export function toOpenAIRequest(body: Json, api: ClientApi, model: string): Json {
  if (api === 'openai') {
    const out: Json = { ...body, model };
    if (body.stream === true) out.stream_options = { include_usage: true };
    return out;
  }
  const messages: Json[] = [];
  const system = blocksText(body.system);
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
      const text = blocksText(content);
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
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      if (block.type === 'text' && typeof block.text === 'string') {
        parts.push({ type: 'text', text: block.text });
        continue;
      }
      if (block.type === 'image') {
        const image = imageFromAnthropic(block);
        if (image !== null) parts.push(imageAsOpenAIPart(image));
      }
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }
  const out: Json = { model, messages };
  const maxTokens = clientMaxTokens(body);
  if (maxTokens !== null) out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = stopList(body.stop_sequences);
  if (stops.length > 0) out.stop = stops;
  const tools = toolsFromAnthropic(body);
  if (tools.length > 0) {
    out.tools = tools.map((t) => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.parameters } }));
  }
  if (isRecord(body.tool_choice)) {
    const choice = body.tool_choice;
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'none') out.tool_choice = 'none';
    else if (choice.type === 'tool' && typeof choice.name === 'string') {
      out.tool_choice = { type: 'function', function: { name: choice.name } };
    }
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** An Anthropic Messages body for `model`, from either client wire. */
export function toAnthropicRequest(body: Json, api: ClientApi, model: string): Json {
  if (api === 'anthropic') return { ...body, model };
  const messages: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw) || raw.role === 'system') continue;
    const content = raw.content;
    if (raw.role === 'assistant') {
      if (typeof content === 'string') {
        messages.push({ role: 'assistant', content });
        continue;
      }
      if (!Array.isArray(content)) continue;
      const blocks: Json[] = [];
      const text = blocksText(content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsObject(call.function.arguments),
        });
      }
      messages.push({ role: 'assistant', content: blocks.length > 0 ? blocks : '' });
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: toolResultText(content) }],
      });
      continue;
    }
    if (typeof content === 'string') {
      messages.push({ role: 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const blocks: Json[] = [];
    for (const part of content) {
      if (!isRecord(part)) continue;
      if (part.type === 'text' && typeof part.text === 'string') blocks.push({ type: 'text', text: part.text });
      else if (part.type === 'image_url') {
        const image = imageFromOpenAI(part);
        if (image !== null) blocks.push(imageAsAnthropicBlock(image));
      }
    }
    if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
  }
  const out: Json = { model, messages };
  const system = openAISystemText(body);
  if (system !== '') out.system = system;
  const maxTokens = clientMaxTokens(body);
  if (maxTokens !== null) out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = stopList(body.stop);
  if (stops.length > 0) out.stop_sequences = stops;
  const tools = toolsFromOpenAI(body);
  if (tools.length > 0) {
    out.tools = tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.parameters }));
  }
  const choice = body.tool_choice;
  if (typeof choice === 'string') {
    if (choice === 'auto') out.tool_choice = { type: 'auto' };
    else if (choice === 'required') out.tool_choice = { type: 'any' };
    else if (choice === 'none') out.tool_choice = { type: 'none' };
  } else if (isRecord(choice) && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** A Gemini generateContent body for `model`, from either client wire. */
export function toGeminiRequest(body: Json, api: ClientApi): Json {
  const contents: Json[] = [];
  let system = '';
  if (api === 'openai') {
    system = openAISystemText(body);
    const source = Array.isArray(body.messages) ? body.messages : [];
    for (const raw of source) {
      if (!isRecord(raw) || raw.role === 'system') continue;
      const content = raw.content;
      if (raw.role === 'assistant') {
        const parts: Json[] = [];
        const text = blocksText(content);
        if (text !== '') parts.push({ text });
        const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
        for (const call of calls) {
          if (!isRecord(call) || !isRecord(call.function)) continue;
          parts.push({
            functionCall: {
              name: typeof call.function.name === 'string' ? call.function.name : '',
              args: argumentsObject(call.function.arguments),
            },
          });
        }
        if (parts.length > 0) contents.push({ role: 'model', parts });
        continue;
      }
      if (raw.role === 'tool') {
        contents.push({ role: 'user', parts: [{ functionResponse: { name: '', response: { result: toolResultText(content) } } }] });
        continue;
      }
      if (typeof content === 'string') {
        contents.push({ role: 'user', parts: [{ text: content }] });
        continue;
      }
      if (!Array.isArray(content)) continue;
      const parts: Json[] = [];
      for (const part of content) {
        if (!isRecord(part)) continue;
        if (part.type === 'text' && typeof part.text === 'string') parts.push({ text: part.text });
        else if (part.type === 'image_url') {
          const image = imageFromOpenAI(part);
          if (image !== null) parts.push(imageAsGeminiPart(image));
        }
      }
      if (parts.length > 0) contents.push({ role: 'user', parts });
    }
  } else {
    system = blocksText(body.system);
    const source = Array.isArray(body.messages) ? body.messages : [];
    for (const raw of source) {
      if (!isRecord(raw)) continue;
      const content = raw.content;
      if (raw.role === 'assistant') {
        const parts: Json[] = [];
        if (typeof content === 'string') {
          if (content !== '') parts.push({ text: content });
        } else if (Array.isArray(content)) {
          for (const block of content) {
            if (!isRecord(block)) continue;
            if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
            else if (block.type === 'tool_use') {
              parts.push({
                functionCall: { name: typeof block.name === 'string' ? block.name : '', args: isRecord(block.input) ? block.input : {} },
              });
            }
          }
        }
        if (parts.length > 0) contents.push({ role: 'model', parts });
        continue;
      }
      if (typeof content === 'string') {
        contents.push({ role: 'user', parts: [{ text: content }] });
        continue;
      }
      if (!Array.isArray(content)) continue;
      const parts: Json[] = [];
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
        else if (block.type === 'image') {
          const image = imageFromAnthropic(block);
          if (image !== null) parts.push(imageAsGeminiPart(image));
        } else if (block.type === 'tool_result') {
          parts.push({ functionResponse: { name: '', response: { result: toolResultText(block.content) } } });
        }
      }
      if (parts.length > 0) contents.push({ role: 'user', parts });
    }
  }
  const out: Json = { contents };
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const tools = api === 'openai' ? toolsFromOpenAI(body) : toolsFromAnthropic(body);
  if (tools.length > 0) {
    out.tools = [{ functionDeclarations: tools.map((t) => ({ name: t.name, description: t.description, parameters: t.parameters })) }];
  }
  const config: Json = {};
  const maxTokens = clientMaxTokens(body);
  if (maxTokens !== null) config.maxOutputTokens = maxTokens;
  const stops = api === 'openai' ? stopList(body.stop) : stopList(body.stop_sequences);
  if (stops.length > 0) config.stopSequences = stops;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (api === 'openai' && clientJsonMode(body)) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

// ---------------------------------------------------------------------------
// Response converters: provider wire -> client wire (JSON bodies)
// ---------------------------------------------------------------------------

/** A stop reason in Anthropic's spelling, from any wire's. */
export function toAnthropicStop(reason: unknown): string {
  if (reason === 'length' || reason === 'MAX_TOKENS' || reason === 'max_tokens') return 'max_tokens';
  if (reason === 'tool_calls' || reason === 'function_call' || reason === 'tool_use') return 'tool_use';
  if (reason === 'content_filter' || reason === 'SAFETY' || reason === 'refusal') return 'refusal';
  return 'end_turn';
}

/** A finish reason in OpenAI's spelling, from any wire's. */
export function toOpenAIStop(reason: unknown): string {
  if (reason === 'max_tokens' || reason === 'MAX_TOKENS' || reason === 'length') return 'length';
  if (reason === 'tool_use' || reason === 'tool_calls' || reason === 'function_call') return 'tool_calls';
  if (reason === 'refusal' || reason === 'SAFETY' || reason === 'content_filter') return 'content_filter';
  return 'stop';
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

interface ProviderAnswer {
  text: string;
  toolCalls: { id: string; name: string; argumentsJson: string }[];
  stop: string | null;
  inputTokens: number;
  outputTokens: number;
  id: string;
}

/** An OpenAI provider's JSON answer, read once. */
function readOpenAIAnswer(body: Json): ProviderAnswer {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const text = typeof message.content === 'string' ? message.content : blocksText(message.content);
  const toolCalls: ProviderAnswer['toolCalls'] = [];
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    toolCalls.push({
      id: typeof call.id === 'string' ? call.id : '',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      argumentsJson: typeof call.function.arguments === 'string' ? call.function.arguments : '{}',
    });
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text,
    toolCalls,
    stop: typeof choice.finish_reason === 'string' ? choice.finish_reason : null,
    inputTokens: numberOrZero(usage.prompt_tokens),
    outputTokens: numberOrZero(usage.completion_tokens),
    id: typeof body.id === 'string' ? body.id : 'aggregate',
  };
}

/** An Anthropic provider's JSON answer, read once. */
function readAnthropicAnswer(body: Json): ProviderAnswer {
  const content = Array.isArray(body.content) ? body.content : [];
  let text = '';
  const toolCalls: ProviderAnswer['toolCalls'] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') text += block.text;
    else if (block.type === 'tool_use') {
      toolCalls.push({
        id: typeof block.id === 'string' ? block.id : '',
        name: typeof block.name === 'string' ? block.name : '',
        argumentsJson: JSON.stringify(isRecord(block.input) ? block.input : {}),
      });
    }
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text,
    toolCalls,
    stop: typeof body.stop_reason === 'string' ? body.stop_reason : null,
    inputTokens: numberOrZero(usage.input_tokens),
    outputTokens: numberOrZero(usage.output_tokens),
    id: typeof body.id === 'string' ? body.id : 'aggregate',
  };
}

/** A Gemini provider's JSON answer, read once. */
function readGeminiAnswer(body: Json): ProviderAnswer {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  let text = '';
  const toolCalls: ProviderAnswer['toolCalls'] = [];
  for (const part of parts) {
    if (!isRecord(part)) continue;
    if (typeof part.text === 'string') text += part.text;
    else if (isRecord(part.functionCall)) {
      toolCalls.push({
        id: '',
        name: typeof part.functionCall.name === 'string' ? part.functionCall.name : '',
        argumentsJson: JSON.stringify(isRecord(part.functionCall.args) ? part.functionCall.args : {}),
      });
    }
  }
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text,
    toolCalls,
    stop: typeof candidate.finishReason === 'string' ? candidate.finishReason : null,
    inputTokens: numberOrZero(usage.promptTokenCount),
    outputTokens: numberOrZero(usage.candidatesTokenCount),
    id: 'gemini',
  };
}

/** The answer as an OpenAI completion for `model`. */
function answerAsOpenAI(answer: ProviderAnswer, model: string): Json {
  const message: Json = { role: 'assistant', content: answer.text };
  if (answer.toolCalls.length > 0) {
    message.tool_calls = answer.toolCalls.map((call, i) => ({
      id: call.id !== '' ? call.id : `call_${i}`,
      type: 'function',
      function: { name: call.name, arguments: call.argumentsJson },
    }));
  }
  return {
    id: `chatcmpl-${answer.id}`,
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: toOpenAIStop(answer.stop), logprobs: null }],
    usage: {
      prompt_tokens: answer.inputTokens,
      completion_tokens: answer.outputTokens,
      total_tokens: answer.inputTokens + answer.outputTokens,
    },
  };
}

/** The answer as an Anthropic message naming `model`. */
function answerAsAnthropic(answer: ProviderAnswer, model: string): Json {
  const content: Json[] = [];
  if (answer.text !== '') content.push({ type: 'text', text: answer.text });
  for (const call of answer.toolCalls) {
    content.push({
      type: 'tool_use',
      id: call.id !== '' ? call.id : 'toolu_aggregate',
      name: call.name,
      input: argumentsObject(call.argumentsJson),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  return {
    id: answer.id.startsWith('msg_') ? answer.id : `msg_${answer.id}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: toAnthropicStop(answer.stop),
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
  };
}

/** A provider's JSON answer re-expressed in the client's wire. */
export function answerToClient(providerWire: 'openai' | 'anthropic' | 'gemini', body: Json, api: ClientApi, model: string): Json {
  const answer = providerWire === 'openai' ? readOpenAIAnswer(body) : providerWire === 'anthropic' ? readAnthropicAnswer(body) : readGeminiAnswer(body);
  return api === 'openai' ? answerAsOpenAI(answer, model) : answerAsAnthropic(answer, model);
}

// ---------------------------------------------------------------------------
// Streaming: SSE frame reading and cross-wire re-framing
// ---------------------------------------------------------------------------

export interface SseFrame {
  event: string | null;
  data: string;
}

/** One SSE stream as frames, in arrival order. */
export async function* readSse(stream: ReadableStream<Uint8Array>): AsyncGenerator<SseFrame> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  const framesOf = (text: string): SseFrame[] => {
    const frames: SseFrame[] = [];
    for (const raw of text.split('\n\n')) {
      if (raw.trim() === '') continue;
      let event: string | null = null;
      const data: string[] = [];
      for (const line of raw.split('\n')) {
        if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
        else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
      }
      if (data.length > 0) frames.push({ event, data: data.join('\n') });
    }
    return frames;
  };
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const head = buffer.slice(0, at + 2);
        buffer = buffer.slice(at + 2);
        for (const frame of framesOf(head)) yield frame;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const frame of framesOf(buffer)) yield frame;
  } finally {
    reader.releaseLock();
  }
}

/** One server-sent event in the client's stream: a named event line, a data line, a blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${typeof data === 'string' ? data : JSON.stringify(data)}\n\n`;
}

function parseFrameJson(frame: SseFrame): Json | null {
  try {
    const value: unknown = JSON.parse(frame.data);
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** An OpenAI provider's SSE stream re-framed as Anthropic events for `model`. */
export async function* openAIStreamAsAnthropic(stream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let started = false;
  let finished = false;
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;

  const start = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: inputTokens, output_tokens: 0 },
        },
      }),
    ];
  };
  const closeBlocks = (): string[] => {
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    }
    toolIndexes.clear();
    return out;
  };
  const finish = (stop: string): string[] => {
    if (finished) return [];
    finished = true;
    return [
      ...closeBlocks(),
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stop, stop_sequence: null }, usage: { output_tokens: outputTokens } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };

  for await (const frame of readSse(stream)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') {
      for (const event of [...start(), ...finish(pendingStop ?? 'end_turn')]) yield event;
      continue;
    }
    const chunk = parseFrameJson(frame);
    if (chunk === null) continue;
    if (isRecord(chunk.usage)) {
      inputTokens = numberOrZero(chunk.usage.prompt_tokens);
      outputTokens = numberOrZero(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    for (const event of start()) yield event;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string' && delta.content !== '') {
      if (textIndex === -1) {
        textIndex = nextIndex;
        nextIndex += 1;
        yield sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } });
      }
      yield sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } });
    }
    const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
    for (const call of calls) {
      if (!isRecord(call)) continue;
      const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
      let index = toolIndexes.get(upstreamIndex);
      if (index === undefined) {
        if (textIndex !== -1) {
          yield sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex });
          textIndex = -1;
        }
        index = nextIndex;
        nextIndex += 1;
        toolIndexes.set(upstreamIndex, index);
        const fn = isRecord(call.function) ? call.function : {};
        yield sseEvent('content_block_start', {
          type: 'content_block_start',
          index,
          content_block: {
            type: 'tool_use',
            id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
            name: typeof fn.name === 'string' ? fn.name : '',
            input: {},
          },
        });
      }
      const fn = isRecord(call.function) ? call.function : {};
      if (typeof fn.arguments === 'string' && fn.arguments !== '') {
        yield sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } });
      }
    }
    if (choice.finish_reason !== null && choice.finish_reason !== undefined) pendingStop = toAnthropicStop(choice.finish_reason);
  }
  if (!finished) {
    for (const event of start()) yield event;
    for (const event of finish(pendingStop ?? 'end_turn')) yield event;
  }
}

/** An OpenAI chunk envelope for the streaming converters below. */
function openAIChunk(model: string, delta: Json, finishReason: string | null): string {
  return sseEvent(null, {
    id: 'chatcmpl-stream',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finishReason }],
  });
}

function openAIUsageChunk(model: string, inputTokens: number, outputTokens: number): string {
  return sseEvent(null, {
    id: 'chatcmpl-stream',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [],
    usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens },
  });
}

/** An Anthropic provider's SSE stream re-framed as OpenAI chunks for `model`. */
export async function* anthropicStreamAsOpenAI(stream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let roleSent = false;
  let inputTokens = 0;
  let outputTokens = 0;
  let stop: string | null = null;
  const toolIndexByBlock = new Map<number, number>();
  let nextToolIndex = 0;

  const role = (): string[] => {
    if (roleSent) return [];
    roleSent = true;
    return [openAIChunk(model, { role: 'assistant', content: '' }, null)];
  };

  for await (const frame of readSse(stream)) {
    const data = parseFrameJson(frame);
    if (data === null) continue;
    if (frame.event === 'message_start') {
      if (isRecord(data.message) && isRecord(data.message.usage)) inputTokens = numberOrZero(data.message.usage.input_tokens);
      for (const event of role()) yield event;
    } else if (frame.event === 'content_block_delta') {
      for (const event of role()) yield event;
      const delta = isRecord(data.delta) ? data.delta : {};
      const blockIndex = typeof data.index === 'number' ? data.index : 0;
      if (delta.type === 'text_delta' && typeof delta.text === 'string' && delta.text !== '') {
        yield openAIChunk(model, { content: delta.text }, null);
      } else if (delta.type === 'input_json_delta' && typeof delta.partial_json === 'string' && delta.partial_json !== '') {
        let toolIndex = toolIndexByBlock.get(blockIndex);
        if (toolIndex === undefined) {
          toolIndex = nextToolIndex;
          nextToolIndex += 1;
          toolIndexByBlock.set(blockIndex, toolIndex);
        }
        yield openAIChunk(model, { tool_calls: [{ index: toolIndex, function: { arguments: delta.partial_json } }] }, null);
      }
    } else if (frame.event === 'content_block_start') {
      for (const event of role()) yield event;
      const block = isRecord(data.content_block) ? data.content_block : {};
      const blockIndex = typeof data.index === 'number' ? data.index : 0;
      if (block.type === 'tool_use' && !toolIndexByBlock.has(blockIndex)) {
        const toolIndex = nextToolIndex;
        nextToolIndex += 1;
        toolIndexByBlock.set(blockIndex, toolIndex);
        yield openAIChunk(
          model,
          {
            tool_calls: [
              {
                index: toolIndex,
                id: typeof block.id === 'string' ? block.id : `call_${toolIndex}`,
                type: 'function',
                function: { name: typeof block.name === 'string' ? block.name : '', arguments: '' },
              },
            ],
          },
          null,
        );
      }
    } else if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stop = data.delta.stop_reason;
      if (isRecord(data.usage)) outputTokens = numberOrZero(data.usage.output_tokens);
    }
  }
  for (const event of role()) yield event;
  yield openAIChunk(model, {}, toOpenAIStop(stop));
  yield openAIUsageChunk(model, inputTokens, outputTokens);
  yield 'data: [DONE]\n\n';
}

/** A Gemini provider's SSE stream re-framed as OpenAI chunks for `model`. */
export async function* geminiStreamAsOpenAI(stream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let roleSent = false;
  let inputTokens = 0;
  let outputTokens = 0;
  let stop: string | null = null;
  for await (const frame of readSse(stream)) {
    const data = parseFrameJson(frame);
    if (data === null) continue;
    if (!roleSent) {
      roleSent = true;
      yield openAIChunk(model, { role: 'assistant', content: '' }, null);
    }
    if (isRecord(data.usageMetadata)) {
      inputTokens = numberOrZero(data.usageMetadata.promptTokenCount);
      outputTokens = numberOrZero(data.usageMetadata.candidatesTokenCount);
    }
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate === null) continue;
    if (typeof candidate.finishReason === 'string') stop = candidate.finishReason;
    const content = isRecord(candidate.content) ? candidate.content : {};
    const parts = Array.isArray(content.parts) ? content.parts : [];
    for (const part of parts) {
      if (isRecord(part) && typeof part.text === 'string' && part.text !== '') {
        yield openAIChunk(model, { content: part.text }, null);
      }
    }
  }
  if (!roleSent) yield openAIChunk(model, { role: 'assistant', content: '' }, null);
  yield openAIChunk(model, {}, toOpenAIStop(stop));
  yield openAIUsageChunk(model, inputTokens, outputTokens);
  yield 'data: [DONE]\n\n';
}

/** A Gemini provider's SSE stream re-framed as Anthropic events for `model`. */
export async function* geminiStreamAsAnthropic(stream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let started = false;
  let blockOpen = false;
  let inputTokens = 0;
  let outputTokens = 0;
  let stop: string | null = null;
  const start = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: inputTokens, output_tokens: 0 },
        },
      }),
      sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
    ];
  };
  for await (const frame of readSse(stream)) {
    const data = parseFrameJson(frame);
    if (data === null) continue;
    for (const event of start()) {
      yield event;
      blockOpen = true;
    }
    if (isRecord(data.usageMetadata)) {
      inputTokens = numberOrZero(data.usageMetadata.promptTokenCount);
      outputTokens = numberOrZero(data.usageMetadata.candidatesTokenCount);
    }
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate === null) continue;
    if (typeof candidate.finishReason === 'string') stop = candidate.finishReason;
    const content = isRecord(candidate.content) ? candidate.content : {};
    const parts = Array.isArray(content.parts) ? content.parts : [];
    for (const part of parts) {
      if (isRecord(part) && typeof part.text === 'string' && part.text !== '') {
        yield sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: part.text } });
      }
    }
  }
  if (!started) for (const event of start()) yield event;
  if (blockOpen) yield sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 });
  yield sseEvent('message_delta', {
    type: 'message_delta',
    delta: { stop_reason: toAnthropicStop(stop), stop_sequence: null },
    usage: { output_tokens: outputTokens },
  });
  yield sseEvent('message_stop', { type: 'message_stop' });
}
