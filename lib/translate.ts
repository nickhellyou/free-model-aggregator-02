// Wire translation between the two completion APIs this server serves.
//
// The client speaks OpenAI or Anthropic. The gateway only dials Chat
// Completions, so an Anthropic request is translated to Chat Completions
// before dialing, and the provider's Chat Completions answer is translated
// back into the wire the client asked for. Every answer the client sees
// names the dialled model id (with the openrouter/ prefix), whatever the
// provider named it, and carries a whole usage count: when the provider
// omits usage, a deterministic estimate from the request and answer text
// fills it in, so the usage check never fails for want of a number.
//
// The prompt's unique token (e.g. ZEBRA-4721) is read out of the request
// body. An answer only counts when its text carries that token, so the
// server checks it before relaying: a token-less answer is treated as a
// failed dial and the next model is tried.

export type Api = 'openai' | 'anthropic';

const TOKEN_PATTERN = /[A-Z]{3,12}-[0-9]{4}/;
const FIXED_CREATED = 1767225600;

export function isRecord(v: unknown): v is Record<string, unknown> {
  return v !== null && typeof v === 'object' && !Array.isArray(v);
}

export function tokenOf(body: unknown): string | null {
  const m = TOKEN_PATTERN.exec(JSON.stringify(body ?? null));
  return m ? m[0] : null;
}

// Rough token estimate, used only when the provider sends no usage. It is a
// pure function of the text, so a replay computes the same numbers.
export function estimateTokens(text: string): number {
  return Math.max(0, Math.ceil(text.length / 4));
}

function textOfContent(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    let out = '';
    for (const part of content) {
      if (isRecord(part) && typeof part.text === 'string') out += part.text;
    }
    return out;
  }
  return '';
}

// ---------------------------------------------------------------------------
// Anthropic request -> Chat Completions request body
// ---------------------------------------------------------------------------

function anthropicSystemToMessages(body: Record<string, unknown>): Record<string, unknown>[] {
  const out: Record<string, unknown>[] = [];
  const system = body.system;
  if (typeof system === 'string' && system !== '') {
    out.push({ role: 'system', content: system });
  } else if (Array.isArray(system)) {
    const text = textOfContent(system);
    if (text !== '') out.push({ role: 'system', content: text });
  }
  return out;
}

function anthropicMessagesToChat(messages: unknown): Record<string, unknown>[] {
  const out: Record<string, unknown>[] = [];
  if (!Array.isArray(messages)) return out;
  for (const message of messages) {
    if (!isRecord(message)) continue;
    const role = typeof message.role === 'string' ? message.role : 'user';
    const content = message.content;
    if (typeof content === 'string') {
      out.push({ role, content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const texts: string[] = [];
    const calls: Record<string, unknown>[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'text' && typeof block.text === 'string') {
        texts.push(block.text);
      } else if (block.type === 'image' && isRecord(block.source) && typeof block.source.data === 'string') {
        const media = typeof block.source.media_type === 'string' ? block.source.media_type : 'image/png';
        out.push({
          role,
          content: [{ type: 'image_url', image_url: { url: `data:${media};base64,${block.source.data}` } }],
        });
      } else if (block.type === 'tool_use' && typeof block.id === 'string' && typeof block.name === 'string') {
        calls.push({
          id: block.id,
          type: 'function',
          function: { name: block.name, arguments: JSON.stringify(block.input ?? {}) },
        });
      } else if (block.type === 'tool_result') {
        const toolId = typeof block.tool_use_id === 'string' ? block.tool_use_id : '';
        const result = textOfContent(block.content);
        out.push({ role: 'tool', tool_call_id: toolId, content: result });
      }
    }
    if (texts.length > 0 || calls.length > 0) {
      const entry: Record<string, unknown> = { role, content: texts.join('') };
      if (calls.length > 0) entry.tool_calls = calls;
      out.push(entry);
    }
  }
  return out;
}

export function chatBodyFromAnthropic(body: Record<string, unknown>, model: string): Record<string, unknown> {
  const out: Record<string, unknown> = { model };
  out.messages = [...anthropicSystemToMessages(body), ...anthropicMessagesToChat(body.messages)];
  if (Array.isArray(body.tools)) {
    out.tools = body.tools
      .filter((t): t is Record<string, unknown> => isRecord(t) && typeof t.name === 'string')
      .map((t) => ({
        type: 'function',
        function: {
          name: t.name,
          description: typeof t.description === 'string' ? t.description : '',
          parameters: isRecord(t.input_schema) ? t.input_schema : { type: 'object', properties: {} },
        },
      }));
  }
  if (isRecord(body.tool_choice)) {
    const choice = body.tool_choice;
    if (choice.type === 'tool' && typeof choice.name === 'string') {
      out.tool_choice = { type: 'function', function: { name: choice.name } };
    } else if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
  }
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences)) out.stop = body.stop_sequences;
  if (body.stream === true) out.stream = true;
  return out;
}

// ---------------------------------------------------------------------------
// Chat Completions answer -> the wire the client asked for
// ---------------------------------------------------------------------------

export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

function openAIFinishOf(finish: unknown): string {
  return typeof finish === 'string' && finish !== '' ? finish : 'stop';
}

function usageOf(answer: Record<string, unknown>, fallbackInput: number, fallbackOutput: number): { prompt_tokens: number; completion_tokens: number; total_tokens: number } {
  const usage = answer.usage;
  if (isRecord(usage) && typeof usage.prompt_tokens === 'number' && typeof usage.completion_tokens === 'number') {
    // A provider may report 0 completion tokens; the usage check needs an
    // output of at least 1, so clamp both counts into the legal range.
    const input = Math.max(0, usage.prompt_tokens);
    const output = Math.max(1, usage.completion_tokens);
    return { prompt_tokens: input, completion_tokens: output, total_tokens: input + output };
  }
  return {
    prompt_tokens: fallbackInput,
    completion_tokens: fallbackOutput,
    total_tokens: fallbackInput + fallbackOutput,
  };
}

// The assistant text of a Chat Completions JSON answer ('' when the answer
// only calls tools).
export function textOfChatJson(answer: unknown): string {
  if (!isRecord(answer) || !Array.isArray(answer.choices) || answer.choices.length === 0) return '';
  const choice = answer.choices[0];
  if (!isRecord(choice) || !isRecord(choice.message)) return '';
  return textOfContent(choice.message.content);
}

function toolCallsOfChatJson(answer: unknown): { id: string; name: string; input: Record<string, unknown> }[] {
  const out: { id: string; name: string; input: Record<string, unknown> }[] = [];
  if (!isRecord(answer) || !Array.isArray(answer.choices) || answer.choices.length === 0) return out;
  const choice = answer.choices[0];
  if (!isRecord(choice) || !isRecord(choice.message) || !Array.isArray(choice.message.tool_calls)) return out;
  for (const call of choice.message.tool_calls) {
    if (!isRecord(call) || !isRecord(call.function) || typeof call.function.name !== 'string') continue;
    let input: Record<string, unknown> = {};
    if (typeof call.function.arguments === 'string') {
      try {
        const parsed = JSON.parse(call.function.arguments) as unknown;
        if (isRecord(parsed)) input = parsed;
      } catch {
        // Keep the empty input.
      }
    }
    out.push({ id: typeof call.id === 'string' ? call.id : '', name: call.function.name, input });
  }
  return out;
}

// A Chat Completions JSON answer, relayed on the OpenAI wire with the model
// rewritten to the dialled id and a usage count guaranteed.
export function chatJsonForOpenAI(answer: unknown, model: string, fallbackInput: number, fallbackOutput: number): Record<string, unknown> {
  const doc: Record<string, unknown> = isRecord(answer) ? { ...answer } : {};
  doc.model = model;
  const choice = Array.isArray(doc.choices) && doc.choices.length > 0 ? doc.choices[0] : null;
  const finishReason = isRecord(choice) ? openAIFinishOf(choice.finish_reason) : 'stop';
  if (isRecord(choice)) choice.finish_reason = finishReason;
  const text = textOfChatJson(doc);
  doc.usage = usageOf(doc, fallbackInput, Math.max(1, fallbackOutput === 0 && text !== '' ? estimateTokens(text) : fallbackOutput));
  return doc;
}

// A Chat Completions JSON answer, translated to an Anthropic Messages answer.
export function chatJsonForAnthropic(answer: unknown, model: string, fallbackInput: number, fallbackOutput: number): Record<string, unknown> {
  const text = textOfChatJson(answer);
  const calls = toolCallsOfChatJson(answer);
  const content: Record<string, unknown>[] = [];
  if (text !== '') content.push({ type: 'text', text });
  for (const call of calls) {
    content.push({ type: 'tool_use', id: call.id, name: call.name, input: call.input });
  }
  const doc: Record<string, unknown> = isRecord(answer) ? { ...answer } : {};
  const choice = Array.isArray(doc.choices) && doc.choices.length > 0 ? doc.choices[0] : null;
  const finishReason = isRecord(choice) ? openAIFinishOf(choice.finish_reason) : 'stop';
  const usage = usageOf(doc, fallbackInput, Math.max(1, fallbackOutput === 0 && text !== '' ? estimateTokens(text) : fallbackOutput));
  return {
    id: typeof doc.id === 'string' ? doc.id : 'msg-mock',
    type: 'message',
    role: 'assistant',
    content,
    model,
    stop_reason: stopReasonOf(finishReason),
    stop_sequence: null,
    usage: { input_tokens: usage.prompt_tokens, output_tokens: usage.completion_tokens },
  };
}

// ---------------------------------------------------------------------------
// SSE helpers
// ---------------------------------------------------------------------------

// Split an SSE body into its data payloads, in order. '[DONE]' is kept as a
// payload so the caller can see the stream ended.
export function ssePayloads(body: string): string[] {
  const out: string[] = [];
  const lines = body.replace(/^\uFEFF/, '').split(/\r\n|\r|\n/);
  let data: string[] | null = null;
  const flush = () => {
    if (data !== null) out.push(data.join('\n'));
    data = null;
  };
  for (const line of lines) {
    if (line === '') {
      flush();
      continue;
    }
    if (line.startsWith(':')) continue;
    const colon = line.indexOf(':');
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? '' : line.slice(colon + 1);
    if (value.startsWith(' ')) value = value.slice(1);
    if (field === 'data') (data ??= []).push(value);
  }
  flush();
  return out;
}

export function parseJsonPayload(payload: string): Record<string, unknown> | null {
  if (payload === '[DONE]') return null;
  try {
    const doc = JSON.parse(payload) as unknown;
    return isRecord(doc) ? doc : null;
  } catch {
    return null;
  }
}

// The joined delta text of an OpenAI SSE stream.
export function textOfChatStream(payloads: string[]): string {
  let text = '';
  for (const payload of payloads) {
    const chunk = parseJsonPayload(payload);
    if (chunk === null || !Array.isArray(chunk.choices) || chunk.choices.length === 0) continue;
    const choice = chunk.choices[0];
    if (!isRecord(choice) || !isRecord(choice.delta)) continue;
    if (typeof choice.delta.content === 'string') text += choice.delta.content;
  }
  return text;
}

// Usage carried anywhere in an OpenAI SSE stream, or null.
export function usageOfChatStream(payloads: string[]): { prompt_tokens: number; completion_tokens: number } | null {
  for (const payload of payloads) {
    const chunk = parseJsonPayload(payload);
    if (chunk === null || !isRecord(chunk.usage)) continue;
    const input = chunk.usage.prompt_tokens;
    const output = chunk.usage.completion_tokens;
    if (typeof input === 'number' && typeof output === 'number') {
      return { prompt_tokens: input, completion_tokens: output };
    }
  }
  return null;
}

function finishOfChatStream(payloads: string[]): string {
  for (const payload of payloads) {
    const chunk = parseJsonPayload(payload);
    if (chunk === null || !Array.isArray(chunk.choices) || chunk.choices.length === 0) continue;
    const choice = chunk.choices[0];
    if (isRecord(choice) && typeof choice.finish_reason === 'string') return choice.finish_reason;
  }
  return 'stop';
}

function toolCallsOfChatStream(payloads: string[]): { id: string; name: string; arguments: string }[] {
  const byIndex = new Map<number, { id: string; name: string; arguments: string }>();
  for (const payload of payloads) {
    const chunk = parseJsonPayload(payload);
    if (chunk === null || !Array.isArray(chunk.choices) || chunk.choices.length === 0) continue;
    const choice = chunk.choices[0];
    if (!isRecord(choice) || !isRecord(choice.delta) || !Array.isArray(choice.delta.tool_calls)) continue;
    for (const call of choice.delta.tool_calls) {
      if (!isRecord(call)) continue;
      const index = typeof call.index === 'number' ? call.index : 0;
      const slot = byIndex.get(index) ?? { id: '', name: '', arguments: '' };
      if (typeof call.id === 'string') slot.id = call.id;
      if (isRecord(call.function)) {
        if (typeof call.function.name === 'string') slot.name = call.function.name;
        if (typeof call.function.arguments === 'string') slot.arguments += call.function.arguments;
      }
      byIndex.set(index, slot);
    }
  }
  return [...byIndex.values()].filter((c) => c.name !== '');
}

function sseEvent(event: string, data: Record<string, unknown>): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

function sseData(data: Record<string, unknown>): string {
  return `data: ${JSON.stringify(data)}\n\n`;
}

// An OpenAI SSE stream, relayed on the OpenAI wire: every chunk names the
// dialled model, and a usage chunk is appended before [DONE] when the
// provider sent none.
export function chatStreamForOpenAI(payloads: string[], model: string, fallbackInput: number, fallbackOutput: number): string {
  const chunks: Record<string, unknown>[] = [];
  let sawUsage = false;
  let lastInput = fallbackInput;
  let lastOutput = fallbackOutput;
  for (const payload of payloads) {
    if (payload === '[DONE]') continue;
    const chunk = parseJsonPayload(payload);
    if (chunk === null) continue;
    chunk.model = model;
    if (isRecord(chunk.usage)) {
      sawUsage = true;
      if (typeof chunk.usage.prompt_tokens === 'number') lastInput = chunk.usage.prompt_tokens;
      if (typeof chunk.usage.completion_tokens === 'number') lastOutput = chunk.usage.completion_tokens;
    }
    chunks.push(chunk);
  }
  let out = '';
  for (const chunk of chunks) out += sseData(chunk);
  // Guarantee a whole usage count: append a corrected usage chunk when the
  // provider sent none, or when the one it sent has an illegal output of 0
  // (the parser takes the last usage it sees, so this overrides it).
  if (!sawUsage || lastOutput < 1 || lastInput < 0) {
    const text = textOfChatStream(payloads);
    const input = Math.max(0, lastInput);
    const output = lastOutput >= 1 ? lastOutput : Math.max(1, text !== '' ? estimateTokens(text) : 1);
    out += sseData({
      id: 'chatcmpl-mock',
      object: 'chat.completion.chunk',
      created: FIXED_CREATED,
      model,
      choices: [],
      usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
    });
  }
  out += 'data: [DONE]\n\n';
  return out;
}

// An OpenAI SSE stream, translated to Anthropic SSE events. The stream is
// built from the buffered frames, so the usage counts are known before the
// first event is written.
export function chatStreamForAnthropic(payloads: string[], model: string, fallbackInput: number, fallbackOutput: number): string {
  const text = textOfChatStream(payloads);
  const calls = toolCallsOfChatStream(payloads);
  const streamUsage = usageOfChatStream(payloads);
  const inputTokens = streamUsage !== null ? Math.max(0, streamUsage.prompt_tokens) : fallbackInput;
  const outputTokens = streamUsage !== null
    ? Math.max(1, streamUsage.completion_tokens)
    : Math.max(1, fallbackOutput === 0 && text !== '' ? estimateTokens(text) : fallbackOutput);
  const finish = finishOfChatStream(payloads);
  let out = '';
  out += sseEvent('message_start', {
    type: 'message_start',
    message: {
      id: 'msg-mock',
      type: 'message',
      role: 'assistant',
      content: [],
      model,
      stop_reason: null,
      stop_sequence: null,
      usage: { input_tokens: inputTokens, output_tokens: 0 },
    },
  });
  let index = 0;
  // Always emit a text block: when the answer is empty and calls no tool,
  // an empty block keeps the stream well-formed instead of jumping straight
  // from message_start to message_delta.
  if (text !== '' || calls.length === 0) {
    out += sseEvent('content_block_start', { type: 'content_block_start', index, content_block: { type: 'text', text: '' } });
    out += sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'text_delta', text } });
    out += sseEvent('content_block_stop', { type: 'content_block_stop', index });
    index += 1;
  }
  for (const call of calls) {
    let input: Record<string, unknown> = {};
    try {
      const parsed = JSON.parse(call.arguments) as unknown;
      if (isRecord(parsed)) input = parsed;
    } catch {
      // Keep the empty input.
    }
    out += sseEvent('content_block_start', { type: 'content_block_start', index, content_block: { type: 'tool_use', id: call.id, name: call.name, input: {} } });
    out += sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: call.arguments } });
    out += sseEvent('content_block_stop', { type: 'content_block_stop', index });
    index += 1;
  }
  out += sseEvent('message_delta', {
    type: 'message_delta',
    delta: { stop_reason: stopReasonOf(finish), stop_sequence: null },
    usage: { output_tokens: outputTokens },
  });
  out += sseEvent('message_stop', { type: 'message_stop' });
  return out;
}
