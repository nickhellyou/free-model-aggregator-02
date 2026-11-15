// Brief parsing and model routing for the free-model aggregator.
//
// The trial brief (MAKEFUN_PROMPT_FILE) is markdown carrying two one-line
// JSON blocks: the prompt manifest (schema .../prompts/v1) and the free
// model catalog (schema .../models/v1). This module reads them, builds the
// catalog the env expects, and ranks the models that may serve a prompt.
//
// Routing rule: among the free models that meet the prompt's requirements,
// serve the highest rated one first. A model that refused earlier in this
// trial is demoted below every model that has not refused yet, but stays
// available as a last resort. The order is a pure function of the brief and
// the refusals seen so far, so a replay makes the same choices.
import { readFileSync } from 'node:fs';

export interface BriefModel {
  id: string;
  upstream: string;
  rating_milli: number;
  context_length: number;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

export interface BriefPrompt {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: number;
  requires: { tools: boolean; vision: boolean; json: boolean };
}

export interface Brief {
  models: BriefModel[];
  prompts: BriefPrompt[];
}

const PROMPTS_SCHEMA = 'free-model-aggregator-02/prompts/v1';
const MODELS_SCHEMA = 'free-model-aggregator-02/models/v1';

function isRecord(v: unknown): v is Record<string, unknown> {
  return v !== null && typeof v === 'object' && !Array.isArray(v);
}

function parseJsonBlock(text: string, schema: string): Record<string, unknown> | null {
  for (const line of text.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed.startsWith('{"schema":')) continue;
    try {
      const doc = JSON.parse(trimmed) as unknown;
      if (isRecord(doc) && doc.schema === schema) return doc;
    } catch {
      // Not a parseable block on this line; keep scanning.
    }
  }
  return null;
}

export function readBrief(): Brief | null {
  const path = process.env.MAKEFUN_PROMPT_FILE;
  if (!path) return null;
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch {
    return null;
  }
  const promptsDoc = parseJsonBlock(text, PROMPTS_SCHEMA);
  const modelsDoc = parseJsonBlock(text, MODELS_SCHEMA);
  if (promptsDoc === null || modelsDoc === null) return null;
  const prompts = Array.isArray(promptsDoc.prompts) ? (promptsDoc.prompts as BriefPrompt[]) : [];
  const models = Array.isArray(modelsDoc.models) ? (modelsDoc.models as BriefModel[]) : [];
  return { models, prompts };
}

export function freeModels(brief: Brief): BriefModel[] {
  const seen = new Set<string>();
  const out: BriefModel[] = [];
  for (const model of brief.models) {
    if (model.free !== true) continue;
    if (seen.has(model.id)) continue;
    seen.add(model.id);
    out.push(model);
  }
  return out;
}

// The catalog the env registers and --list prints, byte for byte the same
// document in both places. Exactly the free models, no more and no fewer.
export function buildCatalog(brief: Brief | null): Record<string, unknown> {
  if (brief === null) return { providers: [] };
  const models = freeModels(brief).map((m) => ({
    id: m.id,
    free: true,
    context_length: m.context_length,
  }));
  if (models.length === 0) return { providers: [] };
  return {
    providers: [
      {
        id: 'openrouter',
        base_url: 'https://openrouter.ai/api/v1',
        key_env: 'OPENROUTER_API_KEY',
        wire: 'openai',
        models,
      },
    ],
  };
}

export function meetsRequirements(
  model: BriefModel,
  requires: BriefPrompt['requires'],
): boolean {
  if (requires.tools === true && model.tools !== true) return false;
  if (requires.vision === true && model.vision !== true) return false;
  if (requires.json === true && model.json !== true) return false;
  return true;
}

// Candidates for a prompt, best first. `cooled` holds the ids of models
// that already refused in this trial: they sort below every model that has
// not refused, keeping rating order inside each group.
export function rankedCandidates(
  brief: Brief,
  requires: BriefPrompt['requires'],
  cooled: ReadonlySet<string>,
): BriefModel[] {
  const eligible = freeModels(brief).filter((m) => meetsRequirements(m, requires));
  const byRating = (a: BriefModel, b: BriefModel) => b.rating_milli - a.rating_milli;
  const fresh = eligible.filter((m) => !cooled.has(m.id)).sort(byRating);
  const stale = eligible.filter((m) => cooled.has(m.id)).sort(byRating);
  return [...fresh, ...stale];
}
