/**
 * Multi-Query Expansion via Claude Haiku
 * Ported from production Ruby implementation (query_expansion_service.rb, 69 LOC)
 *
 * Skip queries < 3 words.
 * Generate 2 alternative phrasings via tool use.
 * Return original + alternatives (max 3 total).
 */

import Anthropic from '@anthropic-ai/sdk';

const MAX_QUERIES = 3;
const MIN_WORDS = 3;
const MINIMAX_ANTHROPIC_BASE_URL = 'https://api.minimaxi.com/anthropic';
const DEFAULT_ANTHROPIC_MODEL = 'claude-haiku-4-5-20251001';
const DEFAULT_MINIMAX_MODEL = 'MiniMax-M2.5-highspeed';

let anthropicClient: Anthropic | null = null;

function getClient(): Anthropic {
  if (!anthropicClient) {
    const provider = getExpansionProvider();
    if (provider === 'minimax') {
      anthropicClient = new Anthropic({
        apiKey: getMinimaxApiKey(),
        baseURL: process.env.MINIMAX_ANTHROPIC_BASE_URL || MINIMAX_ANTHROPIC_BASE_URL,
      });
    } else {
      anthropicClient = new Anthropic();
    }
  }
  return anthropicClient;
}

type ExpansionProvider = 'anthropic' | 'minimax';

export function getExpansionProvider(): ExpansionProvider {
  const raw = (process.env.GBRAIN_EXPANSION_PROVIDER || '').toLowerCase().trim();
  if (raw === 'minimax') return 'minimax';
  return 'anthropic';
}

function getExpansionModel(): string {
  const provider = getExpansionProvider();
  if (provider === 'minimax') {
    return process.env.MINIMAX_MODEL || DEFAULT_MINIMAX_MODEL;
  }
  return process.env.ANTHROPIC_MODEL || DEFAULT_ANTHROPIC_MODEL;
}

function getMinimaxApiKey(): string {
  const key = process.env.MINIMAX_API_KEY || process.env.ANTHROPIC_API_KEY;
  if (!key) {
    throw new Error('MINIMAX_API_KEY is required when GBRAIN_EXPANSION_PROVIDER=minimax');
  }
  return key;
}

export async function expandQuery(query: string): Promise<string[]> {
  // CJK text is not space-delimited — count characters instead of whitespace-separated tokens
  const hasCJK = /[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]/.test(query);
  const wordCount = hasCJK ? query.replace(/\s/g, '').length : (query.match(/\S+/g) || []).length;
  if (wordCount < MIN_WORDS) return [query];

  try {
    const alternatives = await callHaikuForExpansion(query);
    const all = [query, ...alternatives];
    // Deduplicate
    const unique = [...new Set(all.map(q => q.toLowerCase().trim()))];
    return unique.slice(0, MAX_QUERIES).map(q =>
      all.find(orig => orig.toLowerCase().trim() === q) || q,
    );
  } catch {
    return [query];
  }
}

async function callHaikuForExpansion(query: string): Promise<string[]> {
  const response = await getClient().messages.create({
    model: getExpansionModel(),
    max_tokens: 300,
    tools: [
      {
        name: 'expand_query',
        description: 'Generate alternative phrasings of a search query to improve recall',
        input_schema: {
          type: 'object' as const,
          properties: {
            alternative_queries: {
              type: 'array',
              items: { type: 'string' },
              description: '2 alternative phrasings of the original query, each approaching the topic from a different angle',
            },
          },
          required: ['alternative_queries'],
        },
      },
    ],
    tool_choice: { type: 'tool', name: 'expand_query' },
    messages: [
      {
        role: 'user',
        content: `Generate 2 alternative search queries that would find relevant results for this question. Each alternative should approach the topic from a different angle or use different terminology.

Original query: "${query}"`,
      },
    ],
  });

  // Extract tool use result
  for (const block of response.content) {
    if (block.type === 'tool_use' && block.name === 'expand_query') {
      const input = block.input as { alternative_queries?: unknown };
      const alts = input.alternative_queries;
      if (Array.isArray(alts)) {
        return alts.map(String).slice(0, 2);
      }
    }
  }

  return [];
}
