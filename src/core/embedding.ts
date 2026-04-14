/**
 * Embedding Service
 * Ported from production Ruby implementation (embedding_service.rb, 190 LOC)
 *
 * Default provider: OpenAI text-embedding-3-large at 1536 dimensions.
 * Optional provider: SiliconFlow OpenAI-compatible embeddings endpoint.
 * Retry with exponential backoff (4s base, 120s cap, 5 retries).
 * 8000 character input truncation.
 */

import OpenAI from 'openai';

const OPENAI_MODEL = 'text-embedding-3-large';
const SILICONFLOW_MODEL = 'Qwen/Qwen3-Embedding-4B';
const DEFAULT_DIMENSIONS = 1536;
const MAX_CHARS = 8000;
const MAX_RETRIES = 5;
const BASE_DELAY_MS = 4000;
const MAX_DELAY_MS = 120000;
const BATCH_SIZE = 100;

let client: OpenAI | null = null;

export type EmbeddingProvider = 'openai' | 'siliconflow';

export function getEmbeddingProvider(): EmbeddingProvider {
  const raw = (process.env.GBRAIN_EMBEDDING_PROVIDER || '').toLowerCase().trim();
  if (raw === 'siliconflow') return 'siliconflow';
  return 'openai';
}

export function hasEmbeddingCredentials(): boolean {
  const provider = getEmbeddingProvider();
  if (provider === 'siliconflow') {
    return Boolean(process.env.SILICONFLOW_API_KEY);
  }
  return Boolean(process.env.OPENAI_API_KEY);
}

function getEmbeddingModel(): string {
  const provider = getEmbeddingProvider();
  if (provider === 'siliconflow') {
    return process.env.SILICONFLOW_EMBEDDING_MODEL || SILICONFLOW_MODEL;
  }
  return process.env.OPENAI_EMBEDDING_MODEL || OPENAI_MODEL;
}

function getEmbeddingDimensions(): number {
  const raw = process.env.GBRAIN_EMBEDDING_DIMENSIONS;
  if (!raw) return DEFAULT_DIMENSIONS;
  const parsed = Number.parseInt(raw, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : DEFAULT_DIMENSIONS;
}

function getClient(): OpenAI {
  if (!client) {
    const provider = getEmbeddingProvider();
    if (provider === 'siliconflow') {
      const apiKey = process.env.SILICONFLOW_API_KEY;
      if (!apiKey) {
        throw new Error('SILICONFLOW_API_KEY is required when GBRAIN_EMBEDDING_PROVIDER=siliconflow');
      }
      client = new OpenAI({
        apiKey,
        baseURL: process.env.SILICONFLOW_BASE_URL || 'https://api.siliconflow.cn/v1',
      });
    } else {
      client = new OpenAI();
    }
  }
  return client;
}

export async function embed(text: string): Promise<Float32Array> {
  const truncated = text.slice(0, MAX_CHARS);
  const result = await embedBatch([truncated]);
  return result[0];
}

export async function embedBatch(texts: string[]): Promise<Float32Array[]> {
  const truncated = texts.map(t => t.slice(0, MAX_CHARS));
  const results: Float32Array[] = [];

  // Process in batches of BATCH_SIZE
  for (let i = 0; i < truncated.length; i += BATCH_SIZE) {
    const batch = truncated.slice(i, i + BATCH_SIZE);
    const batchResults = await embedBatchWithRetry(batch);
    results.push(...batchResults);
  }

  return results;
}

async function embedBatchWithRetry(texts: string[]): Promise<Float32Array[]> {
  for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
    try {
      const response = await getClient().embeddings.create({
        model: getEmbeddingModel(),
        input: texts,
        dimensions: getEmbeddingDimensions(),
      });

      // Sort by index to maintain order
      const sorted = response.data.sort((a, b) => a.index - b.index);
      return sorted.map(d => new Float32Array(d.embedding));
    } catch (e: unknown) {
      if (attempt === MAX_RETRIES - 1) throw e;

      // Check for rate limit with Retry-After header
      let delay = exponentialDelay(attempt);

      if (e instanceof OpenAI.APIError && e.status === 429) {
        const retryAfter = e.headers?.['retry-after'];
        if (retryAfter) {
          const parsed = parseInt(retryAfter, 10);
          if (!isNaN(parsed)) {
            delay = parsed * 1000;
          }
        }
      }

      await sleep(delay);
    }
  }

  // Should not reach here
  throw new Error('Embedding failed after all retries');
}

function exponentialDelay(attempt: number): number {
  const delay = BASE_DELAY_MS * Math.pow(2, attempt);
  return Math.min(delay, MAX_DELAY_MS);
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

export const EMBEDDING_MODEL = getEmbeddingModel();
export const EMBEDDING_DIMENSIONS = getEmbeddingDimensions();
