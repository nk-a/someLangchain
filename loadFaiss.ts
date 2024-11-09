import { FaissStore } from '@langchain/community/vectorstores/faiss';
import { Ollama, OllamaEmbeddings } from '@langchain/ollama';
import { MultiQueryRetriever } from 'langchain/retrievers/multi_query';
import { LLMChainExtractor } from 'langchain/retrievers/document_compressors/chain_extract';
import { ContextualCompressionRetriever } from 'langchain/retrievers/contextual_compression';

async function load() {
  const dir = './data/db';
  const model = new Ollama({
    baseUrl: 'http://localhost:11434',
    model: 'llama3.1',
  });
  const embedding = new OllamaEmbeddings({
    baseUrl: 'http://localhost:11434',
    model: 'llama3.1',
  });

  const compressor = LLMChainExtractor.fromLLM(model);
  const vectorStore = await FaissStore.load(dir, embedding);

  const retriever = new ContextualCompressionRetriever({
    baseCompressor: compressor,
    baseRetriever: vectorStore.asRetriever(),
    verbose: true,
  });

  const res = await retriever.invoke(
    'Which country does the article talk about'
  );
  console.log(res);
}

load();
