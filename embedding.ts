import { TextLoader } from 'langchain/document_loaders/fs/text';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { Ollama, OllamaEmbeddings } from '@langchain/ollama';
import { FaissStore } from '@langchain/community/vectorstores/faiss';
import { InMemoryStore } from '@langchain/core/stores';
import { CacheBackedEmbeddings } from 'langchain/embeddings/cache_backed';
import 'faiss-node';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';

async function run1() {
  const embedding = new OllamaEmbeddings({
    baseUrl: 'http://localhost:11434',
    model: 'llama3.1',
  });

  const memoryStore = new InMemoryStore();

  const cacheBackedEmbeddings = CacheBackedEmbeddings.fromBytesStore(
    embedding,
    memoryStore,
    {
      namespace: embedding.model,
    }
  );

  const textLoader = new TextLoader('./data/example.txt');

  const document = await textLoader.load();

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 100,
    chunkOverlap: 20,
  });

  const docs = await splitter.splitDocuments(document);

  let time = Date.now();
  const vectorstore = await FaissStore.fromDocuments(
    docs,
    cacheBackedEmbeddings
  );
  console.log(`Initial creation time: ${Date.now() - time}ms`);

  time = Date.now();
  const vectorstore2 = await FaissStore.fromDocuments(
    docs,
    cacheBackedEmbeddings
  );
  console.log(`Cached creation time: ${Date.now() - time}ms`);
}

async function run() {
  const embedding = new OllamaEmbeddings({
    baseUrl: 'http://localhost:11434',
    model: 'llama3.1',
  });

  const textLoader = new TextLoader('./data/example.txt');

  const document = await textLoader.load();

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 0,
  });

  const docs = await splitter.splitDocuments(document);

  const vectorStore = await MemoryVectorStore.fromDocuments(docs, embedding);

  const result = await vectorStore.similaritySearch('video', 1);
  console.log(result);
}

run();
