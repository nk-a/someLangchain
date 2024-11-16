import { FaissStore } from '@langchain/community/vectorstores/faiss';
import { Ollama, OllamaEmbeddings } from '@langchain/ollama';
import { TextLoader } from 'langchain/document_loaders/fs/text';
import { ContextualCompressionRetriever } from 'langchain/retrievers/contextual_compression';
import { LLMChainExtractor } from 'langchain/retrievers/document_compressors/chain_extract';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { RunnableSequence } from '@langchain/core/runnables';

async function start() {
  const model = new Ollama({
    baseUrl: 'http://localhost:11434',
    model: 'llama3.1',
  });
  const embedding = new OllamaEmbeddings({
    baseUrl: 'http://localhost:11434',
    model: 'llama3.1',
  });
  //   const loader = new TextLoader('data/blackcat.txt');
  //   const docs = await loader.load();
  //   const splitter = new RecursiveCharacterTextSplitter({
  //     chunkSize: 500,
  //     chunkOverlap: 50,
  //   });
  //   const splitDoc = await splitter.splitDocuments(docs);
  //   const vectorStore = await FaissStore.fromDocuments(splitDoc, embedding);
  const dir = './data/blackcat';
  //   await vectorStore.save(dir);

  const compressor = LLMChainExtractor.fromLLM(model);
  const vectorStore = await FaissStore.load(dir, embedding);

  const vectorStoreRetriever = vectorStore.asRetriever();

  // const convertDocs = (document) => {
  //   return document.map((doc) => doc.pageContent.join('\n'));
  // };
  const contextRetriverChain = RunnableSequence.from([
    (input) => input.question,
    vectorStoreRetriever,
  ]);
  const result = await contextRetriverChain.invoke({
    question: 'Who am I in the story?',
  });
  console.log(result);
}

start();
