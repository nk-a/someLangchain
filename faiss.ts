import { FaissStore } from '@langchain/community/vectorstores/faiss';
import { OllamaEmbeddings } from '@langchain/ollama';
import { TextLoader } from 'langchain/document_loaders/fs/text';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';

async function saveFaiss() {
  const loader = new TextLoader('./data/example.txt');
  const docs = await loader.load();

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 0,
  });

  const splitDoc = await splitter.splitDocuments(docs);

  const embedding = new OllamaEmbeddings({
    baseUrl: 'http://localhost:11434',
    model: 'llama3.1',
  });

  const vectorStore = await FaissStore.fromDocuments(splitDoc, embedding);
  const dir = './data/db';
  await vectorStore.save(dir);
}

saveFaiss();
