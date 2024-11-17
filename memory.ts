import { Ollama } from '@langchain/ollama';
import {
  BufferMemory,
  BufferWindowMemory,
  ConversationSummaryMemory,
  ENTITY_MEMORY_CONVERSATION_TEMPLATE,
  EntityMemory,
} from 'langchain/memory';
import { ConversationChain } from 'langchain/chains';
import { PromptTemplate } from '@langchain/core/prompts';

const ollama = new Ollama({
  baseUrl: 'http://localhost:11434',
  model: 'llama3.1',
});
async function start() {
  const memory = new BufferMemory();
  const chain = new ConversationChain({ llm: ollama, memory, verbose: true });
  const res = await chain.call({ input: 'It is rainging now' });
  console.log(res);
  const res1 = await chain.call({ input: 'What is the weather like?' });
  console.log(res1);
}

// start();

async function start2() {
  const memory = new BufferWindowMemory({ k: 1 });
  const chain = new ConversationChain({ llm: ollama, memory, verbose: true });
  const res = await chain.call({
    input: 'It is rainging now. I am planning to go outside.',
  });
  console.log(res);
  const res1 = await chain.call({ input: 'What is the weather like?' });
  console.log(res1);
  const res2 = await chain.call({
    input: 'What I am going to do?',
  });
  console.log(res2);
}

async function start3() {
  const memory = new ConversationSummaryMemory({
    memoryKey: 'summary',
    llm: ollama,
  });
  const prompt = PromptTemplate.fromTemplate(`
        Your are a assistant.
        {summary}
        Huname:{input}
        AI:
    `);
  const chain = new ConversationChain({
    llm: ollama,
    memory,
    prompt,
    verbose: true,
  });
  const res = await chain.call({
    input: 'It is rainging now. I am planning to go outside.',
  });
  console.log(res);
  const res1 = await chain.call({ input: 'What is the weather like?' });
  console.log(res1);
}

async function start4() {
  const memory = new EntityMemory({
    llm: ollama,
    chatHistoryKey: 'history',
    entitiesKey: 'entities',
  });
  const chain = new ConversationChain({
    llm: ollama,
    memory,
    verbose: true,
    prompt: ENTITY_MEMORY_CONVERSATION_TEMPLATE,
  });
  const res = await chain.call({
    input: 'My name is nk, I am 30 years old.',
  });
  console.log(res);
  const res2 = await chain.call({ input: 'It is raining now.' });
  console.log(res2);
}

const argv = process.argv.slice(2);
if (argv[0] === '1') {
  start();
} else if (argv[0] === '2') {
  start2();
} else if (argv[0] === '3') {
  start3();
} else {
  start4();
}

// start2();
