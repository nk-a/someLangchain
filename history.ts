import { ChatMessageHistory } from 'langchain/stores/message/in_memory';
import { HumanMessage, AIMessage } from '@langchain/core/messages';
import { Ollama } from '@langchain/ollama';
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from '@langchain/core/prompts';
import {
  RunnableSequence,
  RunnableWithMessageHistory,
} from '@langchain/core/runnables';
import { StringOutputParser } from '@langchain/core/output_parsers';

async function start1() {
  // 1.
  //   const history = new ChatMessageHistory();
  //   await history.addMessage(new HumanMessage('What is LangChain?'));
  //   await history.addMessage(new AIMessage('I am listening'));
  //   const messages = await history.getMessages();
  //   console.log(messages);

  const ollama = new Ollama({
    baseUrl: 'http://localhost:11434',
    model: 'llama3.1',
  });
  const history = new ChatMessageHistory();
  //   await history.addMessage(new HumanMessage('I am a cat'));
  const prompt = ChatPromptTemplate.fromMessages([
    'system',
    'You are a helpful assistant that helps users.',
    new MessagesPlaceholder('history_message'),
    ['human', '${input}'],
  ]);
  const chain = prompt.pipe(ollama);
  const chainWithHistory = new RunnableWithMessageHistory({
    runnable: chain,
    getMessageHistory: (sessionId) => history,
    inputMessagesKey: 'input',
    historyMessagesKey: 'history_message',
  });
  const res1 = await chainWithHistory.invoke(
    {
      input: 'It is raining, tell me something about rain',
    },
    {
      configurable: { sessionId: 'test' },
    }
  );
  console.log(res1);
  console.log('-----------');
  const res2 = await chainWithHistory.invoke(
    {
      input: 'What is the weather like?',
    },
    {
      configurable: { sessionId: 'test' },
    }
  );
  console.log(res2);
}

async function start() {
  const ollama = new Ollama({
    baseUrl: 'http://localhost:11434',
    model: 'llama3.1',
  });
  // summary history
  const prompt = ChatPromptTemplate.fromTemplate(`
    Progressively summarize the lines of conversation provided, adding onto the previous summary returning a new summary
    
    Current summary:
    {summary}
    
    New lines of conversation:
    {new_lines}
    
    New summary:
    `);

  const chain = RunnableSequence.from([
    prompt,
    ollama,
    new StringOutputParser(),
  ]);
  const summary = await chain.invoke({
    summary: '',
    new_lines: 'It is cloudy outside',
  });
  const summary2 = await chain.invoke({
    summary: summary,
    new_lines: 'But it is raining now',
  });
  console.log(summary);
  console.log(summary2);
}

start();
