import { ChatOllama } from '@langchain/ollama';
import { z } from 'zod';
import { zodToJsonSchema } from 'zod-to-json-schema';
import { tool } from '@langchain/core/tools';
import { HumanMessage } from '@langchain/core/messages';

const calculatorSchema = z.object({
  operation: z
    .enum(['add', 'subtract', 'multiply', 'divide'])
    .describe('The type of operation to execute.'),
  number1: z.number().describe('The first number to operate on.'),
  number2: z.number().describe('The second number to operate on.'),
});

const calculatorTool = tool(
  async ({ operation, number1, number2 }) => {
    // Functions must return strings
    if (operation === 'add') {
      return `${number1 + number2}`;
    } else if (operation === 'subtract') {
      return `${number1 - number2}`;
    } else if (operation === 'multiply') {
      return `${number1 * number2}`;
    } else if (operation === 'divide') {
      return `${number1 / number2}`;
    } else {
      throw new Error('Invalid operation.');
    }
  },
  {
    name: 'calculator',
    description: 'Can perform mathematical operations.',
    schema: calculatorSchema,
  }
);

async function start() {
  const model = new ChatOllama({
    baseUrl: 'http://localhost:11434',
    model: 'llama3.1',
  });

  const modelWithTools = model.bindTools([calculatorTool]);
  const messages = [new HumanMessage('What is 2 + 2? and 3 - 1?')];
  const aiMessage = await modelWithTools.invoke(messages);
  console.log(aiMessage);
  messages.push(aiMessage);
  for (const toolCall of aiMessage.tool_calls) {
    const toolMessage = await calculatorTool.invoke(toolCall);
    messages.push(toolMessage);
  }
  console.log(messages);
  const res = await modelWithTools.invoke(messages);
  console.log(res);
}

start();
