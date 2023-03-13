import { test, expect } from "@jest/globals";

import { LangChainTracer } from "../tracers.js";
import { OpenAI } from "../../llms/index.js";
import { Calculator, SerpAPI } from "../../agents/tools/index.js";
import { initializeAgentExecutor } from "../../agents/index.js";

test("Test LangChain tracer", async () => {
  const tracer = new LangChainTracer();
  expect(tracer.alwaysVerbose).toBe(true);

  const chainRunId = Symbol("");
  const toolRunId = Symbol("");
  const llmRunId = Symbol("");
  await tracer.handleChainStart({ name: "test" }, { foo: "bar" }, chainRunId);
  await tracer.handleToolStart({ name: "test" }, "test", toolRunId);
  await tracer.handleLLMStart({ name: "test" }, ["test"], llmRunId);
  await tracer.handleLLMEnd({ generations: [[]] }, llmRunId);
  await tracer.handleToolEnd("output", toolRunId);
  await tracer.handleLLMStart({ name: "test2" }, ["test"], llmRunId);
  await tracer.handleLLMEnd({ generations: [[]] }, llmRunId);
  await tracer.handleChainEnd({ foo: "bar" }, chainRunId);

  const llmRunId2 = Symbol("");
  await tracer.handleLLMStart({ name: "test" }, ["test"], llmRunId2);
  await tracer.handleLLMEnd({ generations: [[]] }, llmRunId2);
});

test("Test Traced Agent with concurrency", async () => {
  process.env.LANGCHAIN_HANDLER = "langchain";
  const model = new OpenAI({ temperature: 0 });
  const tools = [new SerpAPI(), new Calculator()];

  const executor = await initializeAgentExecutor(
    tools,
    model,
    "zero-shot-react-description",
    true
  );

  const input = `Who is Olivia Wilde's boyfriend? What is his current age raised to the 0.23 power?`;

  console.log(`Executing with input "${input}"...`);

  const [resultA, resultB, resultC] = await Promise.all([
    executor.call({ input }),
    executor.call({ input }),
    executor.call({ input }),
  ]);

  console.log(`Got output ${resultA.output}`);
  console.log(`Got output ${resultB.output}`);
  console.log(`Got output ${resultC.output}`);
});
