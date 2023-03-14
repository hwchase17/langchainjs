import { test, expect } from "@jest/globals";
import { Document } from "../../document.js";
import {
  BaseLanguageModel,
  BaseLanguageModelParams,
} from "../../base_language/index.js";
import { loadQAMapReduceChain } from "../question_answering/load.js";
import { BasePromptValue, LLMResult } from "../../schema/index.js";

test("Test MapReduceDocumentsChain", async () => {
  let nrMapCalls = 0;
  let nrReduceCalls = 0;

  class FakeLLM extends BaseLanguageModel {
    constructor(params: BaseLanguageModelParams) {
      super(params);
    }

    _modelType(): string {
      return "fake";
    }

    getNumTokens(_text: string): number {
      return 0;
    }

    generatePrompt(
      promptValues: BasePromptValue[],
      _stop?: string[]
    ): Promise<LLMResult> {
      return new Promise((resolve) => {
        resolve({
          generations: promptValues.map((prompt) => {
            let completion = "";
            if (prompt.toString().startsWith("Use the following portion")) {
              nrMapCalls += 1;
              completion = "a portion of context";
            } else if (
              prompt.toString().startsWith("Given the following extracted")
            ) {
              nrReduceCalls += 1;
              completion = "a final answer";
            }
            return [
              {
                text: completion,
                score: 0,
              },
            ];
          }),
        });
      });
    }
  }

  const model = new FakeLLM({});
  const chain = loadQAMapReduceChain(model);
  const docs = [
    new Document({ pageContent: "harrison went to harvard" }),
    new Document({ pageContent: "ankush went to princeton" }),
  ];
  const res = await chain.call({
    input_documents: docs,
    question: "Where did harrison go to college",
  });
  console.log({ res });

  expect(res).toEqual({
    text: "a final answer",
  });
  expect(nrMapCalls).toBe(0); // below maxTokens
  expect(nrReduceCalls).toBe(1);
});
