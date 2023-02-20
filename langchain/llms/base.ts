import { LLMCallbackManager, LLMResult, OpenAI } from "./index";
import { BaseCache, InMemoryCache } from "../cache";
import { OpenAIRequestConfig } from "./openai";

const getCallbackManager = (): LLMCallbackManager => ({
  handleStart: (..._args) => {
    // console.log(args);
  },
  handleEnd: (..._args) => {
    // console.log(args);
  },
  handleError: (..._args) => {
    // console.log(args);
  },
});

const getVerbosity = () => true;

const cache: BaseCache = new InMemoryCache();

export type SerializedLLM = {
  _type: string;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
} & Record<string, any>;

/**
 * LLM Wrapper. Provides an {@link call} (an {@link generate}) function that takes in a prompt (or prompts) and returns a string.
 */
export abstract class BaseLLM {
  /**
   * The name of the LLM class
   */
  name: string;

  cache?: boolean;

  callbackManager: LLMCallbackManager;

  /**
   * Whether to print out response text.
   */
  verbose?: boolean = false;

  constructor(callbackManager?: LLMCallbackManager, verbose?: boolean) {
    this.callbackManager = callbackManager ?? getCallbackManager();
    this.verbose = verbose ?? getVerbosity();
  }

  /**
   * Run the LLM on the given prompts and input.
   */
  abstract _generate(
    prompts: string[],
    stop?: string[],
    options?: OpenAIRequestConfig
  ): Promise<LLMResult>;

  /** @ignore */
  async _generateUncached(
    prompts: string[],
    stop?: string[],
    options?: OpenAIRequestConfig
  ): Promise<LLMResult> {
    this.callbackManager.handleStart?.(
      { name: this.name },
      prompts,
      this.verbose
    );
    let output;
    try {
      output = await this._generate(prompts, stop, options);
    } catch (err) {
      this.callbackManager.handleError?.(`${err}`, this.verbose);
      throw err;
    }

    this.callbackManager.handleEnd?.(output, this.verbose);
    return output;
  }

  /**
   * Run the LLM on the given propmts an input, handling caching.
   */
  async generate(
    prompts: string[],
    stop?: string[],
    options?: OpenAIRequestConfig
  ): Promise<LLMResult> {
    if (!Array.isArray(prompts)) {
      throw new Error("Argument 'prompts' is expected to be a string[]");
    }

    if (this.cache === true && cache === null) {
      throw new Error("Requested cache, but no cache found");
    }

    if (cache === null || this.cache === false) {
      return this._generateUncached(prompts, stop, options);
    }

    const params = this.serialize();
    params.stop = stop;

    const llmStringKey = `${Object.entries(params).sort()}`;
    const missingPromptIndices: number[] = [];
    const generations = prompts.map((prompt, index) => {
      const result = cache.lookup(prompt, llmStringKey);
      if (!result) {
        missingPromptIndices.push(index);
      }
      return result;
    });

    let llmOutput = {};
    if (missingPromptIndices.length > 0) {
      const results = await this._generateUncached(
        missingPromptIndices.map((i) => prompts[i]),
        stop,
        options
      );
      results.generations.forEach((generation, index) => {
        const promptIndex = missingPromptIndices[index];
        generations[promptIndex] = generation;
        cache.update(prompts[promptIndex], llmStringKey, generation);
      });
      llmOutput = results.llmOutput ?? {};
    }

    return { generations, llmOutput } as LLMResult;
  }

  /**
   * Convenience wrapper for {@link generate} that takes in a single string prompt and returns a single string output.
   */
  async call(prompt: string, stop?: string[], options?: OpenAIRequestConfig) {
    const { generations } = await this.generate([prompt], stop, options);
    return generations[0][0].text;
  }

  /**
   * Get the identifying parameters of the LLM.
   */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  _identifyingParams(): Record<string, any> {
    return {};
  }

  /**
   * Return the string type key uniquely identifying this class of LLM.
   */
  abstract _llmType(): string;

  /**
   * Return a json-like object representing this LLM.
   */
  serialize(): SerializedLLM {
    return {
      ...this._identifyingParams(),
      _type: this._llmType(),
    };
  }

  /**
   * Load an LLM from a json-like object describing it.
   */
  static async deserialize(data: SerializedLLM): Promise<BaseLLM> {
    const { _type, ...rest } = data;
    const Cls = {
      openai: OpenAI,
    }[_type];
    if (Cls === undefined) {
      throw new Error(`Cannot load  LLM with type ${_type}`);
    }
    return new Cls(rest);
  }

  // TODO(sean): save to disk, get_num_tokens
}

/**
 * LLM class that provides a simpler interface to subclass than {@link BaseLLM}.
 *
 * Requires only implementing a simpler {@link _call} method instead of {@link _generate}.
 *
 * @augments BaseLLM
 */
export abstract class LLM extends BaseLLM {
  /**
   * Run the LLM on the given prompt and input.
   */
  abstract _call(
    prompt: string,
    stop?: string[],
    options?: OpenAIRequestConfig
  ): Promise<string>;

  async _generate(
    prompts: string[],
    stop?: string[],
    options?: OpenAIRequestConfig
  ): Promise<LLMResult> {
    const generations = [];
    for (let i = 0; i < prompts.length; i += 1) {
      const text = await this._call(prompts[i], stop, options);
      generations.push([{ text }]);
    }
    return { generations };
  }
}
