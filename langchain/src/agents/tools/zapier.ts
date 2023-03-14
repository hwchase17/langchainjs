import { Tool } from "./base.js";

const zapierNLABaseDescription = `
A wrapper around Zapier NLA actions. The input to this tool is a natural language instruction, 
for example "get the latest email from my bank" or "send a slack message to the #general channel".
This tool specifically used for: `;

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type ZapierValues = Record<string, any>;

export class ZapierNLAWrapper {
  zapierNlaApiKey: string;

  zapierNlaApiBase = "https://nla.zapier.com/api/v1/";

  zapierNlaApiDynamicBase = "https://nla.zapier.com/api/v1/dynamic/";

  constructor(zapierNlaApiKey?: string) {
    const apiKey = zapierNlaApiKey ?? process.env.ZAPIER_NLA_API_KEY;
    if (!apiKey) {
      throw new Error("ZAPIER_NLA_API_KEY not set");
    }
    this.zapierNlaApiKey = apiKey;
  }

  protected _getHeaders(): Record<string, string> {
    return {
      "Content-Type": "application/json",
      Accept: "application/json",
      "x-api-key": this.zapierNlaApiKey,
    };
  }

  protected async _getActionRequest(
    actionId: string,
    instructions: string,
    params?: ZapierValues
  ): Promise<ZapierValues> {
    const data = params ?? {};
    data.instructions = instructions;
    const headers = this._getHeaders();

    // add api key to params
    const resp = await fetch(
      `${this.zapierNlaApiBase}exposed/${actionId}/execute/`,
      {
        method: "POST",
        headers,
        body: JSON.stringify(data),
      }
    );

    if (!resp.ok) {
      throw new Error(
        `Failed to execute action ${actionId} with instructions ${instructions}`
      );
    }
    return resp.json();
  }

  /**
   * Executes an action that is identified by action_id, must be exposed
   * (enabled) by the current user (associated with the set api_key). Change
   * your exposed actions here: https://nla.zapier.com/demo/start/
   * @param actionId
   * @param instructions
   * @param params
   */
  async runAction(
    actionId: string,
    instructions: string,
    params?: ZapierValues
  ): Promise<ZapierValues> {
    const resp = await this._getActionRequest(actionId, instructions, params);
    return resp.result;
  }

  /**
   * Same as run, but instead of actually executing the action, will
   * instead return a preview of params that have been guessed by the AI in
   * case you need to explicitly review before executing.
   * @param actionId
   * @param instructions
   * @param params
   */
  async previewAction(
    actionId: string,
    instructions: string,
    params?: ZapierValues
  ): Promise<ZapierValues> {
    const data = params ?? {};
    data.preview_only = true;
    const resp = await this._getActionRequest(actionId, instructions, data);
    return resp.params;
  }

  /**
   * Returns a list of all exposed (enabled) actions associated with
   * current user (associated with the set api_key). Change your exposed
   * actions here: https://nla.zapier.com/demo/start/
   */
  async listActions(): Promise<ZapierValues[]> {
    const headers = this._getHeaders();
    const resp = await fetch(`${this.zapierNlaApiDynamicBase}exposed/`, {
      method: "GET",
      headers,
    });
    if (!resp.ok) {
      throw new Error("Failed to list actions");
    }
    return (await resp.json()).results;
  }

  /**
   * Same as run, but returns a stringified version of the result.
   * @param actionId
   * @param instructions
   * @param params
   */
  async runAsString(
    actionId: string,
    instructions: string,
    params?: ZapierValues
  ): Promise<string> {
    const result = await this.runAction(actionId, instructions, params);
    return JSON.stringify(result);
  }

  /**
   * Same as preview, but returns a stringified version of the result.
   * @param actionId
   * @param instructions
   * @param params
   */
  async previewAsString(
    actionId: string,
    instructions: string,
    params?: ZapierValues
  ): Promise<string> {
    const result = await this.previewAction(actionId, instructions, params);
    return JSON.stringify(result);
  }

  /**
   * Same as list, but returns a stringified version of the result.
   */
  async listActionsAsString(): Promise<string> {
    const result = await this.listActions();
    return JSON.stringify(result);
  }
}

export class ZapierNLARunAction extends Tool {
  apiWrapper: ZapierNLAWrapper;

  actionId: string;

  params?: ZapierValues;

  name: string;

  description: string;

  constructor(
    apiWrapper: ZapierNLAWrapper,
    actionId: string,
    zapierDescription: string,
    params?: ZapierValues
  ) {
    super();
    this.apiWrapper = apiWrapper;
    this.actionId = actionId;
    this.params = params;
    this.name = zapierDescription;
    this.description = zapierNLABaseDescription + zapierDescription;
  }

  async _call(arg: string): Promise<string> {
    return this.apiWrapper.runAsString(this.actionId, arg, this.params);
  }
}
