import jsonpointer from "jsonpointer";

import { TextLoader } from "./text.js";

export class JSONLinesLoader extends TextLoader {
  constructor(filePathOrBlob: string | Blob, public pointer: string) {
    super(filePathOrBlob);
  }

  protected async parse(raw: string): Promise<string[]> {
    const lines = raw.split("\n");
    const jsons = lines
      .map((line) => line.trim())
      .filter(Boolean)
      .map((line) => JSON.parse(line));
    const pointer = jsonpointer.compile(this.pointer);
    return jsons.map((json) => pointer.get(json));
  }
}
