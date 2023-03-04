import hash from "object-hash";
import { Generation } from "./llms/index.js";

// Takes in an arbitrary number of strings and returns a hash of them
// that can be used as a key in a cache.
export const getKey = (...strings: string[]): string => {
  return hash(strings);
};

export abstract class BaseCache<T = Generation[]> {
  abstract lookup(key: string): T | undefined;

  abstract update(key: string, value: T): void;
}

export class InMemoryCache<T = Generation[]> extends BaseCache<T> {
  private cache: Record<string, T>;

  constructor() {
    super();
    this.cache = {};
  }

  lookup(key: string) {
    return this.cache[key];
  }

  update(key: string, value: T) {
    this.cache[key] = value;
  }
}
