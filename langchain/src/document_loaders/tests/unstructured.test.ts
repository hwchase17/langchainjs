import { test, expect } from "@jest/globals";
import { UnstructuredBaseDocumentLoader } from "../unstructured.js";

test.skip("Test Unstructured base loader", async () => {
  const loader = new UnstructuredBaseDocumentLoader(
    "http://127.0.0.1:8000/general/v0.0.4/general",
    "/Users/nuno/dev/langchainjs/langchain/src/document_loaders/tests/example_data/example.txt"
  );
  const docs = await loader.load();

  expect(docs.length).toBe(3);
  for (const doc of docs) {
    expect(typeof doc.pageContent).toBe("string");
  }
});
