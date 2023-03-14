# Vector DB QA Chain

The Vector DB QA chain operates by taking a question as input, searching for relevant documents in the vector database, and passing both the documents and the question to a question answering chain, which returns a response.

To create one, you will need a vectorstore, which can be created from embeddings.

Below is an end-to-end example of doing question answering over a recent state of the union address.

```typescript
import { OpenAI } from "langchain/llms";
import { VectorDBQAChain } from "langchain/chains";
import { HNSWLib } from "langchain/vectorstores";
import { OpenAIEmbeddings } from "langchain/embeddings";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import * as fs from "fs";

/* Initialize the LLM to use to answer the question */
const model = new OpenAI({});
/* Load in the file we want to do question answering over */
const text = fs.readFileSync("state_of_the_union.txt", "utf8");
/* Split the text into chunks */
const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000 });
const docs = await textSplitter.createDocuments([text]);
/* Create the vectorstore */
const vectorStore = await HNSWLib.fromDocuments(docs, new OpenAIEmbeddings());
/* Create the chain */
const chain = VectorDBQAChain.fromLLM(model, vectorStore);
/* Ask it a question */
const res = await chain.call({
  input_documents: docs,
  query: "What did the president say about Justice Breyer?",
});
```
