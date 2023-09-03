// apps/simple/chatEngine.ts
// https://github.com/run-llama/LlamaIndexTS/blob/main/apps/simple/chatEngine.ts


// openAI api key setting
// https://chat.openai.com/share/c18c18a0-dc36-49f3-bd53-f1ce30852d5d


import { stdin as input, stdout as output } from "node:process";
// readline/promises is still experimental so not in @types/node yet
// @ts-ignore
import readline from "node:readline/promises";

import dotenv from 'dotenv';
dotenv.config();
const apiKey = process.env.OPENAI_API_KEY;



import {
  ContextChatEngine,
  Document,
  serviceContextFromDefaults,
  VectorStoreIndex,
} from "llamaindex";

// import createIndex from 'llamaindex';

import essay from "./doc/essay";


// const index = createIndex({
//   openaiApiKey: "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",
// });

async function main() {
  const document = new Document({ text: essay });
  const serviceContext = serviceContextFromDefaults({ chunkSize: 512 });
  const index = await VectorStoreIndex.fromDocuments([document], {
    serviceContext,
  });
  const retriever = index.asRetriever();
  retriever.similarityTopK = 5;
  const chatEngine = new ContextChatEngine({ retriever });
  const rl = readline.createInterface({ input, output });

  while (true) {
    const query = await rl.question("Query: ");
    const response = await chatEngine.chat(query);
    console.log(response.toString());
  }
}

main().catch(console.error);