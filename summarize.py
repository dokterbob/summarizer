#!/usr/bin/env python

import argparse

from langchain.chains import StuffDocumentsChain, LLMChain
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain_core.callbacks.stdout import StdOutCallbackHandler
from langchain_core.runnables.config import RunnableConfig

from langchain_openai import ChatOpenAI
# from langchain_community.llms import GPT4All
# from langchain_community.llms import LlamaCpp


from langchain.globals import set_verbose

set_verbose(True)

parser = argparse.ArgumentParser()
parser.add_argument("path")

args = parser.parse_args()
path = args.path
print(f"Summarizing directory {path}")

loader = DirectoryLoader(path=args.path, recursive=True, show_progress=True)
docs = loader.load()

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1024, chunk_overlap=32
)
split_docs = text_splitter.split_documents(docs)

llm = ChatOpenAI(
    temperature=0,
    base_url="http://localhost:8080/v1",
    api_key="sk-no-key-required",
    max_tokens=8192,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)
# llm = GPT4All(
#     model="/Users/drbob/Library/Application Support/nomic.ai/GPT4All/mistral-7b-instruct-v0.1.Q4_0.gguf",
#     callbacks=[StdOutCallbackHandler()],
#     verbose=True,
#     device="gpu",
# )

# llm = LlamaCpp(
#     model_path="/Users/drbob/Library/Application Support/nomic.ai/GPT4All/mistral-7b-instruct-v0.1.Q4_0.gguf",
#     temperature=0,
#     n_gpu_layers=32,
#     n_batch=8,
#     n_ctx=8192,
#     max_tokens=8192,
#     streaming=True,
#     callbacks=[StdOutCallbackHandler()],
#     verbose=True,  # Verbose is required to pass to the callback manager
# )


# Map
map_template = """<s>[INST] Write an extensive, complete and accurate summary of the following:
---
{docs} [/INST]"""
map_prompt = PromptTemplate.from_template(map_template)
map_chain = LLMChain(
    llm=llm,
    prompt=map_prompt,
    callbacks=[StdOutCallbackHandler()],
)

# Reduce
reduce_template = """<s>[INST] The following is set of summaries:
---
{docs}
---
Take these and accurately distill it into a consolidated, complete and structured and complete summary. [/INST]"""
reduce_prompt = PromptTemplate.from_template(reduce_template)

# Run chain
reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

# Takes a list of documents, combines them into a single string, and passes this to an LLMChain
combine_documents_chain = StuffDocumentsChain(
    llm_chain=reduce_chain,
    document_variable_name="docs",
    callbacks=[StdOutCallbackHandler()],
)

# Combines and iteratively reduces the mapped documents
reduce_documents_chain = ReduceDocumentsChain(
    # This is final chain that is called.
    combine_documents_chain=combine_documents_chain,
    # If documents exceed context for `StuffDocumentsChain`
    collapse_documents_chain=combine_documents_chain,
    # The maximum number of tokens to group documents into.
    token_max=2048,  # Context window is 2048
    callbacks=[StdOutCallbackHandler()],
)

map_reduce_chain = MapReduceDocumentsChain(
    # Map chain
    llm_chain=map_chain,
    # Reduce chain
    reduce_documents_chain=reduce_documents_chain,
    # The variable name in the llm_chain to put the documents in
    document_variable_name="docs",
    # Return the results of the map steps in the output
    return_intermediate_steps=True,
    callbacks=[StdOutCallbackHandler()],
)

result = map_reduce_chain.invoke(input={"input_documents": split_docs})
print("-------------- FINAL OUTPUT --------------")
print(result["output_text"])
