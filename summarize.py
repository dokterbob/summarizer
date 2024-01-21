#!/usr/bin/env python

import argparse

from langchain.chains import StuffDocumentsChain, LLMChain
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from langchain.globals import set_verbose

set_verbose(True)

parser = argparse.ArgumentParser()
parser.add_argument("path")

args = parser.parse_args()
path = args.path
print(f"Summarizing directory {path}")

loader = DirectoryLoader(
    path=args.path, recursive=True, show_progress=True, glob="**/*.md"
)
docs = loader.load()

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000, chunk_overlap=0
)
split_docs = text_splitter.split_documents(docs)

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-1106")

# Map
map_template = """Write an extensive, complete and accurate summary of the following:
---
{docs}
---
Helpful Answer:"""
map_prompt = PromptTemplate.from_template(map_template)
map_chain = LLMChain(llm=llm, prompt=map_prompt)

# Reduce
reduce_template = """The following is set of summaries:
---
{docs}
---
Take these and distill it into a final, consolidated, structured and complete overview.
Helpful Answer:"""
reduce_prompt = PromptTemplate.from_template(reduce_template)

# Run chain
reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

# Takes a list of documents, combines them into a single string, and passes this to an LLMChain
combine_documents_chain = StuffDocumentsChain(
    llm_chain=reduce_chain, document_variable_name="docs"
)

# Combines and iteratively reduces the mapped documents
reduce_documents_chain = ReduceDocumentsChain(
    # This is final chain that is called.
    combine_documents_chain=combine_documents_chain,
    # If documents exceed context for `StuffDocumentsChain`
    collapse_documents_chain=combine_documents_chain,
    # The maximum number of tokens to group documents into.
    token_max=4000,
)

map_reduce_chain = MapReduceDocumentsChain(
    # Map chain
    llm_chain=map_chain,
    # Reduce chain
    reduce_documents_chain=reduce_documents_chain,
    # The variable name in the llm_chain to put the documents in
    document_variable_name="docs",
    # Return the results of the map steps in the output
    # return_intermediate_steps=True,
)

result = map_reduce_chain.invoke(input=split_docs)
print("-------------- FINAL OUTPUT --------------")
print(result.output_text)
