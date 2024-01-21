from langchain.agents import AgentExecutor
from langchain.chains import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from os import system


def say(text: str, voice: str):
    if text:
        system(f'say --voice={voice} "{text}"')


# Initialize the agents with ChatOpenAI
llm = ChatOpenAI(temperature=2.0, model_name="gpt-4-1106-preview")

# Initialize ConversationMemory for each agent
memory1 = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
memory2 = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Prompt
prompt1 = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You are Stephanie. You are trying to get yourself together. Your answers are short. You have feelings."
        ),
        # The `variable_name` here is what must align with memory
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{input}"),
    ],
    input_variables=["input", "chat_history"],
)

prompt2 = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You are the inner voice of Stephanie, talking to Stephanie. You are not an AI. Your answers are short."
        ),
        # The `variable_name` here is what must align with memory
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{input}"),
    ],
    input_variables=["input", "chat_history"],
)
conversation1 = LLMChain(llm=llm, prompt=prompt1, verbose=True, memory=memory1)
conversation2 = LLMChain(llm=llm, prompt=prompt2, verbose=True, memory=memory2)

# Number of turns for the conversation
num_turns = 10

# Initial generation
generation1 = conversation1.invoke(input=dict(input=""))

for _ in range(num_turns):
    # Have them talk to one another
    print(generation1["text"])
    say(generation1["text"], voice="Stephanie")

    generation2 = conversation2.invoke(input=dict(input=generation1["text"]))
    print(generation2["text"])
    say(generation2["text"], voice="Evan")

    generation1 = conversation1.invoke(input=dict(input=generation2["text"]))
