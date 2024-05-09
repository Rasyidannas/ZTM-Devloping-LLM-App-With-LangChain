# Python-dotenv
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

# os.environ.get('OPENAI_API_KEY')

# Chat Models: GPT-3.5 Turbo and GPT-4
from langchain_openai import ChatOpenAI
llm = ChatOpenAI()
output = llm.invoke('Explain quantum mechanics in one sentence.', model='gpt-3.5-turbo')
print(output.content)

# this is for show detail our configuration CHatOpenAI
# help(ChatOpenAI)

from langchain.schema import (
    SystemMessage,
    AIMessage,
    HumanMessage
)

messages = [
    SystemMessage(content='You are a physicist and respond only in Indonesian'),
    HumanMessage(content='Explain quantum mechanics in one sentence.')
]

output = llm.invoke(messages)
print(output.content)