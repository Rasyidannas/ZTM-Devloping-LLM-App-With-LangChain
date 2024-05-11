import time

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

# this is for show detail our configuration ChatOpenAI
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

# Caching LLM Responses
## In-Memory Cache
from langchain.globals import set_llm_cache
from langchain_openai import OpenAI
llm = OpenAI(model_name='gpt-3.5-turbo-instruct')

start_time = time.time()
from langchain.cache import  InMemoryCache
set_llm_cache(InMemoryCache())
prompt = 'Tell me a joke that a toddler can understand'
llm.invoke(prompt)
end_time = time.time()
print(f'Execution taken: {end_time - start_time}')

start_time = time.time()
llm.invoke(prompt)
end_time = time.time()
print(f'Execution taken: {end_time - start_time}')

## SQLite Caching
from langchain.cache import SQLiteCache
set_llm_cache(SQLiteCache(database_path='.langchain.db'))

### First request (not in cahce, takes Longer)
llm.invoke('Tell me a joke')

### Second request (cached, faster)
llm.invoke('Tell me a joke')

## LLM Streaming
from langchain_openai import ChatOpenAI

llm = ChatOpenAI()
prompt = 'Write a rock song about the Moon and a Raven.'
print(llm.invoke(prompt).content)

### Enable streaming
for chunk in llm.stream(prompt):
    print(chunk.content, end='', flush=True)

## Prompt Templates
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

template = '''You are an experience virologist. Write a few sentences about the following virus {virus} in {language}.'''
prompt_template = PromptTemplate.from_template(template=template)

prompt = prompt_template.format(virus='hiv', language='german')

llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)
output = llm.invoke(prompt)
print(output.content)

## ChatPromptTemplates
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage

chat_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content='You respond only in the JSON format.'),
        HumanMessagePromptTemplate.from_template('Top {n} countries in {area} by population.')
    ]
)

messages = chat_template.format_messages(n='10', area='World')
print(messages)

from langchain_openai import ChatOpenAI
llm = ChatOpenAI()
output = llm.invoke(messages)
print(output.content)