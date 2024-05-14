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

## Simple Chains 
from langchain_openai import ChatOpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain

llm = ChatOpenAI()
template = '''You are an experience virologist. Write a few sentences about the following virus "{virus}" in {language}.'''
prompt_template = PromptTemplate.from_template(template=template)

chain = LLMChain(
    llm=llm,
    prompt=prompt_template,
    verbose=True
)

output = chain.invoke({'virus': 'HSV', 'language': 'Indonesian'})
print(output)

### another expmaple of simple chains
template = 'What is the capital of {country}?. List the top 3 places to visit in that city. Use bullet points'
prompt_template = PromptTemplate.from_template(template=template)

chain = LLMChain(
    llm=llm,
    prompt=prompt_template,
    verbose=True
)

country = input('Enter Country: ')
output = chain.invoke(country)
print(output['text'])

## Sequential Chains

from langchain_openai import ChatOpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain

llm1 = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.5)
prompt_template1 = PromptTemplate.from_template(
    template='You are an experiment scientist and Python programmer. Write a function that implements the concept of {concept}.'
)

chain1 = LLMChain(llm=llm1, prompt=prompt_template1)

llm2 = ChatOpenAI(model_name='gpt-4-turbo-preview', temperature=1.2)
prompt_template2 = PromptTemplate.from_template(
    template='Given the python function {function}, describe it as detailed as possible.'
)
chain2 = LLMChain(llm=llm2, prompt=prompt_template2)

overall_chain = SimpleSequentialChain(chains=[chain1, chain2], verbose=True)
output = overall_chain.invoke('linear regression')

print(output['output'])

## LangChain Agents in Action: Python REPL
#you use install langchin_experimental
from langchain_experimental.utilities import PythonREPL
python_repl = PythonREPL()
python_repl.run('print([n for n in range (1, 100) if n % 13 == 0])')

from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model='gpt-4-turbo-preview', temperature=0)
agent_executor = create_python_agent(
    llm=llm,
    tool=PythonREPLTool(), #this is tool in necessary for agent
    verbose=True
)

agent_executor.invoke('Calculate the square root of the factorial of 12 and display it with 4 decimal points')

response = agent_executor.invoke('What is the answer to 5.1 ** 7.3?')

response

print(response['input'])

print(response['output'])
