# What is LangChain?

LangChain is an OpenSource framework that allows developers working with AI to combine LLMs with external sources of computation and data. LangChain allow you to connect an LLM like GPT-4 to your own sources of data (you can make your LLM Application take actions). LangChain is data-aware and agentic-aware

## LangChain Use-Cases

1. Chat Bots
2. Question Answering Systems
3. Summarization Tools

## 3 Mains in LangChain Concepts

1. **LangChain Components**
   - **LLM Wrappers are** components that encapsulate Large Language Models (LLMs) like GPT-4, providing an interface for these models to interact with external systems and data sources. These wrappers handle the communication between the LLM and other components of the LangChain framework, allowing the LLM to be used in various applications such as chatbots, question-answering systems, and summarization tools.
   - **Prompt Templates** are predefined formats or structures used to generate queries or commands that are sent to Large Language Models (LLMs) like GPT-4. These templates help standardize the interaction between the LLM and the application, ensuring that the inputs to the LLM are consistent and effectively structured to elicit the desired response or behavior.
   - **Indexes** in the context of AI and data management, indexes are used to efficiently access, retrieve, and manage data.
   - **Memory** in AI systems, memory refers to the capability to store information from past interactions or data processing sessions, which can then be used to inform future responses or decisions. Memory have 2 kinds (Short Memory and Long Memory)
2. **Chains** allow us to combine multiple components together to solve a specific task and build an entire LLM application. This modular approach allows developers to link different functionalities, such as LLM Wrappers, Prompt Templates, Indexes, and Memory systems, in a sequence or network to create complex workflows.
3. **Agents** facilitate interaction between the LLM and external APIs. The play a crucial role in decision-making, determining which actions the LLM should undertake. This process involves taking an action, observing the result, and then repeating the cycle until completion.

# What is LLMs?

LLMs alone are often limited in their ability to understand the context, interact with real world, or learn and adapt. LLMs have an impressive general knowledge but are limited to their training data.

# What is Caching in LLM

Caching alone is is the is the practice of storing frequently accessed data or results in temporary, faster storage layer. It can boost performance and save costs with API. Caching optimize interactions with LLMs by reducing API calls and speeding up applications, resulting in a more efficient user experience.

# What is LLM Streaming?

Streaming refers to the process of delivering the response in a continuous stream of data instead of sending the entire response at one. Thi allows the user to recieve the response piece by piece as it is generated, which can improve the user experience and reduce the overall latency.

# What is Prompt Template?

A prompt refers to the input to the model. Prompt templates are a way to create dynamic prompts for LLMs. A prompt template takes a pice of text and injects a user's input into that piece of text. in Langchain there are **PromptTemplates** and **ChatPromptTemplates**

# What is LangChain Chains

Chains allow us to combine multiple components together to solve a specific task and build an entire LLM application

# What is Sequentail Chains

with sequentaiil chains, you can make a series of calls to one or more LLMs. You can take the output from one chain and use it as the input to another chain. This is suit for complex task.  
There are two types of sequential chains:

1. **SimpleSequentialChain** is represents a series of chains, where each individual chain has a single input anda single output, and the output of one step is used as input to the next.
2. General form of sequential chains

# What is Langchain tools?

Langchain tools are like speciallized apps for your LLM. They are tiny code modules that allow it to access information and services.

These tools connect your LLM to search engines, databses, APIs, and more expanding its knowledge and capabilities.

# What is Reasoning and Acting (ReAct)?

ReAct is a new approach that combines reasoning (chain-of-thoughts prompting) and acting capabilities of LLMs.

With ReAct LLMs generate reasoning traces and task-specific actions in an interleaved manner.

# What is Embeddings?

Embeddings are the core of Building LLMs applications. Text embedding are numeric representations of text and are used in NLP and ML tasks.  
The distance between two embedings or two vectores measures their relatedness which translates to the relatedness between the text concepts they represent. Similar embeddings or vectors represent similar concept

**Embeddings Aplications**:

- Text Classification: assign a label to a piece of text.
- Text Clustering: grouping together pieces of text that are similar in meaning.
- Question-Answering: answering a question posed in natural language

# What is Vector Databses?

Vector databases are a new type of database, designed to store and query unstructured data (Unstructured data is data that does not have a fixed schema, such as text, images, and audio).

## How Pipeline for Vector Databses

Vector databses use a combination of different optimized algorithms that all participate in **Approximate Nearest Neighbor(ANN)** search

# What is Pinecone Indexs?

An index is the highest-level organizational unit of vector data in Piecone.
It accepts and stores vectors, server queries over the vectors it contains, and does other vector operations over its contenes.

**Kind of Pinecones Indexes**

1. Serverless indexes: you don't configure or manage any compute or storage resources (they scale automatically).
2. Pod-based indexes: you choose one or more preconfigured unit of hardware(pods).
