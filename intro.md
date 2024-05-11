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
