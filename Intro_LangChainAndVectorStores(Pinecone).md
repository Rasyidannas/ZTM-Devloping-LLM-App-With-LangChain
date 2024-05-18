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
