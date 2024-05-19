## ================ PINECONE ================ ##
# authenticating to Pinecone. 
# the API KEY is in .env
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

from pinecone import Pinecone, ServerlessSpec

# Initilizing and authenticating the pinecone client
pc = Pinecone()
# pc = Pinecone(api_key='YOUR_API_KEY')

# checking authentication and read index in the pinecone
pc.list_indexes()

## Working with Pinecone Indexes
pc.list_indexes().names()

### creating pinecone indexes with serveeless
# from pinecone import ServerlessSpec
# index_name = 'langchain'
# if index_name not in pc.list_indexes().names():
#     print(f"Creating index {index_name}")
#     pc.create_index(
#         name=index_name, 
#         dimension=1536,
#         metric="cosine",
#         spec=ServerlessSpec(
#             cloud="aws",
#             region="us-east-1",
#             )
#         )
#     print('Index created! :D')
# else:
#     print(f"Index {index_name} already exists")

### deleting pinecone indexes
# index_name = 'langchain'
# if index_name in pc.list_indexes().names():
#     print(f'Deleting index {index_name}...')
#     pc.delete_index(index_name)
#     print('Done')
# else: 
#     print(f'Index {index_name} does not exist!')

# index = pc.Index(index_name)
# index.describe_index_stats()

## ================ Working with Vectors ================ ##
### insering vectors
# import random
# vectors = [[random.random() for _ in range(1536)] for v in range(5)]
# # print(vectors)
# ids = list('abcde')

# index_name = 'langchain'
# index = pc.Index(index_name)

# index.upsert(vectors=zip(ids, vectors))

# ### update vectors
# index.upsert(vectors=[('c', [0.5] * 1536)])

# ### fetch vectors
# # index = pc.Index(index_name)
# index.fetch(ids=['c', 'd'])

# ### delete vectors
# index.delete(ids=['b', 'c'])

# index.describe_index_stats()

# index.fetch(ids=['x'])

# ### query 
# query_vector = [random.random() for _ in range(1536)]
# # print(query_vector)

# # This retrieves the query_vectors of the most similar records in your index, along with their similarity scores.
# index.query(
#     vector=query_vector,
#     top_k=3,
#     include_values=False
# )

## ================ Namespaces  ================ ##
# # index.describe_index_stats()
# index = pc.Index('langchain')

# import random
# vectors = [[random.random() for _ in range(1536)] for v in range(5)]
# ids = list('abcde')
# index.upsert(vectors=zip(ids, vectors))

# vectors = [[random.random() for _ in range(1536)] for v in range(3)]
# ids = list('xyz')
# index.upsert(vectors=zip(ids, vectors), namespace='first-namespace')

# vectors = [[random.random() for _ in range(1536)] for v in range(3)]
# ids = list('aq')
# index.upsert(vectors=zip(ids, vectors), namespace='second-namespace')

# index.describe_index_stats()

# ### get specific namespace
# index.fetch(ids=['x'], namespace='first-namespace')

# ### delete specific id in namespace
# index.delete(ids=['x'], namespace='first-namespace')

# ### delete namespace
# index.delete(delete_all=True, namespace='first-namespace')

## ================ Splitting and Embedding Text Using LangChain ================ ##

import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

from langchain.text_splitter import RecursiveCharacterTextSplitter
with open('files/churchill_speech.txt') as f:
    churchill_speech = f.read()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=len
)

chunks = text_splitter.create_documents([churchill_speech])
# print(chunks[0])
# print(chunks[10].page_content)
print(f'Now you have {len(chunks)}')

## ================ Embeding Cost ================ ##
def print_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-3-small')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    # check prices here: https://openai.com/pricing
    print(f'Total Tokens: {total_tokens}')
    print(f'Embedding Cost in USD: {total_tokens / 1000 * 0.00255:.6f}')

print_embedding_cost(chunks)

## ================ Creating Embeding ================ ##
from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

vector = embeddings.embed_query(chunks[0].page_content)
print(vector)

## ================ Inserting the Embeding into a Pinecone Index ================ ##
import pinecone
from langchain_community.vectorstores import Pinecone
pc = pinecone.Pinecone()

# deleting all indexes
indexes = pc.list_indexes().names()
for i in indexes:
    print('Deleting all indexes ... ', end='')
    pc.delete_index(i)
    print('Done')

# creating an index
from pinecone import ServerlessSpec
index_name = 'churchill-speech'
if index_name not in pc.list_indexes().names():
    print(f'Creating index {index_name}')
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ) 
    )
    print('Index created! 😊')
else:
    print(f'Index {index_name} already exists!')

# processing the input documents, generating embeddings using the provided `OpenAIEmbeddings` instance,
# inserting the embeddings into the index and returning a new Pinecone vector store object. 
vector_store = Pinecone.from_documents(chunks, embeddings, index_name=index_name)

index = pc.Index(index_name)
index.describe_index_stats()


## ================ Asking Questions (Similarity Search) ================ ##
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# Initialize the LLM with the specified model and temperature
llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.2)

# Use the provided vector store with similarity search and retrieve top 3 results
retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 3})

# Create a RetrievalQA chain using the defined LLM, chain type 'stuff', and retriever
chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever)

# query = 'Answer only from the provided input. Where should we fight?'
# query = 'Who was the King of Belgium at the time?'
query = 'What about the French Armies?'
answer = chain.invoke(query)
print(answer)