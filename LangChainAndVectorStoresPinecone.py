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

pc.list_indexes().names()

#creating pinecone indexes with serveeless
from pinecone import ServerlessSpec
index_name = 'langchain'
if index_name not in pc.list_indexes().names():
    print(f"Creating index {index_name}")
    pc.create_index(
        name=index_name, 
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1",
            )
        )
    print('Index created! :D')
else:
    print(f"Index {index_name} already exists")

#deleting pinecone indexes
# index_name = 'langchain'
# if index_name in pc.list_indexes().names():
#     print(f'Deleting index {index_name}...')
#     pc.delete_index(index_name)
#     print('Done')
# else: 
#     print(f'Index {index_name} does not exist!')

index = pc.Index(index_name)
index.describe_index_stats()