# authenticating to Pinecone. 
# the API KEY is in .env
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

from pinecone import Pinecone, ServerlessSpec

# Initilizing and authenticating the pinecone client
pc = Pinecone()
# pc = Pinecone(api_key='YOUR_API_KEY')

# checking authentication
pc.list_indexes()