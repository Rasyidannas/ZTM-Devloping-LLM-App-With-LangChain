#Project: Question Answering on Private Documents
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

def load_document(file):
    from langchain.document_loaders import pyPDFLoader
    print(f'Loading {file}')
    loader = pyPDFLoader(file)
    data = loader.load()
    return data

data = load_document('files/us_constitution.pdf')
# print(data[1],page_content)
# print(data[10].metadata)

print(f'You have {len(data)} pages in your data')