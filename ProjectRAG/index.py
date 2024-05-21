#Project: Question Answering on Private Documents
## ================ Using Pinecone as a Vector DB ================ ##
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

def load_document(file):
    import os
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain_community.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain_community.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    else:
        print('Document format is not supported!')
        return None

    data = loader.load()
    return data

# Wikipedia
def load_from_wikipedia(query, lang='en', load_max_docs=2):
    from langchain_community.document_loaders import WikipediaLoader
    loader = WikipediaLoader(query=query, lang=lang, load_max_docs=load_max_docs)
    data = loader.load()
    return data

def chunk_data(data, chunk_size=256):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    chunks = text_splitter.split_documents(data)
    return chunks

## Embedding Cost
def print_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-3-small')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    # check prices here: https://openai.com/pricing
    print(f'Total Tokens: {total_tokens}')
    print(f'Embedding Cost in USD: {total_tokens / 1000 * 0.00255:.6f}')

## Embedding and Uploading to a Vector Database (Pinecone)
def insert_or_fetch_embeddings(index_name, chunks):
    import pinecone
    from langchain_community.vectorstores import Pinecone
    from langchain_openai import OpenAIEmbeddings
    from pinecone import PodSpec

    pc = pinecone.Pinecone()
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)

    if index_name in pc.list_indexes().names():
        print(f'Index {index_name} already exists. Loading embeddings ...', end='')
        vector_store = Pinecone.from_existing_index(index_name, embeddings)
        print('OK')
    else:
        print(f'Creating index {index_name} and embeddings ...', end='')
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric='cosine',
            spec=PodSpec(environment='gcp-starter')
        )
        vector_store = Pinecone.from_documents(chunks, embeddings, index_name=index_name)
        print('OK')
        return vector_store
    
## Deleting all indexes
def delete_pinecone_index(index_name='all'):
    import pinecone
    pc = pinecone.Pinecone()
    if index_name == 'all':
        indexes = pc.list_indexes().names()
        print('Deleting all indexes ...')
        for index in indexes:
            pc.delete_index(index)
        print('OK')
    else:
        print(f'Deleting index {index_name} ...', end='')
        pc.delete_index(index_name)
        print('OK')

## Asking and Getting Answers 
def ask_and_get_answer(vector_store, q, k=3):
    from langchain.chains import RetrievalQA
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)

    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})

    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    answer = chain.run(q)
    return answer

## Runnig Code
data = load_document('files/us_constitution.pdf')
# print(data[1],page_content)
# print(data[10].metadata)

print(f'You have {len(data)} pages in your data')
print(f'There are {len(data[20].page_content)} characters in the page')

# data = load_document('files/the_great_gatsby.docx')
# print(data[0].page_content)

# data = load_from_wikipedia('GPT-4')
# print(data[0].page_content)

chunks = chunk_data(data)
print(len(chunks))
print(chunks[10].page_content)

print_embedding_cost(chunks)

delete_pinecone_index()

index_name = 'askadocument'
vector_store = insert_or_fetch_embeddings(index_name, chunks)

q = 'What is the whole document about?'
answer = ask_and_get_answer(vector_store, q)
print(answer)

## Looping questing and answering
import time
i = 1
print('Write Quit or Exit to quit.')

while True:
    q = input(f'Question #{i}: ')
    i = i + 1
    if q.lower() in ['quit', 'exit']:
        print('Quitting ... bye bye!')
        time.sleep(2)
        break

    answer = ask_and_get_answer(vector_store, q)
    print(f'\nAnswer: {answer}')
    print(f'\n {"-" * 50} \n')

delete_pinecone_index()

## Loading from Wikipedia
data = load_from_wikipedia('ChatGPT', 'id')
chunks = chunk_data(data)
index_name = 'chatgpt'
vector_store = insert_or_fetch_embeddings(index_name, chunks)

q = "Apa itu chatgpt"
answer = ask_and_get_answer(vector_store, q)
print(answer)

## ================ Using Chroma as a Vector DB ================ ##
def create_embeddings_chroma(chunks, persist_directory='./chroma_db'):
    from langchain.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings

    # Instantiate an embedding model from OpenAI (smaller version for efficiency)
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)  

    # Create a Chroma vector store using the provided text chunks and embedding model, 
    # configuring it to save data to the specified directory 
    vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory) 

    return vector_store  # Return the created vector store

def load_embeddings_chroma(persist_directory='./chroma_db'):
    from langchain.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings

    # Instantiate the same embedding model used during creation
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536) 

    # Load a Chroma vector store from the specified directory, using the provided embedding function
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings) 

    return vector_store  # Return the loaded vector store

data = load_document('files/rag_powered_by_google_search.pdf')
chunks = chunk_data(data, chunk_size=256)
vector_store = create_embeddings_chroma(chunks)

q = 'What is Vertex AI Search'
answer = ask_and_get_answer(vector_store, q)
print(answer)

# Load a Chroma vector store from the specified directory (default ./chroma_db) 
db = load_embeddings_chroma()
q = 'How many pairs of questions and answers had the StackOverflow dataset?'
answer = ask_and_get_answer(vector_store, q)
print(answer)

q = 'Multiply that number by 2'
answer = ask_and_get_answer(vector_store, q)
print(answer)

## Adding Memory (Chat History)
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain  # Import class for building conversational AI chains 
from langchain.memory import ConversationBufferMemory  # Import memory for storing conversation history

# Instantiate a ChatGPT LLM (temperature controls randomness)
llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)  

# Configure vector store to act as a retriever (finding similar items, returning top 5)
retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 5})  


# Create a memory buffer to track the conversation
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

crc = ConversationalRetrievalChain.from_llm(
    llm=llm,  # Link the ChatGPT LLM
    retriever=retriever,  # Link the vector store based retriever
    memory=memory,  # Link the conversation memory
    chain_type='stuff',  # Specify the chain type
    verbose=False  # Set to True to enable verbose logging for debugging
)

def ask_question(q, chain):
    result = chain.invoke({'question': q})
    return result

data = load_document('files/rag_powered_by_google_search.pdf')
chunks = chunk_data(data, chunk_size=256)
vector_store = create_embeddings_chroma(chunks)

q = 'How many pairs of questions and answers had the StackOverflow dataset?'
result = ask_question(q, crc)
print(result)
print(result['answer'])

q = 'Multiply that number by 10'
result = ask_question(q, crc)
print(result['answer'])

q = 'Devide that result by 80'
result = ask_question(q, crc)
print(result['answer'])

for item in result['chat_history']:
    print(item)

## Using a Custom Prompt
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)
retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 5})
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)


system_template = r'''
Use the following pieces of context to answer the user's question.
Before answering translate your response to Indonesian.
If you don't find the answer in the provided context, just respond "I don't know."
---------------
Context: ```{context}```
'''

user_template = '''
Question: ```{question}```
'''

messages= [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template(user_template)
]

qa_prompt = ChatPromptTemplate.from_messages(messages)

crc = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    chain_type='stuff',
    combine_docs_chain_kwargs={'prompt': qa_prompt },
    verbose=True
)

print(qa_prompt)

db = load_embeddings_chroma()
q = 'How many pairs of questions and answers had the StackOverflow dataset?'
result = ask_question(q, crc)
print(result)

q = 'When was Elon Musk born?'
result = ask_question(q, crc)
print(result)

print(result['answer'])