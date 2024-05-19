#Project: Question Answering on Private Documents
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

## Runnig Code
data = load_document('files/us_constitution.pdf')
# print(data[1],page_content)
# print(data[10].metadata)

print(f'You have {len(data)} pages in your data')
print(f'There are {len(data[20].page_content)} characters in the page')

data = load_document('files/the_great_gatsby.docx')
print(data[0].page_content)