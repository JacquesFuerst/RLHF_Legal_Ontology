from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

#TODO: maybe do not split this way, for now one page is one chunk, maybe make the chunks smaller such that they are more specific


def split_text(file_path):
    """
    Split text used in the RAG into chunks. load text based on document type (PDF/txt).
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50) #TODO: these are design choices I need to think about
    return splitter.split_documents(documents)