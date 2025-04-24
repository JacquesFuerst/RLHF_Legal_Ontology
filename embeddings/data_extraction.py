import fitz  # PyMuPDF
from langchain.schema import Document
from tqdm import tqdm

def extract_text(file_path):
        """
        
        Extract the text from a PDF file.
        
        Parameters:
        file_path (str): The path to the PDF file
        
        Returns:
        list: List of Document objects with the extracted text and metadata (the corresponding page number in the original document)
        """

        documents = []
        with fitz.open(file_path) as doc:
            for page_num, page in enumerate(tqdm(doc, desc="Extracting text"), start=1):
                # print(f"page number: {page_num}")

                page_text = page.get_text()
                # print(f"page text: {page_text}")
                # lines = page_text.split('\n')

                #add the original page number to the metadata to make sure that the model can retrieve it
                metadata = {
                    "paginanummer": page_num,
                }
                documents.append(Document(page_content=page_text, metadata=metadata))
        return documents