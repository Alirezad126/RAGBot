from pypdf import PdfReader, PdfWriter
import glob
import numpy as np
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader

#Reading the files:
data_root = "./pdf_files/"
local_pdfs = glob.glob(data_root + "*.pdf")

filenames = ["main.pdf"]
for local_pdf in local_pdfs:
    pdf_reader = PdfReader(local_pdf)
    pdf_writer = PdfWriter()
    for pagenum in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[pagenum]
        pdf_writer.add_page(page)
        
    with open(local_pdf, "wb") as new_file:
        new_file.seek(0)
        pdf_writer.write(new_file)
        new_file.truncate()
        
        
#Making chunks from the files
documents = []
for idx, file in enumerate(filenames):
    loader = PyPDFLoader(data_root + file)
    document = loader.load()
    
    documents += document
    
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 512,
    chunk_overlap = 20,
    separators=["\n\n","\n"," ",""]
)

docs = text_splitter.split_documents(documents)
