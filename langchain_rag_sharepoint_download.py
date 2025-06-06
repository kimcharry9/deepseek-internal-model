import os
import subprocess
from langchain_community.document_loaders.sharepoint import SharePointLoader
from langchain_community.document_loaders.parsers.msword import MsWordParser
from langchain_community.document_loaders.parsers.pdf import PDFMinerParser
from langchain_community.document_loaders.parsers.audio import OpenAIWhisperParser
from langchain_community.document_loaders.parsers.generic import Blob
from dotenv import load_dotenv
from fpdf import FPDF

# Load credentials from .env
load_dotenv(override=True)

# SharePoint setup
document_library_id = "b!TZcDuBcmZUSrOa2XqBEz6PCqIIHxey1OjVKybcDrpVOY1-rp-Y4wQJsti4qHapTs"
handlers = {
    "application/msword": MsWordParser(),
    "application/pdf": PDFMinerParser(),
    "audio/mpeg": OpenAIWhisperParser()
}

# Destination folder to save files
DEST_FOLDER = "/data/open-webui_data/sync_dir"
os.makedirs(DEST_FOLDER, exist_ok=True)

# Initialize the SharePoint document loader
doc_loader = SharePointLoader(
    document_library_id=document_library_id,
    folder_path="read_doc_test",
    recursive=True,
    auth_with_token=True,
    handlers=handlers,
)

# Load documents
documents = doc_loader.load()

# Save document contents locally and print filename
for doc in documents:
    filename = os.path.basename(doc.metadata.get("source", "unknown.txt"))
    base_name = os.path.splitext(filename)[0]
    local_path = os.path.join(DEST_FOLDER, base_name + ".txt")
    pdf_path = os.path.join(DEST_FOLDER, filename)

    with open(local_path, "wb", encoding="utf-8") as f:
        f.write(doc.page_content)
    
    #subprocess.run(["pandoc", local_path, "-o", pdf_path, "--pdf-engine=xelatex", "-V", "mainfont=/usr/share/fonts/custom-thai/THSarabunNew.ttf"], check=True)
    #os.remove(local_path)
    
    print(f"✅ Downloaded: {filename}")
