import pymupdf
from pathlib import Path
import faiss
from sentence_transformers import SentenceTransformer


def create_docs():
    pdf_path = Path(__file__).with_name("rag_test_file.pdf")
    doc = pymupdf.open(pdf_path)
    pages = []
    for page_number,page in enumerate(doc,start=1):
        pages.append({
            "page":page_number,
            "text":page.get_text()
        })
    selected_pages = pages[1:5]
    docs = []
    for page in selected_pages:
        docs.append(page["text"])
    return docs


def chunk_text_paragraph(text,max_chars=1000):
  paragraphs = text.split("\n")
  chunks = []
  metadata = []
  current = ""

  id = 0
  for paragraph in paragraphs:
    if len(current) + len(paragraph) <= max_chars:
      current += paragraph + "\n"

    else:
      chunks.append(current.strip())
      metadata.append({"text":current,
                     "chunk_id":id})
      id += 1
      current = paragraph + "\n"


  if current.strip():
    chunks.append(current.strip())
    metadata.append({"text":current,
                     "chunk_id":id})
    id += 1


  return chunks,metadata

def embedDoc(docs):
    chunks_all = []
    meta_data_all = []

    for doc in docs:
        chunks,metadata = chunk_text_paragraph(doc,max_chars=40)
        chunks_all.extend(chunks)
        meta_data_all.append(metadata)

    model = SentenceTransformer("all-MiniLM-L6-v2")

    embedding = model.encode(chunks_all)

    dimension = embedding.shape[1]
    index = faiss.IndexFlatL2(dimension) # its DB analyses  chunks of similarties and create distance based on that
    index.add(embedding)
    return index,chunks_all