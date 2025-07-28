import fitz
from typing import List, Dict

def parse_pdf_pages(file_path: str) -> List[Dict]:
    doc = fitz.open(file_path)
    sections = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("blocks")
        blocks.sort(key=lambda b: (b[1], b[0]))
        page_text_parts = []
        for b in blocks:
            text = b[4].strip()
            if text:
                page_text_parts.append(text)
        page_text = " ".join(page_text_parts).strip()
        sections.append({
            "document": file_path,
            "page": page_num + 1,
            "section_title": f"Page {page_num + 1}",
            "content": page_text
        })
    return sections

def chunk_section(section: Dict, chunk_size: int, overlap: int) -> List[Dict]:
    if chunk_size <= 0:
        return [section]
    text = section["content"]
    if len(text) <= chunk_size:
        return [section]
    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]
        chunks.append({
            "document": section["document"],
            "page": section["page"],
            "section_title": f"{section['section_title']}_chunk{idx+1}",
            "content": chunk_text
        })
        if end >= len(text):
            break
        start = end - overlap if overlap > 0 else end
        idx += 1
    return chunks

def parse_and_chunk(file_path: str, chunk_size: int = 0, overlap: int = 0) -> List[Dict]:
    base_sections = parse_pdf_pages(file_path)
    if chunk_size <= 0:
        return base_sections
    expanded = []
    for sec in base_sections:
        expanded.extend(chunk_section(sec, chunk_size, overlap))
    return expanded
