import os, sys, time, json
import faiss
import numpy as np
from datetime import datetime
from tqdm import tqdm
from app.pdf_parser import parse_and_chunk
from app.embeddings import EmbeddingIndex
from app.summarizer import Summarizer
from app.utils import expand_paths, set_seed, save_json, TimeBudget, get_model_size_mb
from app.config import (DEFAULT_EMBED_MODEL, DEFAULT_SUM_MODEL,
                        DEFAULT_TOP_K, TIME_BUDGET_SECONDS,
                        FINE_TUNE_EXAMPLES_THRESHOLD)
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

def read_challenge_input(input_file):
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        documents = [doc["filename"] for doc in data.get("documents", [])]
        persona = data.get("persona", {}).get("role", "")
        job = data.get("job_to_be_done", {}).get("task", "")
        
        input_dir = os.path.dirname(os.path.abspath(input_file))
        pdf_folder = os.path.join(input_dir, "PDFs")
        
        for i, doc in enumerate(documents):
            if not os.path.isabs(doc):
                pdf_path = os.path.join(pdf_folder, doc)
                if not os.path.exists(pdf_path):
                    print(f"Warning: PDF file not found: {pdf_path}")
                documents[i] = pdf_path
        
        return {
            "docs": documents,
            "persona": persona,
            "job": job
        }
    except Exception as e:
        print(f"Error reading input file: {e}")
        return None

def interactive_if_missing(args):
    if not args.docs and not args.dir:
        print("No documents specified. Enter path(s) (comma separated):")
        raw = input("> ").strip()
        if raw:
            args.docs = [r.strip() for r in raw.split(",")]
    if not args.persona:
        args.persona = input("Enter Persona: ").strip()
    if not args.job:
        args.job = input("Enter Job to be done: ").strip()
    return args

def generate_dynamic_section_title(document_name, content, current_title, persona=None, job=None):
    if current_title.lower().startswith("page "):
        doc_topic = document_name.split(" - ")[-1].replace(".pdf", "")
        
        lines = content.split("\n")
        first_lines = lines[:3]
        
        for line in first_lines:
            line = line.strip()
            if 3 <= len(line) <= 50 and not line.isupper() and not line.islower():
                return line
        
        sentences = content.split('. ')
        first_sentence = sentences[0] if sentences else ""
        words = first_sentence.split()[:10]  
        
        key_phrases = []
        topic_keywords = ["guide", "tips", "experience", "adventure", "tour", 
                         "activities", "attractions", "highlights", "overview",
                         "introduction", "essentials", "recommendations"]
        
        
        for i, word in enumerate(words):
            if word.lower() in topic_keywords and i > 0:
                
                phrase = f"{words[i-1]} {word}"
                if len(phrase) > 5:  
                    key_phrases.append(phrase)
        
        if key_phrases:
            return f"{doc_topic} {key_phrases[0].title()}"
        
        if len(words) >= 3:
            contextual_words = " ".join(words[:3])
            return f"{doc_topic} - {contextual_words}..."
        
        return f"{doc_topic} Overview"
    
    return current_title

def main(input_file=None, output_file=None):
    start_time = time.time()
    
    if input_file:
        input_data = read_challenge_input(input_file)
        if not input_data:
            print("ERROR: Invalid input file format.")
            sys.exit(1)
        
        docs = input_data["docs"]
        persona = input_data["persona"]
        job = input_data["job"]
        dir_path = None
        
        embed_model_name = DEFAULT_EMBED_MODEL
        sum_model_name = DEFAULT_SUM_MODEL
        chunk_size = 0
        chunk_overlap = 150
        quantize = False
        seed = 42
        top_k = DEFAULT_TOP_K
    else:
        from app.cli import build_parser
        parser = build_parser()
        args = parser.parse_args()
        args = interactive_if_missing(args)

        docs = args.docs
        dir_path = args.dir
        persona = args.persona
        job = args.job
        embed_model_name = args.embed_model or DEFAULT_EMBED_MODEL
        sum_model_name = args.sum_model or DEFAULT_SUM_MODEL
        chunk_size = args.chunk_size
        chunk_overlap = args.chunk_overlap
        quantize = args.quantize
        seed = args.seed
        top_k = args.top_k or DEFAULT_TOP_K

    set_seed(seed)

    pdf_paths = expand_paths(docs, dir_path)
    if not pdf_paths:
        print("ERROR: No PDF files found.")
        sys.exit(1)

    print(f"Found {len(pdf_paths)} PDF(s). Parsing...")
    all_sections = []
    doc_to_sections = defaultdict(list)
    
    max_workers = min(len(pdf_paths), multiprocessing.cpu_count())
    
    def process_pdf(pdf):
        try:
            secs = parse_and_chunk(pdf, chunk_size, chunk_overlap)
            return pdf, secs
        except Exception as e:
            print(f"Failed to parse {pdf}: {e}")
            return pdf, []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_pdf, pdf): pdf for pdf in pdf_paths}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Parsing PDFs"):
            pdf, secs = future.result()
            if secs:
                all_sections.extend(secs)
                doc_to_sections[pdf].extend(secs)

    if not all_sections:
        print("ERROR: No sections extracted.")
        sys.exit(1)

    print(f"Total sections (after chunking): {len(all_sections)}")

    doc_indices = {}
    for i, section in enumerate(all_sections):
        if section["document"] not in doc_indices:
            doc_indices[section["document"]] = []
        doc_indices[section["document"]].append(i)

    def init_embedding_model():
        return EmbeddingIndex(embed_model_name)
    
    def init_summarizer():
        return Summarizer(sum_model_name, quantize=quantize)
    with ThreadPoolExecutor(max_workers=2) as executor:
        emb_future = executor.submit(init_embedding_model)
        sum_future = executor.submit(init_summarizer)
        
        emb_index = emb_future.result()
        summarizer = sum_future.result()

    query = f"As a {persona}, {job}"
    
    print(f"Retrieving sections from each document...")
    
    retrieved = []
    sections_per_doc = max(1, top_k // len(pdf_paths))
    
    all_texts = []
    section_mapping = []
    
    min_content_length = 20
    
    for doc_path in pdf_paths:
        if doc_path not in doc_indices:
            continue
            
        indices = doc_indices[doc_path]
        doc_sections = [all_sections[i] for i in indices]
        
        for section in doc_sections:
            if len(section['content']) < min_content_length:
                continue
                
            all_texts.append(section['content'])
            section_mapping.append(section)
    
    print(f"Generating embeddings for {len(all_texts)} sections...")
    batch_size = min(64, len(all_texts))
    all_embeddings = emb_index.model.encode(all_texts, convert_to_numpy=True, 
                                           batch_size=batch_size,
                                           show_progress_bar=False)
    
    nlist = 1
    if len(all_texts) > 1000:
        nlist = 100
    
    quantizer = faiss.IndexFlatL2(emb_index.dim)
    index = faiss.IndexIVFFlat(quantizer, emb_index.dim, nlist)
    
    if not index.is_trained:
        index.train(all_embeddings)
    index.add(all_embeddings)
    
    index.nprobe = 1
    
    q_vec = emb_index.model.encode([query], convert_to_numpy=True)
    
    k = min(top_k, len(all_texts))
    distances, indices = index.search(q_vec, k)
    
    retrieved = []
    global_rank = 1
    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
        if idx >= len(section_mapping) or idx < 0:
            continue
            
        section = section_mapping[idx]
        doc_name = os.path.basename(section["document"])
        dynamic_title = generate_dynamic_section_title(
            doc_name, 
            section["content"], 
            section["section_title"],
            persona,
            job
        )
        
        retrieved.append({
            "document": section["document"],
            "page_number": section["page"],
            "section_title": dynamic_title,
            "content": section["content"],
            "importance_rank": global_rank,
            "score": float(dist)
        })
        global_rank += 1
    
    retrieved.sort(key=lambda x: x["score"])
    
    print(f"Retrieved {len(retrieved)} sections from {len(pdf_paths)} documents")
    print(f"Summarizing with model: {sum_model_name} (quantize={quantize})")
    summarizer = Summarizer(sum_model_name, quantize=quantize)

    time_budget = TimeBudget(TIME_BUDGET_SECONDS)
    subsection_analysis = []
    
    def summarize_section(sec):
        if time_budget.exceeded(safety_margin=10):
            return None
        try:
            summary = summarizer.summarize(persona, job, sec["content"])
            return {
                "section": sec,
                "summary": summary
            }
        except Exception as e:
            print(f"Error summarizing section: {e}")
            return None
    
    retrieved.sort(key=lambda x: x["score"])
    sum_workers = min(4, multiprocessing.cpu_count())
    with ThreadPoolExecutor(max_workers=sum_workers) as executor:
        futures = {executor.submit(summarize_section, sec): sec for sec in retrieved}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Summarizing"):
            result = future.result()
            if result and not time_budget.exceeded():
                sec = result["section"]
                summary = result["summary"]
                
                summary_entry = {
                    "document": sec["document"],
                    "page_number": sec["page_number"],
                    "refined_text": summary
                }
                subsection_analysis.append(summary_entry)

    from datetime import datetime
    output = {
        "metadata": {
            "input_documents": [os.path.basename(path) for path in pdf_paths],
            "persona": persona,
            "job_to_be_done": job,
            "processing_timestamp": datetime.now().isoformat()
        },
        "extracted_sections": [
            {
                "document": os.path.basename(s["document"]),
                "section_title": s["section_title"],
                "importance_rank": s["importance_rank"],
                "page_number": s["page_number"]
            } for s in retrieved
        ],
        "subsection_analysis": [
            {
                "document": os.path.basename(s["document"]),
                "refined_text": s["refined_text"],
                "page_number": s["page_number"]
            } for s in subsection_analysis
        ]
    }

    if output_file:
        save_json(output, output_file)
        print(f"\nDone. JSON saved to: {output_file}")
    else:
        save_json(output, "output/result.json")
        print("\nDone. JSON saved to: output/result.json")
        
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\nTotal processing time: {int(hours):02d}:{int(minutes):02d}:{seconds:.2f}")
    print(f"(Total seconds: {elapsed_time:.2f})")
    del all_embeddings
    del all_texts
    del section_mapping
    del index

if __name__ == "__main__":
    app_folder = os.path.dirname(os.path.abspath(__file__))
    
    challenge_input = None
    possible_locations = [
        "challenge1b_input.json",
        os.path.join(os.path.dirname(__file__), "challenge1b_input.json"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "challenge1b_input.json")
    ]
    
    for location in possible_locations:
        if os.path.exists(location):
            challenge_input = location
            break
            
    if challenge_input:
        output_file = os.path.join(app_folder, "challenge1b_output.json")
        main(challenge_input, output_file)
    elif len(sys.argv) > 1 and sys.argv[1].endswith('.json'):
        input_file = sys.argv[1]
        if len(sys.argv) > 2 and sys.argv[2].endswith('.json'):
            output_file = sys.argv[2]
        else:
            output_file = os.path.join(app_folder, "challenge1b_output.json")
        main(input_file, output_file)
    else:
        main()
