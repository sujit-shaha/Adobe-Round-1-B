# app/cli.py
import argparse

def build_parser():
    p = argparse.ArgumentParser(
        description="Persona-driven document intelligence (terminal mode)"
    )
    p.add_argument("--docs", nargs="*", default=[], help="PDF file paths or glob patterns.")
    p.add_argument("--dir", type=str, default=None, help="Directory containing PDFs.")
    p.add_argument("--persona", type=str, help="Persona description.")
    p.add_argument("--job", type=str, help="Job to be done description.")
    p.add_argument("--top-k", type=int, default=5, help="Number of sections to retrieve.")
    p.add_argument("--embed-model", type=str, default=None, help="Embedding model override.")
    p.add_argument("--sum-model", type=str, default=None, help="Summarization model override.")
    p.add_argument("--chunk-size", type=int, default=0, help="Chunk size in characters (0 = whole page).")
    p.add_argument("--chunk-overlap", type=int, default=150, help="Overlap between chunks.")
    p.add_argument("--quantize", action="store_true", help="Enable dynamic INT8 quantization for summarizer.")
    p.add_argument("--out", type=str, default="output/result.json", help="Output JSON path.")
    p.add_argument("--seed", type=int, default=42)
    return p
