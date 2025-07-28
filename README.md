# PDF Document Intelligence System

This repository contains a Python-based system for intelligent processing of PDF documents, including parsing, embedding, summarization, and information extraction. The system is containerized with Docker for easy deployment.

## Overview

The system processes PDF documents and provides:
- Document parsing and text extraction
- Content chunking and embedding
- Semantic search capabilities
- Document summarization
- Information extraction

## Requirements

- Docker
- Git
- 8+ GB RAM recommended
- Internet connection (for initial model downloads)

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/sujit-shaha/Adobe-Round-1-B.git
cd Adobe-Round-1-B/persona_doc_intelligence
```

### 2. Build the Docker image

```bash
docker build -t adobe_problem2 .
```

### 3. Run the application

The application can be run in different modes:

#### Process a single PDF file:

```bash
docker run --rm adobe_problem2 python -m app.main --pdf_path /app/PDFs/South\ of\ France\ -\ History.pdf
```

#### Process a specific page range:

```bash
docker run --rm adobe_problem2 python -m app.main --pdf_path /app/PDFs/South\ of\ France\ -\ History.pdf --page_start 1 --page_end 5
```

#### Using a custom language model:

```bash
docker run --rm adobe_problem2 python -m app.main --pdf_path /app/PDFs/South\ of\ France\ -\ History.pdf --llm_model meta-llama/Llama-2-7b-chat-hf
```

#### Process documents with mounted local files:

```bash
docker run --rm -v /path/to/your/pdfs:/app/my_pdfs adobe_problem2 python -m app.main --pdf_path /app/my_pdfs/your_document.pdf
```

## Command Line Arguments

The application supports several command line arguments:

- `--pdf_path`: Path to the PDF file (required)
- `--page_start`: Starting page number (default: 0, which processes from the beginning)
- `--page_end`: Ending page number (default: None, which processes all pages)
- `--llm_model`: Language model to use for summarization (default: 'meta-llama/Llama-2-7b-chat-hf')

## Project Structure

```
persona_doc_intelligence/
├── app/                   # Core application code
│   ├── __init__.py
│   ├── cli.py             # Command-line interface handling
│   ├── config.py          # Configuration settings
│   ├── embeddings.py      # Vector embeddings implementation
│   ├── main.py            # Main entry point
│   ├── model_trainer.py   # Model training utilities
│   ├── pdf_parser.py      # PDF parsing and text extraction
│   ├── ranker.py          # Content ranking implementation
│   ├── summarizer.py      # Document summarization
│   └── utils.py           # Utility functions
├── PDFs/                  # Sample PDF documents
├── models/                # Directory for model storage
│   └── fine_tuned/        # Fine-tuned models
├── output/                # Output storage
├── Dockerfile             # Docker configuration
└── requirements.txt       # Python dependencies
```

## Technical Details

### Models Used

The system uses the following models:
- PDF parsing: PyMuPDF (pymupdf==1.22.3)
- Text embeddings: Sentence Transformers
- LLM: Llama-2-7b-chat-hf (default)
- Vector storage: FAISS

### Dependencies

Key dependencies include:
- PyMuPDF for PDF processing
- PyTorch for deep learning
- HuggingFace Transformers for language models
- FAISS for vector similarity search
- NumPy, Pandas for data manipulation
- scikit-learn for machine learning utilities

## Troubleshooting

### Common Issues:

1. **Out of memory errors**: 
   - Try using a smaller model with `--llm_model` flag
   - Process fewer pages at once with `--page_start` and `--page_end`

2. **Slow processing**:
   - PDF parsing and model loading are the most resource-intensive steps
   - First run will be slower due to model downloading

3. **PDF parsing issues**:
   - Ensure PDF files are not corrupted
   - Some heavily encrypted or image-based PDFs may not parse correctly

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
