# Use an official Python image as base
FROM python:3.9-slim

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for PyMuPDF
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy only requirements.txt first to leverage Docker cache
COPY requirements.txt /app/

# Create a fixed requirements file
RUN echo "numpy==1.25.0\nfaiss-cpu==1.7.4\npymupdf==1.22.3\ntorch==2.0.1\nhuggingface-hub==0.15.1\nacccelerate==0.21.0\ntransformers==4.30.2\nsentence-transformers==2.2.2\ntokenizers==0.13.3\nsafetensors==0.3.1\npeft==0.4.0\nPyPDF2==3.0.1\npdfminer.six==20221105\nscikit-learn==1.3.0\ntqdm==4.65.0\nnetworkx==2.8.8\npandas==2.0.3\nscipy==1.10.1" > /app/core_requirements.txt

# Fix the accelerate typo
RUN sed -i 's/acccelerate==0.21.0/accelerate==0.21.0/g' /app/core_requirements.txt

# Install dependencies in smaller batches to avoid network timeouts
RUN pip install --upgrade pip && \
    pip install --no-cache-dir numpy==1.25.0 && \
    pip install --no-cache-dir pymupdf==1.22.3 && \
    pip install --no-cache-dir faiss-cpu==1.7.4 && \
    pip install --no-cache-dir torch==2.0.1 --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir transformers==4.30.2 && \
    pip install --no-cache-dir huggingface-hub==0.15.1 && \
    pip install --no-cache-dir accelerate==0.21.0 && \
    pip install --no-cache-dir tokenizers==0.13.3 && \
    pip install --no-cache-dir safetensors==0.3.1 && \
    pip install --no-cache-dir sentence-transformers==2.2.2 && \
    pip install --no-cache-dir peft==0.4.0 && \
    pip install --no-cache-dir PyPDF2==3.0.1 pdfminer.six==20221105 && \
    pip install --no-cache-dir scikit-learn==1.3.0 tqdm==4.65.0 && \
    pip install --no-cache-dir networkx==2.8.8 pandas==2.0.3 scipy==1.10.1

# Copy the rest of the application
COPY . /app/

# Create sample directory for PDF files if it doesn't exist
RUN mkdir -p /app/data

# Expose the port the app runs on (if applicable)
EXPOSE 8080

# Set the default command to run the main script
CMD ["python", "-m", "app.main"]
