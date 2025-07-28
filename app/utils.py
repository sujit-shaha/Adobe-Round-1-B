import glob, os, json, random, numpy as np, time
from typing import List
from pathlib import Path

def expand_paths(docs: List[str], directory: str = None) -> List[str]:
    paths = []
    
    # Handle collection structure - check if docs contains just filenames
    if docs and all(not os.path.isabs(doc) and '/' not in doc and '\\' not in doc for doc in docs):
        current_dir = Path.cwd()
        pdf_dir = None
        
        # Look for a PDFs directory up to 3 levels up
        for i in range(4):
            test_dir = current_dir / "PDFs"
            if test_dir.exists():
                pdf_dir = test_dir
                break
            current_dir = current_dir.parent
            
        if pdf_dir:
            for doc in docs:
                full_path = pdf_dir / doc
                if full_path.exists():
                    paths.append(str(full_path))
    
    # If we found paths, return them
    if paths:
        return paths
            
    # Otherwise, use the original logic
    if directory:
        for f in os.listdir(directory):
            if f.lower().endswith(".pdf"):
                paths.append(os.path.join(directory, f))
    for pattern in docs:
        paths.extend(glob.glob(pattern))
    
    # dedupe preserving order
    seen = set()
    unique = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def save_json(data, path):
    # If the path contains a directory, make sure it exists
    directory = os.path.dirname(path)
    if directory:  # Only try to create directory if one is specified
        os.makedirs(directory, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

class TimeBudget:
    def __init__(self, budget_seconds: float):
        self.start = time.time()
        self.budget = budget_seconds
        self.checkpoints = {}  # Track time at various stages

    def mark_checkpoint(self, name: str):
        """Mark a checkpoint to track time spent on different operations"""
        self.checkpoints[name] = time.time() - self.start
        return self.time_left()

    def time_left(self):
        return max(0, self.budget - (time.time() - self.start))

    def exceeded(self, safety_margin=5):
        return self.time_left() < safety_margin
        
    def time_percentage_used(self):
        """Calculate percentage of time budget used"""
        elapsed = time.time() - self.start
        return min(100, (elapsed / self.budget) * 100)

def get_model_size_mb(model):
    """Calculate the size of a PyTorch model in MB"""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size = param_size + buffer_size
    return total_size / (1024 * 1024)
