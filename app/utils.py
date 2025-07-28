import glob, os, json, random, numpy as np, time
from typing import List
from pathlib import Path

def expand_paths(docs: List[str], directory: str = None) -> List[str]:
    paths = []
    
    if docs and all(not os.path.isabs(doc) and '/' not in doc and '\\' not in doc for doc in docs):
        current_dir = Path.cwd()
        pdf_dir = None
        
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
    
    if paths:
        return paths
            
    if directory:
        for f in os.listdir(directory):
            if f.lower().endswith(".pdf"):
                paths.append(os.path.join(directory, f))
    for pattern in docs:
        paths.extend(glob.glob(pattern))
    
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
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

class TimeBudget:
    def __init__(self, budget_seconds: float):
        self.start = time.time()
        self.budget = budget_seconds
        self.checkpoints = {}

    def mark_checkpoint(self, name: str):
        self.checkpoints[name] = time.time() - self.start
        return self.time_left()

    def time_left(self):
        return max(0, self.budget - (time.time() - self.start))

    def exceeded(self, safety_margin=5):
        return self.time_left() < safety_margin
        
    def time_percentage_used(self):
        elapsed = time.time() - self.start
        return min(100, (elapsed / self.budget) * 100)

def get_model_size_mb(model):
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size = param_size + buffer_size
    return total_size / (1024 * 1024)
