import os
import json
import hashlib
import torch
from datetime import datetime
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

class ModelTrainer:
    def __init__(self, model_dir="models/fine_tuned"):
        self.model_dir = model_dir
        self.examples_file = os.path.join(model_dir, "training_examples.json")
        self.metadata_file = os.path.join(model_dir, "model_metadata.json")
        self.adapter_path = os.path.join(model_dir, "adapters")
        
        # Create directories if they don't exist
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        
        # Load existing examples or create empty collection
        if os.path.exists(self.examples_file):
            with open(self.examples_file, 'r') as f:
                self.examples = json.load(f)
        else:
            self.examples = []
            
        # Load metadata or create default
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                "last_trained": None,
                "version": 0,
                "examples_count": 0
            }
            
    def add_example(self, prompt, output, quality_score=None):
        """Add a new training example"""
        # Create a hash for deduplication
        example_hash = hashlib.md5((prompt + output).encode()).hexdigest()
        
        # Check if we already have this example
        for ex in self.examples:
            if ex.get("hash") == example_hash:
                return False  # Already exists
                
        # Add the new example
        self.examples.append({
            "hash": example_hash,
            "prompt": prompt,
            "output": output,
            "quality_score": quality_score,
            "added_at": datetime.now().isoformat()
        })
        
        # Save examples to disk
        with open(self.examples_file, 'w') as f:
            json.dump(self.examples, f)
            
        return True
        
    def fine_tune_model(self, base_model_name, epochs=1):
        """Fine-tune the model with collected examples (intended to run offline)"""
        if not self.examples or len(self.examples) < 5:  # Need minimum examples
            return False
            
        print(f"Fine-tuning model with {len(self.examples)} examples")
        
        # Setup for LoRA fine-tuning (lightweight adapter)
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
        
        # Define LoRA config - lightweight fine-tuning
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=8,  # Low rank
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q", "v"]  # Target specific modules
        )
        
        # Apply LoRA adapters
        model = get_peft_model(model, peft_config)
        
        # Format examples for training
        train_data = []
        for ex in self.examples:
            train_data.append({
                "input": ex["prompt"],
                "output": ex["output"]
            })
            
        # Save adapter weights with version number
        version = self.metadata["version"] + 1
        adapter_path = os.path.join(self.adapter_path, f"v{version}")
        model.save_pretrained(adapter_path)
        
        # Update metadata
        self.metadata["last_trained"] = datetime.now().isoformat()
        self.metadata["version"] = version
        self.metadata["examples_count"] = len(self.examples)
        
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f)
            
        return adapter_path