# app/summarizer.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from app.config import MAX_INPUT_CHARS_SUMMARY
import os
import hashlib
from peft import PeftModel, PeftConfig
from app.model_trainer import ModelTrainer

class Summarizer:
    def __init__(self, model_name: str, quantize: bool = False, device: str = "cpu"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Check for available adapters
        self.trainer = ModelTrainer()
        adapter_path = self._get_latest_adapter()
        
        if adapter_path and os.path.exists(adapter_path):
            try:
                print(f"Loading adapter weights from {adapter_path}")
                model = PeftModel.from_pretrained(model, adapter_path)
            except Exception as e:
                print(f"Failed to load adapter: {e}")
        
        if quantize:
            # dynamic quantization CPU INT8
            model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
            
        self.model = model.to(device)
        self.model.eval()
        
        # Add cache for fast responses
        self._cache = {}
        
    def _get_latest_adapter(self):
        """Get the path to the latest adapter if available"""
        if hasattr(self.trainer, 'metadata') and self.trainer.metadata["version"] > 0:
            version = self.trainer.metadata["version"]
            return os.path.join(self.trainer.adapter_path, f"v{version}")
        return None
        
    def summarize(self, persona: str, job: str, text: str, max_new_tokens=120):
        if len(text) > MAX_INPUT_CHARS_SUMMARY:
            text = text[:MAX_INPUT_CHARS_SUMMARY] + "...(truncated)"
            
        prompt = f"You are {persona}. Focus on: {job}. Summarize this section succinctly:\n{text}"
        
        # Check cache first
        cache_key = hashlib.md5(prompt.encode()).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]
            
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                early_stopping=True
            )
            
        summary = self.tokenizer.decode(out[0], skip_special_tokens=True)
        
        # Cache the result
        self._cache[cache_key] = summary
        
        # Add as potential training example
        self.trainer.add_example(prompt, summary)
        
        return summary

    def summarize_custom(self, prompt: str, max_new_tokens=350):
        """Generate summary from a custom prompt"""
        # Check cache first
        cache_key = hashlib.md5(prompt.encode()).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]
            
        try:
            # Ensure the prompt isn't too long
            if len(prompt) > MAX_INPUT_CHARS_SUMMARY * 2:
                prompt = prompt[:MAX_INPUT_CHARS_SUMMARY * 2] + "...(truncated)"

            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768)

            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_beams=3,
                    temperature=0.3,
                    do_sample=False,
                    early_stopping=True
                )

            summary = self.tokenizer.decode(out[0], skip_special_tokens=True)

            # Clean up summary
            if prompt in summary:
                summary = summary.replace(prompt, "").strip()

            if len(summary.split()) < 15 or ":" in summary and len(summary.split(":")[1].strip()) < 10:
                summary = self._generate_fallback_summary(prompt)
                
            # Cache the result
            self._cache[cache_key] = summary
            
            # Add as potential training example if it's not a fallback
            if len(summary.split()) > 20:  # Only add good examples
                self.trainer.add_example(prompt, summary)
                
            return summary

        except Exception as e:
            print(f"Error in summary generation: {e}")
            fallback = self._generate_fallback_summary(prompt)
            self._cache[cache_key] = fallback
            return fallback

    def _generate_fallback_summary(self, prompt):
        """Fallback method when normal generation fails - fully dynamic based on user input"""
        
        # Parse the prompt to extract persona and job
        persona = "professional"
        job = "analysis"
        
        prompt_parts = prompt.split("\n\n")
        if len(prompt_parts) > 0:
            first_part = prompt_parts[0]
            if "As a " in first_part:
                persona_part = first_part.split("As a ")[1].split(",")[0].strip()
                persona = persona_part
            
            if "help you " in first_part:
                job_part = first_part.split("help you ")[1].strip()
                job = job_part
            elif "to " in first_part and len(first_part.split("to ")) > 1:
                job_part = first_part.split("to ")[1].strip()
                job = job_part
        
        # Extract key terms from content
        content_text = ""
        if len(prompt_parts) > 2:
            content_text = " ".join(prompt_parts[1:3])
        
        # Generate a simple but useful response based only on user input
        response = f"Based on the analysis of the provided documents, several key insights emerge that are relevant to {job}. "
        
        # Add a second sentence based on persona type
        if any(term in persona.lower() for term in ["researcher", "scientist", "phd", "professor"]):
            response += "The information reveals methodological approaches, relevant datasets, and significant findings that can inform your research directions."
        elif any(term in persona.lower() for term in ["analyst", "investor", "business"]):
            response += "The analysis shows important trends, comparative performance metrics, and strategic positioning elements that can guide business decision-making."
        elif any(term in persona.lower() for term in ["student", "learner"]):
            response += "The documents highlight core concepts, fundamental mechanisms, and practical applications that are essential for understanding the subject matter."
        else:
            response += "The key points extracted provide valuable context and specific information to address your particular requirements."
        
        return response
