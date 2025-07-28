import argparse
from app.model_trainer import ModelTrainer
from app.config import DEFAULT_SUM_MODEL

def main():
    parser = argparse.ArgumentParser(description="Fine-tune model adapters with collected examples")
    parser.add_argument("--model", default=DEFAULT_SUM_MODEL, help="Base model to fine-tune")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    args = parser.parse_args()
    
    trainer = ModelTrainer()
    if len(trainer.examples) < 5:
        print(f"Not enough examples for fine-tuning. Found {len(trainer.examples)}, need at least 5.")
        return
        
    print(f"Fine-tuning model {args.model} with {len(trainer.examples)} examples")
    adapter_path = trainer.fine_tune_model(args.model, args.epochs)
    
    if adapter_path:
        print(f"Successfully fine-tuned model. Adapter saved to: {adapter_path}")
        print("This adapter will be automatically used in future runs.")
    else:
        print("Fine-tuning failed.")

if __name__ == "__main__":
    main()