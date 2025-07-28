import os
import sys
import json
from pathlib import Path
from app.main import main as process_document

def process_challenge_collection(collection_path):
    """Process a challenge collection folder"""
    collection_dir = Path(collection_path)
    
    # Find input and output files
    input_file = collection_dir / "challenge1b_input.json"
    output_file = collection_dir / "challenge1b_output.json"
    
    if not input_file.exists():
        print(f"Error: Input file not found at {input_file}")
        return False
    
    print(f"Processing collection: {collection_dir.name}")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    
    # Call the main processing function
    process_document(str(input_file), str(output_file))
    
    print(f"Completed processing collection: {collection_dir.name}")
    return True

def main():
    if len(sys.argv) < 2:
        print("Usage: python process_challenge.py <challenge_folder>")
        sys.exit(1)
        
    challenge_path = sys.argv[1]
    if not os.path.exists(challenge_path):
        print(f"Error: Challenge folder not found: {challenge_path}")
        sys.exit(1)
        
    # Process a single collection or all collections in the folder
    challenge_dir = Path(challenge_path)
    
    # Check if this is a collection with PDFs or a parent folder with multiple collections
    if (challenge_dir / "PDFs").exists():
        # Single collection
        success = process_challenge_collection(challenge_dir)
    else:
        # Multiple collections
        collections = [d for d in challenge_dir.iterdir() if d.is_dir() and (d / "PDFs").exists()]
        
        if not collections:
            print(f"Error: No valid collections found in {challenge_path}")
            sys.exit(1)
            
        success = True
        for collection in collections:
            print(f"\nProcessing collection: {collection.name}")
            success = process_challenge_collection(collection) and success
            
    if success:
        print("\nAll collections processed successfully.")
    else:
        print("\nThere were errors processing some collections.")
        sys.exit(1)

if __name__ == "__main__":
    main()