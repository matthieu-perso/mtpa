#!/usr/bin/env python3
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.pipeline import main as run_pipeline

if __name__ == "__main__":
    print("Starting dataset processing pipeline...")
    run_pipeline()
    print("Pipeline completed successfully!") 