#!/usr/bin/env python3
"""
Quick start script
For rapid testing of project components
"""

import sys
from pathlib import Path

# Add project root directory to path
sys.path.append(str(Path(__file__).parent))

def test_data_processing():
    """Test data processing module"""
    print("=== Test Data Processing Module ===")
    
    try:
        from src.data_processing import CrossLinguisticDataCollector, DataPreprocessor
        
        # Test data collection
        collector = CrossLinguisticDataCollector()
        collector.collect_parallel_corpus()
        collector.collect_grammar_patterns()
        
        print("✅ Data collector working normally")
        
        # Test data preprocessing
        preprocessor = DataPreprocessor()
        test_data = collector.data_sources['grammar_comparisons'][:1]
        processed = preprocessor.process_grammar_data(test_data)
        
        print("✅ Data preprocessor working normally")
        print(f"   Processed {len(processed)} data items")
        
    except Exception as e:
        print(f"❌ Data processing module test failed: {e}")


def test_training_setup():
    """Test training module setup"""
    print("\n=== Test Training Module Setup ===")
    
    try:
        from src.training import set_seed
        
        # Test utility functions
        set_seed(42)
        print("✅ Random seed setting normal")
        
        # Test memory estimation
        from src.training.utils import estimate_memory_usage
        memory_info = estimate_memory_usage(
            "microsoft/Phi-4-multimodal-instruct",
            batch_size=4,
            sequence_length=1024,
            use_lora=True
        )
        print("✅ Memory estimation function normal")
        print(f"   Estimated total memory: {memory_info['total_memory_gb']:.2f} GB")
        
    except Exception as e:
        print(f"❌ Training module test failed: {e}")


def test_evaluation():
    """Test evaluation module"""
    print("\n=== Test Evaluation Module ===")
    
    try:
        from src.evaluation import evaluate_model_outputs
        
        # Test evaluation metrics
        predictions = [
            "English and German present perfect tenses differ. English uses have/has + past participle, while German uses haben/sein + Partizip II."
        ]
        references = [
            "English present perfect uses have/has + past participle, German uses haben/sein + past participle."
        ]
        
        metrics = evaluate_model_outputs(predictions, references)
        print("✅ Evaluation metrics calculation normal")
        print(f"   BLEU score: {metrics.get('bleu_score', 0):.4f}")
        print(f"   Comparison coverage: {metrics.get('comparison_coverage', 0):.4f}")
        
    except Exception as e:
        print(f"❌ Evaluation module test failed: {e}")


def test_config_loading():
    """Test configuration file loading"""
    print("\n=== Test Configuration Files ===")
    
    try:
        import yaml
        
        # Test training configuration
        with open('config/training_config.yaml', 'r') as f:
            training_config = yaml.safe_load(f)
        print("✅ Training config file loaded normally")
        print(f"   Model: {training_config.get('model_name', 'Unknown')}")
        
        # Test model configuration
        with open('config/model_config.yaml', 'r') as f:
            model_config = yaml.safe_load(f)
        print("✅ Model config file loaded normally")
        print(f"   Using LoRA: {model_config.get('lora', {}).get('use_lora', False)}")
        
    except Exception as e:
        print(f"❌ Config file test failed: {e}")


def test_environment():
    """Test environment"""
    print("\n=== Check Environment ===")
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Check key dependencies
    dependencies = [
        'torch', 'transformers', 'datasets', 'peft', 'trl',
        'numpy', 'pandas', 'yaml', 'tqdm', 'sklearn'
    ]
    
    missing_deps = []
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✅ {dep}")
        except ImportError:
            print(f"❌ {dep} (not installed)")
            missing_deps.append(dep)
    
    if missing_deps:
        print(f"\nPlease install missing dependencies: pip install {' '.join(missing_deps)}")
    
    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
            print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("⚠️  GPU: No CUDA GPU detected")
    except:
        print("❌ Unable to check GPU status")


def show_usage_examples():
    """Show usage examples"""
    print("\n=== Usage Examples ===")
    
    print("""
🚀 Quick Start:

1. Prepare data:
   python scripts/prepare_data.py --output_dir ./data

2. Complete training pipeline:
   python scripts/train_full_pipeline.py --config config/training_config.yaml

3. SFT training only:
   python scripts/train_full_pipeline.py --config config/training_config.yaml --sft_only

4. Model evaluation:
   python scripts/evaluate_model.py --model ./models/final_model --benchmark ./data/benchmark.json

5. Interactive evaluation:
   python scripts/evaluate_model.py --model ./models/final_model --benchmark ./data/benchmark.json --interactive_eval

📁 Important directories:
   - config/: Configuration files
   - data/: Data storage
   - models/: Model storage
   - results/: Result output
   - scripts/: Executable scripts

📖 For more information see README.md
""")


def main():
    """Main function"""
    print("🤖 AI Tutor Cross-Linguistic Learning System - Quick Test")
    print("=" * 50)
    
    # Run various tests
    test_environment()
    test_config_loading()
    test_data_processing()
    test_training_setup()
    test_evaluation()
    
    # Show usage examples
    show_usage_examples()
    
    print("\n" + "=" * 50)
    print("✅ Quick testing completed!")
    print("If all tests pass, you can start using this training framework.")


if __name__ == "__main__":
    main()