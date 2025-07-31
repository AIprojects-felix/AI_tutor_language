#!/usr/bin/env python3
"""
Data preparation script
Collect, process and build training datasets
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_processing import (
    CrossLinguisticDataCollector,
    DataPreprocessor, 
    DatasetBuilder
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Data preparation script")
    
    parser.add_argument("--output_dir", type=str, default="./data", help="Output directory")
    parser.add_argument("--external_data", type=str, nargs="+", help="External data files")
    
    # Data collection options
    parser.add_argument("--collect_parallel", action="store_true", help="Collect parallel corpus")
    parser.add_argument("--collect_grammar", action="store_true", help="Collect grammar patterns")
    parser.add_argument("--collect_vocab", action="store_true", help="Collect vocabulary data")
    parser.add_argument("--collect_errors", action="store_true", help="Collect error data")
    
    # Data processing options
    parser.add_argument("--balance_dataset", action="store_true", help="Balance dataset")
    parser.add_argument("--augment_data", action="store_true", help="Data augmentation")
    parser.add_argument("--augmentation_factor", type=float, default=1.5, help="Augmentation factor")
    
    # Dataset split
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Training set ratio")
    parser.add_argument("--eval_ratio", type=float, default=0.1, help="Evaluation set ratio")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Test set ratio")
    
    # Other options
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    return parser.parse_args()


def collect_data(args) -> dict:
    """Collect data"""
    logger.info("Starting data collection...")
    
    # Initialize data collector
    collector = CrossLinguisticDataCollector()
    
    # Collect various types of data
    try:
        # Parallel corpus
        if args.collect_parallel:
            collector.collect_parallel_corpus()
        
        # Grammar comparison patterns
        if args.collect_grammar:
            collector.collect_grammar_patterns()
        
        # False friends vocabulary
        if args.collect_vocab:
            collector.collect_false_friends()
        
        # Common errors
        if args.collect_errors:
            collector.collect_common_errors()
            
    except Exception as e:
        logger.error(f"Error during data collection: {e}")
        
    # Load external data
    if args.external_data:
        for external_file in args.external_data:
            if not os.path.exists(external_file):
                logger.warning(f"External data file does not exist: {external_file}")
                continue
            
            try:
                # Infer data type from filename
                if 'grammar' in external_file.lower():
                    data_type = 'grammar_comparisons'
                elif 'vocab' in external_file.lower() or 'false' in external_file.lower():
                    data_type = 'false_friends'
                elif 'error' in external_file.lower():
                    data_type = 'common_errors'
                elif 'parallel' in external_file.lower():
                    data_type = 'parallel_texts'
                else:
                    data_type = 'parallel_texts'  # Default type
                
                collector.load_external_data(external_file, data_type)
            except Exception as e:
                logger.error(f"Error loading external data {external_file}: {e}")
    
    # Save collected data
    raw_data_dir = Path(args.output_dir) / "raw"
    collector.save_data(str(raw_data_dir))
    
    # Print statistics
    stats = collector.get_statistics()
    logger.info("Data collection statistics:")
    for data_type, info in stats.items():
        logger.info(f"  {data_type}: {info['count']} items")
    
    return collector.data_sources


def preprocess_data(raw_data: dict, args) -> list:
    """Preprocess data"""
    logger.info("Starting data preprocessing...")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(
        max_length=1024,
        min_length=10,
        remove_duplicates=True,
        normalize_text=True
    )
    
    # Process various types of data
    processed_data = []
    
    if 'grammar_comparisons' in raw_data and raw_data['grammar_comparisons']:
        logger.info("Processing grammar comparison data...")
        grammar_data = preprocessor.process_grammar_data(
            raw_data['grammar_comparisons']
        )
        processed_data.extend(grammar_data)
        logger.info(f"Processed {len(grammar_data)} grammar data items")
    
    if 'false_friends' in raw_data and raw_data['false_friends']:
        logger.info("Processing vocabulary comparison data...")
        vocab_data = preprocessor.process_vocabulary_data(
            raw_data['false_friends']
        )
        processed_data.extend(vocab_data)
        logger.info(f"Processed {len(vocab_data)} vocabulary data items")
    
    if 'common_errors' in raw_data and raw_data['common_errors']:
        logger.info("Processing error data...")
        error_data = preprocessor.process_error_data(
            raw_data['common_errors']
        )
        processed_data.extend(error_data)
        logger.info(f"Processed {len(error_data)} error data items")
    
    if 'parallel_texts' in raw_data and raw_data['parallel_texts']:
        logger.info("Processing parallel corpus...")
        parallel_data = preprocessor.process_parallel_data(
            raw_data['parallel_texts']
        )
        processed_data.extend(parallel_data)
        logger.info(f"Processed {len(parallel_data)} parallel corpus items")
    
    # Format training data
    formatted_data = preprocessor.format_training_data(processed_data)
    
    # Save preprocessed data
    processed_dir = Path(args.output_dir) / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    preprocessor.save_processed_data(formatted_data, str(processed_dir / "preprocessed.json"))
    
    logger.info(f"Preprocessing complete, total {len(formatted_data)} data items")
    
    return formatted_data


def build_datasets(processed_data: list, args) -> dict:
    """Build datasets"""
    logger.info("Starting dataset construction...")
    
    # Initialize dataset builder
    builder = DatasetBuilder(
        train_ratio=args.train_ratio,
        eval_ratio=args.eval_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    # Data balancing
    if args.balance_dataset:
        logger.info("Balancing dataset...")
        processed_data = builder.balance_dataset(processed_data)
    
    # Data augmentation
    if args.augment_data:
        logger.info(f"Data augmentation (factor: {args.augmentation_factor})...")
        processed_data = builder.augment_data(
            processed_data, 
            augmentation_factor=args.augmentation_factor
        )
    
    # Split dataset
    train_data, eval_data, test_data = builder.split_dataset(processed_data)
    
    # Save datasets
    output_dir = Path(args.output_dir) / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    builder.save_datasets(train_data, eval_data, test_data, str(output_dir))
    
    # Create benchmark test set
    benchmark_data = builder.create_benchmark_dataset(
        test_data, 
        include_capability_tests=True,
        include_robustness_tests=True
    )
    
    benchmark_file = output_dir / "benchmark" / "crosslinguistic_benchmark.json"
    benchmark_file.parent.mkdir(parents=True, exist_ok=True)
    builder.save_benchmark(benchmark_data, str(benchmark_file))
    logger.info(f"Benchmark test set saved to: {benchmark_file}")
    
    # Print statistics
    logger.info("Dataset statistics:")
    for split_name, data in [('Training set', train_data), ('Validation set', eval_data), ('Test set', test_data)]:
        stats = builder.get_dataset_statistics(data)
        logger.info(f"  {split_name}: {stats['total_samples']} items")
        if args.verbose:
            logger.info(f"    Type distribution: {stats['type_distribution']}")
    
    return {
        'train': train_data,
        'eval': eval_data, 
        'test': test_data,
        'benchmark': benchmark_data
    }


def main():
    """Main function"""
    args = parse_args()
    
    # Setup logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Set random seed
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    try:
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Data preparation started, output directory: {output_dir}")
        
        # Step 1: Collect data
        raw_data = collect_data(args)
        
        # Step 2: Preprocess data
        processed_data = preprocess_data(raw_data, args)
        
        # Step 3: Build datasets
        datasets = build_datasets(processed_data, args)
        
        # Summary
        total_samples = sum(len(datasets[split]) for split in ['train', 'eval', 'test'])
        logger.info(f"Data preparation complete!")
        logger.info(f"Total samples: {total_samples}")
        logger.info(f"Training set: {len(datasets['train'])}")
        logger.info(f"Validation set: {len(datasets['eval'])}")
        logger.info(f"Test set: {len(datasets['test'])}")
        logger.info(f"Benchmark set: {len(datasets['benchmark'])}")
        logger.info(f"Results saved to: {output_dir}")
        
    except KeyboardInterrupt:
        logger.info("Data preparation interrupted by user")
    except Exception as e:
        logger.error(f"Data preparation failed: {e}")
        raise


if __name__ == "__main__":
    main()
