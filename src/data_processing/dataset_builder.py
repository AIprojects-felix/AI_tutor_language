"""
Dataset building module
Build processed data into training, validation and test datasets
"""

import json
import random
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetBuilder:
    """Dataset builder"""
    
    def __init__(self, 
                 train_ratio: float = 0.8,
                 eval_ratio: float = 0.1,
                 test_ratio: float = 0.1,
                 seed: int = 42):
        """
        Args:
            train_ratio: Training set ratio
            eval_ratio: Validation set ratio  
            test_ratio: Test set ratio
            seed: Random seed
        """
        assert abs(train_ratio + eval_ratio + test_ratio - 1.0) < 1e-5, \
            "Dataset ratios must sum to 1"
            
        self.train_ratio = train_ratio
        self.eval_ratio = eval_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        
        random.seed(seed)
        
    def load_processed_data(self, data_paths: List[str]) -> List[Dict]:
        """Load all processed data files"""
        all_data = []
        
        for path in data_paths:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    all_data.extend(data)
                    logger.info(f"Loaded {len(data)} items from {path}")
            except Exception as e:
                logger.error(f"Failed to load {path}: {e}")
                
        logger.info(f"Total loaded {len(all_data)} data items")
        return all_data
    
    def split_data(self, data: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split data into train, validation and test sets"""
        
        # Group by type to ensure balanced distribution
        data_by_type = {}
        for item in data:
            dtype = item.get('type', 'unknown')
            if dtype not in data_by_type:
                data_by_type[dtype] = []
            data_by_type[dtype].append(item)
        
        train_data = []
        eval_data = []
        test_data = []
        
        # Split each data type separately
        for dtype, items in data_by_type.items():
            random.shuffle(items)
            
            # Calculate split points
            n_train = int(len(items) * self.train_ratio)
            n_eval = int(len(items) * self.eval_ratio)
            
            train_items = items[:n_train]
            eval_items = items[n_train:n_train + n_eval]
            test_items = items[n_train + n_eval:]
            
            train_data.extend(train_items)
            eval_data.extend(eval_items)
            test_data.extend(test_items)
            
            logger.info(f"{dtype}: train {len(train_items)}, "
                       f"eval {len(eval_items)}, test {len(test_items)}")
        
        # Shuffle again
        random.shuffle(train_data)
        random.shuffle(eval_data)
        random.shuffle(test_data)
        
        logger.info(f"Dataset split complete - train: {len(train_data)}, "
                   f"eval: {len(eval_data)}, test: {len(test_data)}")
        
        return train_data, eval_data, test_data
    
    def create_dataset_dict(self, 
                           train_data: List[Dict],
                           eval_data: List[Dict],
                           test_data: List[Dict]) -> DatasetDict:
        """Create HuggingFace dataset dictionary"""
        
        # Convert to DataFrame format
        def to_dataframe(data: List[Dict]) -> pd.DataFrame:
            records = []
            for item in data:
                record = {
                    'text': item['text'],
                    'type': item.get('type', 'unknown'),
                    'metadata': json.dumps(item.get('metadata', {}))
                }
                records.append(record)
            return pd.DataFrame(records)
        
        # Create datasets
        train_df = to_dataframe(train_data)
        eval_df = to_dataframe(eval_data)
        test_df = to_dataframe(test_data)
        
        # Convert to HuggingFace datasets
        dataset_dict = DatasetDict({
            'train': Dataset.from_pandas(train_df),
            'validation': Dataset.from_pandas(eval_df),
            'test': Dataset.from_pandas(test_df)
        })
        
        logger.info("Created HuggingFace dataset dictionary")
        return dataset_dict
    
    def add_instruction_wrapper(self, data: List[Dict], 
                               instruction_template: Optional[str] = None) -> List[Dict]:
        """Add instruction wrapper to data"""
        
        if instruction_template is None:
            instruction_template = "Please provide a detailed explanation for the following cross-linguistic learning question:\n\n{input}\n\nAnswer:"
        
        wrapped_data = []
        for item in data:
            wrapped_item = item.copy()
            wrapped_item['text'] = instruction_template.format(input=item['text'])
            wrapped_data.append(wrapped_item)
            
        logger.info(f"Added instruction wrapper to {len(wrapped_data)} items")
        return wrapped_data
    
    def balance_dataset(self, data: List[Dict], 
                       max_samples_per_type: Optional[int] = None) -> List[Dict]:
        """Balance dataset to ensure relatively equal data amounts per type"""
        
        # Group by type
        data_by_type = {}
        for item in data:
            dtype = item.get('type', 'unknown')
            if dtype not in data_by_type:
                data_by_type[dtype] = []
            data_by_type[dtype].append(item)
        
        # Determine max samples per type
        if max_samples_per_type is None:
            # Use the minimum count among all types
            max_samples_per_type = min(len(items) for items in data_by_type.values())
        
        balanced_data = []
        for dtype, items in data_by_type.items():
            # Randomly sample up to max_samples_per_type
            random.shuffle(items)
            selected_items = items[:max_samples_per_type]
            balanced_data.extend(selected_items)
            
            logger.info(f"Selected {len(selected_items)} samples for type {dtype}")
        
        random.shuffle(balanced_data)
        logger.info(f"Balanced dataset: {len(balanced_data)} total samples")
        
        return balanced_data
    
    def augment_data(self, data: List[Dict], augmentation_factor: float = 0.2) -> List[Dict]:
        """Data augmentation (simple synonym replacement, word order adjustment, etc.)"""
        
        augmented_data = data.copy()
        n_augment = int(len(data) * augmentation_factor)
        
        # Simple augmentation: randomly select samples for duplication with minor changes
        for _ in range(n_augment):
            original = random.choice(data)
            augmented = original.copy()
            
            # Simple text variation (this is a placeholder - in practice you'd use more sophisticated methods)
            text = augmented['text']
            # Add slight variations to avoid exact duplicates
            augmented['text'] = text + " (Variation)"
            augmented['metadata'] = augmented.get('metadata', {})
            augmented['metadata']['augmented'] = True
            
            augmented_data.append(augmented)
        
        logger.info(f"Data augmentation: added {n_augment} samples")
        return augmented_data
    
    def save_datasets(self, 
                     train_data: List[Dict],
                     eval_data: List[Dict],
                     test_data: List[Dict],
                     output_dir: str):
        """Save datasets to files"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON files
        datasets = {
            'train': train_data,
            'eval': eval_data,
            'test': test_data
        }
        
        for split_name, split_data in datasets.items():
            file_path = output_path / f"{split_name}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(split_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {split_name} dataset to {file_path}")
    
    def create_benchmark_dataset(self, test_data: List[Dict]) -> List[Dict]:
        """Create benchmark test dataset"""
        
        # Create standardized benchmark format
        benchmark_data = []
        for item in test_data:
            benchmark_item = {
                'id': item.get('id', f"benchmark_{len(benchmark_data)}"),
                'input': item['text'],
                'expected_output': item.get('expected_output', ''),
                'type': item.get('type', 'unknown'),
                'difficulty': item.get('difficulty', 'medium'),
                'metadata': item.get('metadata', {})
            }
            benchmark_data.append(benchmark_item)
        
        logger.info(f"Created benchmark dataset with {len(benchmark_data)} items")
        return benchmark_data
    
    def get_statistics(self, data: List[Dict]) -> Dict:
        """Get dataset statistics"""
        
        stats = {
            'total_samples': len(data),
            'type_distribution': {},
            'avg_length': 0,
            'max_length': 0,
            'min_length': float('inf')
        }
        
        total_length = 0
        for item in data:
            # Type distribution
            dtype = item.get('type', 'unknown')
            stats['type_distribution'][dtype] = stats['type_distribution'].get(dtype, 0) + 1
            
            # Length statistics
            text_length = len(item['text'])
            total_length += text_length
            stats['max_length'] = max(stats['max_length'], text_length)
            stats['min_length'] = min(stats['min_length'], text_length)
        
        stats['avg_length'] = total_length / len(data) if data else 0
        
        return stats


def build_complete_dataset(raw_data_dir: str, output_dir: str, config: Dict):
    """Build complete dataset pipeline"""
    
    from .data_collector_en import CrossLinguisticDataCollector
    from .preprocessor_en import DataPreprocessor
    
    # 1. Collect data
    logger.info("=== Starting data collection ===")
    collector = CrossLinguisticDataCollector()
    collector.collect_parallel_corpus()
    collector.collect_grammar_patterns()
    collector.collect_false_friends()
    collector.collect_common_errors()
    
    # 2. Preprocess data
    logger.info("=== Starting data preprocessing ===")
    preprocessor = DataPreprocessor()
    
    all_processed_data = []
    
    # Process different types of data
    if collector.data_sources['grammar_comparisons']:
        grammar_data = preprocessor.process_grammar_data(collector.data_sources['grammar_comparisons'])
        all_processed_data.extend(grammar_data)
        
    if collector.data_sources['vocabulary_pairs']:
        vocab_data = preprocessor.process_vocabulary_data(collector.data_sources['vocabulary_pairs'])
        all_processed_data.extend(vocab_data)
        
    if collector.data_sources['common_errors']:
        error_data = preprocessor.process_error_data(collector.data_sources['common_errors'])
        all_processed_data.extend(error_data)
        
    if collector.data_sources['parallel_texts']:
        parallel_data = preprocessor.process_parallel_data(collector.data_sources['parallel_texts'])
        all_processed_data.extend(parallel_data)
    
    # 3. Build datasets
    logger.info("=== Starting dataset building ===")
    builder = DatasetBuilder(
        train_ratio=config.get('train_ratio', 0.8),
        eval_ratio=config.get('eval_ratio', 0.1),
        test_ratio=config.get('test_ratio', 0.1),
        seed=config.get('seed', 42)
    )
    
    # Balance dataset
    if config.get('balance_dataset', True):
        all_processed_data = builder.balance_dataset(all_processed_data)
    
    # Data augmentation
    if config.get('augment_data', False):
        all_processed_data = builder.augment_data(
            all_processed_data, 
            augmentation_factor=config.get('augmentation_factor', 0.2)
        )
    
    # Split dataset
    train_data, eval_data, test_data = builder.split_data(all_processed_data)
    
    # Save datasets
    builder.save_datasets(train_data, eval_data, test_data, output_dir)
    
    # Create benchmark dataset
    benchmark_data = builder.create_benchmark_dataset(test_data)
    benchmark_path = Path(output_dir) / 'benchmark' / 'crosslinguistic_benchmark.json'
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    with open(benchmark_path, 'w', encoding='utf-8') as f:
        json.dump(benchmark_data, f, ensure_ascii=False, indent=2)
    
    # Print statistics
    logger.info("=== Dataset statistics ===")
    for split_name, data in [('Training set', train_data), ('Validation set', eval_data), ('Test set', test_data)]:
        stats = builder.get_statistics(data)
        logger.info(f"{split_name}: {stats['total_samples']} samples")
        logger.info(f"  Type distribution: {stats['type_distribution']}")
        logger.info(f"  Average length: {stats['avg_length']:.0f} characters")
    
    logger.info("Dataset building completed!")


if __name__ == "__main__":
    # Test dataset building
    config = {
        'train_ratio': 0.8,
        'eval_ratio': 0.1,
        'test_ratio': 0.1,
        'balance_dataset': True,
        'augment_data': False,
        'seed': 42
    }
    
    build_complete_dataset(
        raw_data_dir="./data/raw",
        output_dir="./data/processed",
        config=config
    )
