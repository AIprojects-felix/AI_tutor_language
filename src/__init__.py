"""
AI Tutor Cross-Linguistic Learning System

English-German cross-linguistic AI tutoring system based on large language models
Built on microsoft/Phi-4-multimodal-instruct model

Main modules:
- data_processing: Data collection and preprocessing
- training: SFT and RLHF training modules
- evaluation: Model evaluation modules

Usage example:
    from src.training import CrossLinguisticSFTTrainer
    from src.evaluation import CrossLinguisticEvaluator
    
    # Train model
    trainer = CrossLinguisticSFTTrainer("config/training_config.yaml")
    trainer.train(train_dataset, eval_dataset)
    
    # Evaluate model
    evaluator = CrossLinguisticEvaluator("./models/trained_model")
    results = evaluator.evaluate_on_benchmark("./data/benchmark.json")
"""

__version__ = "1.0.0"
__author__ = "AI Tutor Team"
__email__ = "contact@ai-tutor.com"

# Import main components
from .data_processing import (
    CrossLinguisticDataCollector,
    DataPreprocessor,
    DatasetBuilder,
    build_complete_dataset
)

from .training import (
    CrossLinguisticSFTTrainer,
    CrossLinguisticRLHFTrainer,
    RewardModel,
    set_seed
)

from .evaluation import (
    CrossLinguisticEvaluator,
    CrossLinguisticMetrics,
    evaluate_model_outputs
)

__all__ = [
    # Data processing
    'CrossLinguisticDataCollector',
    'DataPreprocessor', 
    'DatasetBuilder',
    'build_complete_dataset',
    
    # Training
    'CrossLinguisticSFTTrainer',
    'CrossLinguisticRLHFTrainer',
    'RewardModel',
    'set_seed',
    
    # Evaluation
    'CrossLinguisticEvaluator',
    'CrossLinguisticMetrics',
    'evaluate_model_outputs'
]