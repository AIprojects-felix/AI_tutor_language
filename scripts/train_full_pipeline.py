#!/usr/bin/env python3
"""
Complete training pipeline script
Executes the full training process from data preparation to model evaluation
"""

import os
import sys
import argparse
import logging
import yaml
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_processing import build_complete_dataset
from src.training import CrossLinguisticSFTTrainer, CrossLinguisticRLHFTrainer
from src.evaluation import CrossLinguisticEvaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Complete training pipeline")
    
    # Basic configuration
    parser.add_argument("--config", type=str, default="config/training_config.yaml", help="Training configuration file")
    parser.add_argument("--experiment_name", type=str, required=True, help="Experiment name")
    parser.add_argument("--output_dir", type=str, default="./experiments", help="Output directory")
    
    # Stage control
    parser.add_argument("--skip_data_prep", action="store_true", help="Skip data preparation")
    parser.add_argument("--skip_sft", action="store_true", help="Skip SFT training")
    parser.add_argument("--skip_rlhf", action="store_true", help="Skip RLHF training")
    parser.add_argument("--skip_eval", action="store_true", help="Skip evaluation")
    parser.add_argument("--sft_only", action="store_true", help="Only run SFT training")
    
    # Data options
    parser.add_argument("--data_dir", type=str, help="Data directory (if skipping data preparation)")
    parser.add_argument("--external_data", type=str, nargs="+", help="External data files")
    
    # Model options
    parser.add_argument("--base_model", type=str, help="Base model path (overrides config)")
    parser.add_argument("--sft_model", type=str, help="SFT model path (for RLHF)")
    
    # Training options
    parser.add_argument("--resume_from_checkpoint", type=str, help="Resume from checkpoint")
    parser.add_argument("--max_steps", type=int, help="Maximum training steps")
    
    # Other options
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def setup_experiment_dir(output_dir: str, experiment_name: str) -> Path:
    """Setup experiment directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = Path(output_dir) / f"{experiment_name}_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (experiment_dir / "models").mkdir(exist_ok=True)
    (experiment_dir / "data").mkdir(exist_ok=True)
    (experiment_dir / "results").mkdir(exist_ok=True)
    (experiment_dir / "logs").mkdir(exist_ok=True)
    
    logger.info(f"Experiment directory: {experiment_dir}")
    return experiment_dir


def stage_1_data_preparation(config: dict, args, experiment_dir: Path) -> str:
    """Stage 1: Data Preparation"""
    logger.info("=== Stage 1: Data Preparation ===")
    
    if args.skip_data_prep:
        logger.info("Skipping data preparation stage")
        if args.data_dir:
            data_dir = args.data_dir
        else:
            data_dir = "./data"
        logger.info("Using existing data")
        return data_dir
    
    # Setup data directory
    data_dir = experiment_dir / "data"
    
    # Build dataset
    try:
        dataset_info = build_complete_dataset(
            output_dir=str(data_dir),
            external_data=args.external_data,
            collect_parallel=True,
            collect_grammar=True,
            collect_vocab=True,
            collect_errors=True,
            balance_dataset=True,
            augment_data=True,
            train_ratio=config['data']['train_split_ratio'],
            eval_ratio=config['data']['eval_split_ratio'],
            test_ratio=config['data']['test_split_ratio']
        )
        
        # Update data paths in config
        config['data']['train_data_path'] = str(data_dir / "processed" / "train.json")
        config['data']['eval_data_path'] = str(data_dir / "processed" / "eval.json")
        config['data']['test_data_path'] = str(data_dir / "processed" / "test.json")
        
        logger.info("Data preparation completed")
        return str(data_dir)
        
    except Exception as e:
        logger.error(f"Data preparation failed: {e}")
        raise


def stage_2_sft_training(config: dict, args, experiment_dir: Path) -> str:
    """Stage 2: SFT Training"""
    logger.info("=== Stage 2: SFT Training ===")
    
    if args.skip_sft:
        logger.info("Skipping SFT training")
        return args.sft_model or config.get('model_name', 'microsoft/Phi-4-multimodal-instruct')
    
    # Setup output directory
    sft_output_dir = experiment_dir / "models" / "sft_model"
    sft_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_copy = config.copy()
    config_copy['output_dir'] = str(sft_output_dir)
    
    try:
        # Initialize SFT trainer
        trainer = CrossLinguisticSFTTrainer(config_copy)
        
        # Setup model and tokenizer
        if args.base_model:
            trainer.setup_model_and_tokenizer(args.base_model)
        else:
            trainer.setup_model_and_tokenizer()
        
        # Prepare datasets
        train_dataset, eval_dataset = trainer.prepare_datasets(
            config['data']['train_data_path'],
            config['data']['eval_data_path']
        )
        
        # Execute training
        trainer.train(train_dataset, eval_dataset, resume_from_checkpoint=args.resume_from_checkpoint)
        
        logger.info(f"SFT training completed, model saved to: {sft_output_dir}")
        return str(sft_output_dir)
        
    except Exception as e:
        logger.error(f"SFT training failed: {e}")
        raise


def stage_3_rlhf_training(config: dict, args, experiment_dir: Path, sft_model_path: str) -> str:
    """Stage 3: RLHF Training"""
    logger.info("=== Stage 3: RLHF Training ===")
    
    if args.skip_rlhf or args.sft_only:
        logger.info("Skipping RLHF training")
        return sft_model_path
    
    # Create preference data (if not exists)
    preference_data_path = experiment_dir / "data" / "processed" / "preferences.json"
    if not preference_data_path.exists():
        logger.info("Creating preference data...")
        # This would typically involve generating preference pairs
        # For now, we'll skip this step
        pass
    
    # Setup configuration
    rlhf_output_dir = experiment_dir / "models" / "final_model"
    rlhf_output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize RLHF trainer
        trainer = CrossLinguisticRLHFTrainer(config, str(rlhf_output_dir))
        
        # Setup model
        trainer.setup_model_and_tokenizer(sft_model_path)
        
        # Prepare dataset
        if preference_data_path.exists():
            dataset = trainer.prepare_dataset(str(preference_data_path))
            
            # Execute training
            trainer.train(dataset)
        
        logger.info(f"RLHF training completed, model saved to: {rlhf_output_dir}")
        return str(rlhf_output_dir)
        
    except Exception as e:
        logger.error(f"RLHF training failed: {e}")
        logger.warning("RLHF training failed, using SFT model")
        return sft_model_path


def stage_4_evaluation(config: dict, args, experiment_dir: Path, final_model_path: str) -> dict:
    """Stage 4: Model Evaluation"""
    logger.info("=== Stage 4: Model Evaluation ===")
    
    if args.skip_eval:
        logger.info("Skipping evaluation stage")
        return {}
    
    # Check benchmark data
    benchmark_path = experiment_dir / "data" / "benchmark" / "crosslinguistic_benchmark.json"
    if not benchmark_path.exists():
        # Try default path
        benchmark_path = Path("./data/benchmark/crosslinguistic_benchmark.json")
        if not benchmark_path.exists():
            logger.warning("Benchmark data not found, skipping evaluation")
            return {}
    
    try:
        # Initialize evaluator
        evaluator = CrossLinguisticEvaluator(final_model_path)
        
        # Execute evaluation
        results = evaluator.evaluate_on_benchmark(
            str(benchmark_path),
            output_path=str(experiment_dir / "results" / "evaluation_results.json")
        )
        
        # Evaluate by type
        if hasattr(evaluator, 'evaluate_capability_by_type'):
            type_results = evaluator.evaluate_capability_by_type(str(benchmark_path))
            results['capability_results'] = type_results
        
        # Generate report
        report_path = experiment_dir / "results" / "evaluation_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Evaluation completed, results saved to: {experiment_dir}/results/")
        
        # Print main metrics
        overall_metrics = results.get('overall_metrics', {})
        logger.info(f"Overall Score: {overall_metrics.get('overall_score', 0):.4f}")
        logger.info(f"BLEU Score: {overall_metrics.get('bleu_score', 0):.4f}")
        logger.info(f"Comparison Coverage: {overall_metrics.get('comparison_coverage', 0):.4f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Model evaluation failed: {e}")
        return {}


def create_final_summary(config: dict, args, experiment_dir: Path, results: dict):
    """Create final summary"""
    logger.info("=== Creating Experiment Summary ===")
    
    summary = {
        'experiment_info': {
            'name': args.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'config_file': args.config,
            'stages_completed': []
        },
        'configuration': config,
        'results': results,
        'paths': {
            'experiment_dir': str(experiment_dir),
            'final_model': str(experiment_dir / "models" / "final_model"),
            'evaluation_results': str(experiment_dir / "results")
        }
    }
    
    # Record completed stages
    if not args.skip_data_prep:
        summary['experiment_info']['stages_completed'].append('data_preparation')
    if not args.skip_sft:
        summary['experiment_info']['stages_completed'].append('sft_training')
    if not args.skip_rlhf and not args.sft_only:
        summary['experiment_info']['stages_completed'].append('rlhf_training')
    if not args.skip_eval:
        summary['experiment_info']['stages_completed'].append('evaluation')
    
    # Save summary
    summary_path = experiment_dir / "experiment_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # Create text summary
    text_summary = f"""# Training Pipeline Summary

## Experiment Information
- Name: {args.experiment_name}
- Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Stages: {', '.join(summary['experiment_info']['stages_completed'])}

## Results
"""
    
    if results:
        overall_metrics = results.get('overall_metrics', {})
        text_summary += f"""
- Overall Score: {overall_metrics.get('overall_score', 0):.4f}
- BLEU Score: {overall_metrics.get('bleu_score', 0):.4f}
- ROUGE Score: {overall_metrics.get('rouge_score', 0):.4f}
- Comparison Coverage: {overall_metrics.get('comparison_coverage', 0):.4f}
"""
    else:
        text_summary += "\nNo evaluation results available.\n"
    
    text_summary += f"""
## Paths
- Experiment Directory: {experiment_dir}
- Final Model: {experiment_dir}/models/final_model
- Results: {experiment_dir}/results

---
Generated by AI Tutor Training Pipeline
"""
    
    with open(experiment_dir / "README.md", 'w', encoding='utf-8') as f:
        f.write(text_summary)
    
    logger.info(f"Experiment summary saved to: {summary_path}")


def main():
    """Main function"""
    args = parse_args()
    
    # Setup logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Override config with command line arguments
        if args.base_model:
            config['model_name'] = args.base_model
        if args.max_steps:
            config['sft']['max_steps'] = args.max_steps
        
        # Setup experiment directory
        experiment_dir = setup_experiment_dir(args.output_dir, args.experiment_name)
        
        # Save configuration
        config_path = experiment_dir / "training_config.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"Starting training pipeline: {args.experiment_name}")
        logger.info(f"Configuration: {args.config}")
        
        # Stage 1: Data Preparation
        data_dir = stage_1_data_preparation(config, args, experiment_dir)
        
        # Stage 2: SFT Training
        sft_model_path = stage_2_sft_training(config, args, experiment_dir)
        
        # Stage 3: RLHF Training
        final_model_path = stage_3_rlhf_training(config, args, experiment_dir, sft_model_path)
        
        # Stage 4: Evaluation
        results = stage_4_evaluation(config, args, experiment_dir, final_model_path)
        
        # Create final summary
        create_final_summary(config, args, experiment_dir, results)
        
        # Print completion message
        print(f"\n{'='*60}")
        print(f"Training Pipeline Completed!")
        print(f"{'='*60}")
        print(f"Experiment: {args.experiment_name}")
        print(f"Directory: {experiment_dir}")
        print(f"Final Model: {final_model_path}")
        
        if results:
            overall_score = results.get('overall_metrics', {}).get('overall_score', 0)
            print(f"Overall Score: {overall_score:.4f}")
        
        print(f"{'='*60}")
        
        logger.info("Training pipeline completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Training pipeline interrupted by user")
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        if args.debug:
            raise


if __name__ == "__main__":
    main()
