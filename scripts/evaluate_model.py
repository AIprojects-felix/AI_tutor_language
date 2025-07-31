#!/usr/bin/env python3
"""
Model evaluation script
Comprehensive evaluation of trained models
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.evaluation import CrossLinguisticEvaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Model evaluation script")
    
    # Basic parameters
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--benchmark", type=str, required=True, help="Benchmark data path")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results", help="Results output directory")
    parser.add_argument("--config", type=str, help="Configuration file path")
    
    # Evaluation options
    parser.add_argument("--batch_size", type=int, default=4, help="Evaluation batch size")
    parser.add_argument("--max_samples", type=int, help="Maximum evaluation samples (for quick testing)")
    parser.add_argument("--save_outputs", action="store_true", help="Save model outputs")
    
    # Evaluation types
    parser.add_argument("--quick_eval", action="store_true", help="Quick evaluation (using fewer samples)")
    parser.add_argument("--capability_eval", action="store_true", help="Evaluation by capability type")
    parser.add_argument("--robustness_eval", action="store_true", help="Robustness evaluation")
    parser.add_argument("--interactive_eval", action="store_true", help="Interactive evaluation")
    
    # Generation parameters
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum generated tokens")
    
    # Other options
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    
    return parser.parse_args()


def setup_output_dir(output_dir: str, model_path: str) -> Path:
    """Setup output directory"""
    # Create directory using model name and timestamp
    model_name = Path(model_path).name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    eval_dir = Path(output_dir) / f"{model_name}_{timestamp}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Evaluation results will be saved to: {eval_dir}")
    return eval_dir


def load_and_filter_benchmark(benchmark_path: str, max_samples: int = None) -> list:
    """Load and filter benchmark data"""
    logger.info(f"Loading benchmark data: {benchmark_path}")
    
    with open(benchmark_path, 'r', encoding='utf-8') as f:
        benchmark_data = json.load(f)
    
    if max_samples and len(benchmark_data) > max_samples:
        import random
        random.shuffle(benchmark_data)
        benchmark_data = benchmark_data[:max_samples]
        logger.info(f"Using {max_samples} samples for quick evaluation")
    
    logger.info(f"Benchmark data: {len(benchmark_data)} samples")
    
    # Count task type distribution
    task_counts = {}
    for item in benchmark_data:
        task_type = item.get('type', 'unknown')
        task_counts[task_type] = task_counts.get(task_type, 0) + 1
    
    logger.info("Task type distribution:")
    for task_type, count in task_counts.items():
        logger.info(f"  {task_type}: {count}")
    
    return benchmark_data


def run_basic_evaluation(evaluator: CrossLinguisticEvaluator, 
                        benchmark_data: list,
                        output_dir: Path,
                        args) -> dict:
    """Run basic evaluation"""
    logger.info("=== Running Basic Evaluation ===")
    
    # Create temporary benchmark file
    temp_benchmark = output_dir / "temp_benchmark.json"
    with open(temp_benchmark, 'w', encoding='utf-8') as f:
        json.dump(benchmark_data, f, ensure_ascii=False, indent=2)
    
    # Execute evaluation
    results = evaluator.evaluate_on_benchmark(
        str(temp_benchmark),
        output_path=str(output_dir / "evaluation_results.json")
    )
    
    # Clean up temporary files
    temp_benchmark.unlink()
    
    return results


def run_capability_evaluation(evaluator: CrossLinguisticEvaluator,
                             benchmark_data: list,
                             output_dir: Path) -> dict:
    """Run capability classification evaluation"""
    logger.info("=== Running Capability Classification Evaluation ===")
    
    # Create temporary benchmark file
    temp_benchmark = output_dir / "temp_benchmark.json"
    with open(temp_benchmark, 'w', encoding='utf-8') as f:
        json.dump(benchmark_data, f, ensure_ascii=False, indent=2)
    
    # Evaluate by type
    type_results = evaluator.evaluate_capability_by_type(str(temp_benchmark))
    
    # Save results
    with open(output_dir / "capability_evaluation.json", 'w', encoding='utf-8') as f:
        json.dump(type_results, f, ensure_ascii=False, indent=2)
    
    # Clean up temporary files
    temp_benchmark.unlink()
    
    return type_results


def run_robustness_evaluation(evaluator: CrossLinguisticEvaluator,
                             benchmark_data: list,
                             output_dir: Path,
                             num_runs: int = 3) -> dict:
    """Run robustness evaluation"""
    logger.info(f"=== Running Robustness Evaluation ({num_runs} runs) ===")
    
    # Select subset of samples for robustness testing
    test_prompts = [item['prompt'] for item in benchmark_data[:10]]
    
    # Execute robustness evaluation
    robustness_results = evaluator.evaluate_robustness(test_prompts, num_runs)
    
    # Save results
    with open(output_dir / "robustness_evaluation.json", 'w', encoding='utf-8') as f:
        json.dump(robustness_results, f, ensure_ascii=False, indent=2)
    
    return robustness_results


def run_interactive_evaluation(evaluator: CrossLinguisticEvaluator):
    """Run interactive evaluation"""
    logger.info("=== Interactive Evaluation Mode ===")
    print("\nEntering interactive evaluation mode. Type 'quit' to exit.")
    
    while True:
        try:
            prompt = input("\nPlease enter test prompt > ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                break
            
            if not prompt:
                continue
            
            print(f"\nGenerating...")
            response = evaluator.generate_response(prompt)
            
            print(f"\nModel output:")
            print("-" * 50)
            print(response)
            print("-" * 50)
            
            # Simple scoring
            print(f"\nPlease rate this response (1-5, press Enter to skip): ", end="")
            try:
                score_input = input().strip()
                if score_input:
                    score = int(score_input)
                    print(f"Your rating: {score}/5")
            except ValueError:
                pass
            except KeyboardInterrupt:
                break
        
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nExiting interactive evaluation")


def create_evaluation_summary(results: dict, output_dir: Path, args) -> dict:
    """Create evaluation summary"""
    logger.info("Creating evaluation summary...")
    
    overall_metrics = results.get('overall_metrics', {})
    overall_score = overall_metrics.get('overall_score', 0)
    
    # Create summary report
    summary = {
        'evaluation_info': {
            'model': args.model,
            'benchmark': args.benchmark,
            'timestamp': datetime.now().isoformat(),
            'overall_score': overall_score
        },
        'metrics': overall_metrics,
        'performance_analysis': {},
        'strengths': [],
        'weaknesses': [],
        'recommendations': []
    }
    
    # Performance analysis
    if overall_score >= 0.8:
        summary['performance_analysis']['level'] = "Excellent"
        summary['performance_analysis']['description'] = "Model performs excellently and is ready for deployment"
    elif overall_score >= 0.6:
        summary['performance_analysis']['level'] = "Good"
        summary['performance_analysis']['description'] = "Model performs well but has room for improvement"
    else:
        summary['performance_analysis']['level'] = "Needs Improvement"
        summary['performance_analysis']['description'] = "Model needs further optimization"
    
    # Strengths and weaknesses analysis
    strengths = []
    weaknesses = []
    
    if overall_metrics.get('comparison_coverage', 0) > 0.7:
        strengths.append("Strong comparative analysis capability")
    else:
        weaknesses.append("Comparative analysis capability needs strengthening")
    
    if overall_metrics.get('explanation_quality', 0) > 0.7:
        strengths.append("High quality explanations")
    else:
        weaknesses.append("Explanation quality needs improvement")
    
    if overall_metrics.get('example_provision_rate', 0) > 0.7:
        strengths.append("Good at providing examples")
    else:
        weaknesses.append("Insufficient example provision")
    
    summary['strengths'] = strengths
    summary['weaknesses'] = weaknesses
    
    # Save summary
    with open(output_dir / "evaluation_summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # Generate text report
    report_text = f"""# Model Evaluation Report

## Basic Information
- Model: {args.model}
- Benchmark Data: {args.benchmark}
- Evaluation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overall Performance
- **Overall Score**: {overall_score:.4f}
- **Performance Level**: {summary['performance_analysis']['level']}
- **Assessment**: {summary['performance_analysis']['description']}

## Detailed Metrics
- BLEU Score: {overall_metrics.get('bleu_score', 0):.4f}
- ROUGE Score: {overall_metrics.get('rouge_score', 0):.4f}
- BERTScore: {overall_metrics.get('bert_score', 0):.4f}
- Comparison Coverage: {overall_metrics.get('comparison_coverage', 0):.4f}
- Explanation Quality: {overall_metrics.get('explanation_quality', 0):.4f}
- Example Provision Rate: {overall_metrics.get('example_provision_rate', 0):.4f}

## Strengths
{chr(10).join([f"- {s}" for s in strengths]) if strengths else "- None identified"}

## Weaknesses
{chr(10).join([f"- {w}" for w in weaknesses]) if weaknesses else "- None identified"}

## Recommendations
- Continue training if overall score < 0.6
- Consider fine-tuning specific capabilities based on weaknesses
- Deploy for testing if score >= 0.8

---
Generated by AI Tutor Evaluation System
"""
    
    with open(output_dir / "evaluation_report.md", 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    return summary


def main():
    """Main function"""
    args = parse_args()
    
    # Setup logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    
    try:
        # Setup output directory
        output_dir = setup_output_dir(args.output_dir, args.model)
        
        # Load evaluator
        logger.info(f"Loading model: {args.model}")
        evaluator = CrossLinguisticEvaluator(args.model)
        
        # Interactive evaluation mode
        if args.interactive_eval:
            run_interactive_evaluation(evaluator)
            return
        
        # Load benchmark data
        benchmark_data = load_and_filter_benchmark(args.benchmark, args.max_samples)
        
        if not benchmark_data:
            logger.error("No benchmark data loaded")
            return
        
        results = {}
        
        # Basic evaluation
        if not args.quick_eval or not (args.capability_eval or args.robustness_eval):
            basic_results = run_basic_evaluation(evaluator, benchmark_data, output_dir, args)
            results.update(basic_results)
        
        # Capability classification evaluation
        if args.capability_eval:
            capability_results = run_capability_evaluation(evaluator, benchmark_data, output_dir)
            results['capability_results'] = capability_results
        
        # Robustness evaluation
        if args.robustness_eval:
            robustness_results = run_robustness_evaluation(evaluator, benchmark_data, output_dir)
            results['robustness_results'] = robustness_results
        
        # Create evaluation summary
        if results:
            summary = create_evaluation_summary(results, output_dir, args)
            
            # Print main results
            overall_score = results.get('overall_metrics', {}).get('overall_score', 0)
            print(f"\n{'='*60}")
            print(f"Evaluation Complete")
            print(f"{'='*60}")
            print(f"Overall Score: {overall_score:.4f}")
            print(f"Performance Level: {summary['performance_analysis']['level']}")
            print(f"Results saved to: {output_dir}")
            print(f"{'='*60}")
        
        logger.info("Evaluation completed!")
        
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        if args.debug:
            raise


if __name__ == "__main__":
    main()
