"""
Model Evaluator
Execute complete model evaluation pipeline
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Optional, Tuple, Any
import json
import yaml
from pathlib import Path
import logging
from tqdm import tqdm
import time
from datetime import datetime

from .metrics import evaluate_model_outputs, CrossLinguisticMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CrossLinguisticEvaluator:
    """Cross-linguistic learning model evaluator"""
    
    def __init__(self, model_path: str, config_path: Optional[str] = None):
        """
        Args:
            model_path: Model path
            config_path: Configuration file path
        """
        self.model_path = model_path
        self.config = self._load_config(config_path) if config_path else {}
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        self.metrics_calculator = CrossLinguisticMetrics()
        
        # Load model
        self._load_model()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load configuration file: {e}")
            return {}
    
    def _load_model(self):
        """Load model and tokenizer"""
        logger.info(f"Loading model: {self.model_path}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            self.model.eval()
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise
    
    def generate_response(self, prompt: str, **generation_kwargs) -> str:
        """Generate single response"""
        # Default generation parameters
        default_kwargs = {
            'max_new_tokens': 256,
            'temperature': 0.7,
            'top_p': 0.9,
            'do_sample': True,
            'repetition_penalty': 1.1
        }
        default_kwargs.update(generation_kwargs)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **default_kwargs,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode (only return newly generated part)
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def generate_responses_batch(self, prompts: List[str], 
                                batch_size: int = 4,
                                **generation_kwargs) -> List[str]:
        """Generate responses in batch"""
        responses = []
        
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating responses"):
            batch_prompts = prompts[i:i + batch_size]
            batch_responses = []
            
            for prompt in batch_prompts:
                try:
                    response = self.generate_response(prompt, **generation_kwargs)
                    batch_responses.append(response)
                except Exception as e:
                    logger.warning(f"Failed to generate response for prompt: {e}")
                    batch_responses.append("")
            
            responses.extend(batch_responses)
        
        return responses
    
    def evaluate_on_benchmark(self, benchmark_path: str, 
                             output_path: Optional[str] = None) -> Dict[str, Any]:
        """Evaluate on benchmark dataset"""
        logger.info(f"Loading benchmark data: {benchmark_path}")
        
        # Load benchmark data
        with open(benchmark_path, 'r', encoding='utf-8') as f:
            benchmark_data = json.load(f)
        
        logger.info(f"Benchmark contains {len(benchmark_data)} samples")
        
        # Extract prompts and references
        prompts = [item['prompt'] for item in benchmark_data]
        references = [item.get('expected', item.get('reference', '')) for item in benchmark_data]
        
        # Generate responses
        start_time = time.time()
        logger.info("Generating model responses...")
        
        predictions = self.generate_responses_batch(prompts)
        
        generation_time = time.time() - start_time
        logger.info(f"Generation completed in {generation_time:.2f} seconds")
        
        # Evaluate
        logger.info("Computing evaluation metrics...")
        metrics = evaluate_model_outputs(predictions, references)
        
        # Add performance metrics
        metrics['generation_time'] = generation_time
        metrics['avg_time_per_sample'] = generation_time / len(prompts)
        metrics['throughput'] = len(prompts) / generation_time
        
        # Prepare results
        results = {
            'benchmark_path': benchmark_path,
            'num_samples': len(benchmark_data),
            'generation_time': generation_time,
            'overall_metrics': metrics,
            'predictions': predictions,
            'references': references,
            'prompts': prompts,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save results
        if output_path:
            self._save_evaluation_results(results, output_path)
        
        logger.info("Evaluation completed")
        logger.info(f"Overall score: {metrics.get('overall_score', 0):.4f}")
        
        return results
    
    def evaluate_capability_by_type(self, benchmark_path: str) -> Dict[str, Any]:
        """Evaluate model by capability type"""
        logger.info("Evaluating by capability type...")
        
        # Load benchmark data
        with open(benchmark_path, 'r', encoding='utf-8') as f:
            benchmark_data = json.load(f)
        
        # Group by type
        type_groups = {}
        for item in benchmark_data:
            task_type = item.get('type', 'unknown')
            if task_type not in type_groups:
                type_groups[task_type] = []
            type_groups[task_type].append(item)
        
        # Evaluate each type
        type_results = {}
        for task_type, items in type_groups.items():
            logger.info(f"Evaluating {task_type}: {len(items)} samples")
            
            prompts = [item['prompt'] for item in items]
            references = [item.get('expected', item.get('reference', '')) for item in items]
            
            predictions = self.generate_responses_batch(prompts)
            metrics = evaluate_model_outputs(predictions, references)
            
            type_results[task_type] = {
                'num_samples': len(items),
                'metrics': metrics,
                'predictions': predictions,
                'references': references
            }
        
        return type_results
    
    def evaluate_robustness(self, test_prompts: List[str], 
                           num_runs: int = 3) -> Dict[str, Any]:
        """Evaluate model robustness (stability across multiple runs)"""
        logger.info(f"Evaluating robustness with {num_runs} runs...")
        
        all_outputs = []
        all_metrics = []
        
        for run in range(num_runs):
            logger.info(f"Run {run + 1}/{num_runs}")
            
            # Generate responses with different random seeds
            outputs = []
            for prompt in test_prompts:
                # Use different temperature for variation
                temp = 0.7 + (run * 0.1)
                response = self.generate_response(
                    prompt, 
                    temperature=temp,
                    do_sample=True
                )
                outputs.append(response)
            
            all_outputs.append(outputs)
            
            # Compute metrics for this run (using first run as reference)
            if run > 0:
                metrics = evaluate_model_outputs(outputs, all_outputs[0])
                all_metrics.append(metrics)
        
        # Analyze consistency
        consistency_analysis = self._analyze_output_consistency(all_outputs)
        
        # Analyze metrics stability
        if all_metrics:
            stability_analysis = self._analyze_metrics_stability(all_metrics)
        else:
            stability_analysis = {}
        
        return {
            'num_runs': num_runs,
            'test_prompts': test_prompts,
            'all_outputs': all_outputs,
            'consistency_analysis': consistency_analysis,
            'stability_analysis': stability_analysis
        }
    
    def _analyze_output_consistency(self, all_outputs: List[List[str]]) -> Dict[str, float]:
        """Analyze output consistency"""
        if len(all_outputs) < 2:
            return {}
        
        similarities = []
        
        # Compare each pair of runs
        for i in range(len(all_outputs)):
            for j in range(i + 1, len(all_outputs)):
                for k in range(len(all_outputs[i])):
                    sim = self._compute_text_similarity(
                        all_outputs[i][k], 
                        all_outputs[j][k]
                    )
                    similarities.append(sim)
        
        return {
            'mean_similarity': np.mean(similarities),
            'std_similarity': np.std(similarities),
            'min_similarity': np.min(similarities),
            'max_similarity': np.max(similarities)
        }
    
    def _compute_text_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two texts"""
        # Simple word overlap similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _analyze_metrics_stability(self, all_metrics: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Analyze metrics stability"""
        stability_analysis = {}
        
        # Get all metric keys
        metric_keys = set()
        for metrics in all_metrics:
            metric_keys.update(metrics.keys())
        
        # Analyze each metric
        for key in metric_keys:
            values = [metrics.get(key, 0) for metrics in all_metrics]
            
            stability_analysis[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'coefficient_of_variation': np.std(values) / np.mean(values) if np.mean(values) > 0 else 0
            }
        
        return stability_analysis
    
    def _save_evaluation_results(self, results: Dict[str, Any], output_path: str):
        """Save evaluation results"""
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main results
        with open(output_path, 'w', encoding='utf-8') as f:
            # Remove large arrays for JSON serialization
            results_copy = results.copy()
            if 'predictions' in results_copy and len(results_copy['predictions']) > 100:
                results_copy['predictions'] = results_copy['predictions'][:100]  # Save only first 100
            if 'references' in results_copy and len(results_copy['references']) > 100:
                results_copy['references'] = results_copy['references'][:100]
            
            json.dump(results_copy, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Results saved to: {output_path}")
    
    def generate_evaluation_report(self, results: Dict[str, Any]) -> str:
        """Generate evaluation report"""
        overall_metrics = results.get('overall_metrics', {})
        
        report = f"""# Model Evaluation Report

## Basic Information
- Evaluation Time: {results.get('timestamp', 'Unknown')}
- Benchmark: {results.get('benchmark_path', 'Unknown')}
- Number of Samples: {results.get('num_samples', 0)}
- Generation Time: {results.get('generation_time', 0):.2f} seconds

## Overall Performance
- **Overall Score**: {overall_metrics.get('overall_score', 0):.4f}

### Basic Metrics
- BLEU Score: {overall_metrics.get('bleu_score', 0):.4f}
- ROUGE Score: {overall_metrics.get('rouge_score', 0):.4f}
- BERTScore: {overall_metrics.get('bert_score', 0):.4f}

### Content Quality Metrics
- Comparison Coverage: {overall_metrics.get('comparison_coverage', 0):.4f}
- Explanation Quality: {overall_metrics.get('explanation_quality', 0):.4f}
- Example Provision Rate: {overall_metrics.get('example_provision', 0):.4f}
- Response Completeness: {overall_metrics.get('response_completeness', 0):.4f}

### Task-Specific Performance
- Grammar Comparison Accuracy: {overall_metrics.get('grammar_comparison_rule_explanation_quality', 0):.4f}
- Vocabulary Analysis Accuracy: {overall_metrics.get('vocabulary_comparison_false_friend_identification', 0):.4f}
- Error Feedback Quality: {overall_metrics.get('error_feedback_correction_quality', 0):.4f}

### Performance Metrics
- Total Generation Time: {overall_metrics.get('generation_time', 0):.2f} seconds
- Average Time per Sample: {overall_metrics.get('avg_time_per_sample', 0):.3f} seconds
- Throughput: {overall_metrics.get('throughput', 0):.2f} samples/second

### Linguistic Metrics
- Average Response Length: {overall_metrics.get('avg_response_length', 0):.0f} characters
- Vocabulary Diversity: {overall_metrics.get('vocabulary_diversity', 0):.4f}
- Average Sentence Length: {overall_metrics.get('avg_sentence_length', 0):.1f} words

## Overall Score
**{overall_metrics.get('overall_score', 0):.4f}**

## Recommendations
"""
        
        # Add recommendations
        overall_score = overall_metrics.get('overall_score', 0)
        if overall_score >= 0.8:
            report += "- Model performs excellently and can be considered for deployment\n"
        elif overall_score >= 0.6:
            report += "- Model performs well but still has room for improvement\n"
        else:
            report += "- Model needs further optimization\n"
        
        # Specific recommendations
        if overall_metrics.get('comparison_coverage', 0) < 0.7:
            report += "- Recommend strengthening comparative analysis capability training\n"
        
        if overall_metrics.get('explanation_quality', 0) < 0.7:
            report += "- Recommend improving explanation quality\n"
        
        if overall_metrics.get('example_provision', 0) < 0.5:
            report += "- Recommend increasing example provision training\n"
        
        return report


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Model evaluation script")
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--benchmark", type=str, required=True, help="Benchmark data path")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--output", type=str, help="Results output path")
    parser.add_argument("--report", type=str, help="Report output path")
    parser.add_argument("--robustness", action="store_true", help="Perform robustness testing")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = CrossLinguisticEvaluator(args.model, args.config)
    
    # Benchmark evaluation
    results = evaluator.evaluate_on_benchmark(args.benchmark, args.output)
    
    # Evaluate by type
    type_results = evaluator.evaluate_capability_by_type(args.benchmark)
    results['results_by_type'] = type_results
    
    # Robustness testing
    if args.robustness:
        with open(args.benchmark, 'r', encoding='utf-8') as f:
            benchmark_data = json.load(f)
        test_prompts = [item['prompt'] for item in benchmark_data[:10]]  # Take first 10
        
        robustness_results = evaluator.evaluate_robustness(test_prompts)
        results['robustness_results'] = robustness_results
    
    # Generate report
    if args.report:
        report = evaluator.generate_evaluation_report(results)
        with open(args.report, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"Evaluation report saved to: {args.report}")
    
    # Print main results
    overall_metrics = results['overall_metrics']
    print(f"\n=== Evaluation Results ===")
    print(f"Overall Score: {overall_metrics.get('overall_score', 0):.4f}")
    print(f"BLEU Score: {overall_metrics.get('bleu_score', 0):.4f}")
    print(f"Comparison Coverage: {overall_metrics.get('comparison_coverage', 0):.4f}")
    print(f"Explanation Quality: {overall_metrics.get('explanation_quality', 0):.4f}")
    print(f"Average Generation Time: {overall_metrics.get('avg_time_per_sample', 0):.3f} seconds/sample")


if __name__ == "__main__":
    main()
