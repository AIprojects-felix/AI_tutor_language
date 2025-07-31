"""
Evaluation metrics module
Defines calculation methods for various evaluation metrics
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
import re
from collections import Counter
import logging
from sklearn.metrics import accuracy_score, f1_score
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import bert_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class CrossLinguisticMetrics:
    """Cross-linguistic learning evaluation metrics"""
    
    def __init__(self):
        # Initialize evaluators
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothie = SmoothingFunction().method4
        
        # Keyword definitions
        self.comparison_keywords = [
            'compare', 'difference', 'similar', 'contrast', 'differ',
            'versus', 'between', 'distinguish', 'comparison'
        ]
        
        self.explanation_keywords = [
            'because', 'therefore', 'since', 'thus', 'hence',
            'reason', 'explanation', 'due to', 'result', 'cause'
        ]
        
        self.example_keywords = [
            '例如', '比如', '举例', '比方说',
            'for example', 'for instance', 'such as', 'e.g.'
        ]
    
    def compute_all_metrics(self, predictions: List[str], 
                           references: List[str],
                           task_types: List[str] = None) -> Dict[str, float]:
        """Compute all evaluation metrics
        
        Args:
            predictions: Model predictions
            references: Reference answers
            task_types: Task type list
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics.update(self.compute_basic_metrics(predictions, references))
        
        # Content quality metrics
        metrics.update(self.compute_content_metrics(predictions, references))
        
        # Task-specific metrics
        if task_types:
            metrics.update(self.compute_task_specific_metrics(
                predictions, references, task_types
            ))
        
        # Linguistic metrics
        metrics.update(self.compute_linguistic_metrics(predictions))
        
        return metrics
    
    def compute_basic_metrics(self, predictions: List[str], 
                             references: List[str]) -> Dict[str, float]:
        """Compute basic metrics"""
        metrics = {}
        
        # BLEU scores
        bleu_scores = []
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.split()
            ref_tokens = [ref.split()]  # BLEU needs reference as list of list
            
            try:
                bleu = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=self.smoothie)
                bleu_scores.append(bleu)
            except:
                bleu_scores.append(0.0)
        
        metrics['bleu_score'] = np.mean(bleu_scores) if bleu_scores else 0.0
        
        # ROUGE scores
        rouge_1_scores = []
        rouge_2_scores = []
        rouge_l_scores = []
        
        for pred, ref in zip(predictions, references):
            try:
                scores = self.rouge_scorer.score(ref, pred)
                rouge_1_scores.append(scores['rouge1'].f1)
                rouge_2_scores.append(scores['rouge2'].f1)
                rouge_l_scores.append(scores['rougeL'].f1)
            except:
                rouge_1_scores.append(0.0)
                rouge_2_scores.append(0.0)
                rouge_l_scores.append(0.0)
        
        metrics['rouge1_f1'] = np.mean(rouge_1_scores)
        metrics['rouge2_f1'] = np.mean(rouge_2_scores)
        metrics['rougeL_f1'] = np.mean(rouge_l_scores)
        
        # BERTScore (if available)
        try:
            P, R, F1 = bert_score.score(predictions, references, lang='en')
            metrics['bertscore_f1'] = F1.mean().item()
        except:
            logger.warning("BERTScore unavailable, skipping")
            metrics['bertscore_f1'] = 0.0
        
        return metrics
    
    def compute_content_metrics(self, predictions: List[str], 
                               references: List[str]) -> Dict[str, float]:
        """Compute content quality metrics"""
        metrics = {}
        
        # Comparison content coverage
        comparison_coverage = []
        for pred in predictions:
            has_comparison = any(kw in pred.lower() for kw in self.comparison_keywords)
            comparison_coverage.append(1.0 if has_comparison else 0.0)
        
        metrics['comparison_coverage'] = np.mean(comparison_coverage)
        
        # Explanation quality
        explanation_quality = []
        for pred in predictions:
            has_explanation = any(kw in pred.lower() for kw in self.explanation_keywords)
            explanation_quality.append(1.0 if has_explanation else 0.0)
        
        metrics['explanation_quality'] = np.mean(explanation_quality)
        
        # Example provision rate
        example_provision = []
        for pred in predictions:
            has_example = any(kw in pred.lower() for kw in self.example_keywords)
            example_provision.append(1.0 if has_example else 0.0)
        
        metrics['example_provision'] = np.mean(example_provision)
        
        # Response completeness
        completeness_scores = []
        for pred, ref in zip(predictions, references):
            # Simple completeness assessment: prediction length vs reference length
            length_ratio = min(len(pred) / max(len(ref), 1), 1.0)
            completeness_scores.append(length_ratio)
        
        metrics['response_completeness'] = np.mean(completeness_scores)
        
        return metrics
    
    def compute_task_specific_metrics(self, predictions: List[str],
                                     references: List[str],
                                     task_types: List[str]) -> Dict[str, float]:
        """Compute task-specific metrics"""
        metrics = {}
        
        # Group by task type
        task_groups = {}
        for i, task_type in enumerate(task_types):
            if task_type not in task_groups:
                task_groups[task_type] = []
            task_groups[task_type].append((predictions[i], references[i]))
        
        # Compute metrics for each task type
        for task_type, pairs in task_groups.items():
            task_preds = [p[0] for p in pairs]
            task_refs = [p[1] for p in pairs]
            
            if task_type == 'grammar_comparison':
                task_metrics = self._evaluate_grammar_comparison(task_preds, task_refs)
            elif task_type == 'vocabulary_comparison':
                task_metrics = self._evaluate_vocabulary_comparison(task_preds, task_refs)
            elif task_type == 'error_feedback':
                task_metrics = self._evaluate_error_feedback(task_preds, task_refs)
            elif task_type == 'parallel_explanation':
                task_metrics = self._evaluate_parallel_explanation(task_preds, task_refs)
            else:
                task_metrics = {}
            
            # Add task prefix
            for key, value in task_metrics.items():
                metrics[f"{task_type}_{key}"] = value
        
        return metrics
    
    def _evaluate_grammar_comparison(self, predictions: List[str], 
                                   references: List[str]) -> Dict[str, float]:
        """Evaluate grammar comparison tasks"""
        metrics = {}
        
        # Grammar term coverage
        grammar_terms = [
            'verb', 'noun', 'adjective', 'auxiliary', 'tense', 'voice', 'clause',
            '动词', '名词', '形容词', '助动词', '时态', '语态', '从句'
        ]
        
        term_coverage = []
        for pred in predictions:
            pred_lower = pred.lower()
            covered_terms = sum(1 for term in grammar_terms if term in pred_lower)
            coverage = covered_terms / len(grammar_terms)
            term_coverage.append(coverage)
        
        metrics['grammar_term_coverage'] = np.mean(term_coverage)
        
        # Rule explanation quality
        rule_indicators = ['rule', 'principle', 'position', '规则', '原则', '第一位', '第二位']
        rule_quality = []
        for pred in predictions:
            has_rule = any(indicator in pred.lower() for indicator in rule_indicators)
            rule_quality.append(1.0 if has_rule else 0.0)
        
        metrics['rule_explanation_quality'] = np.mean(rule_quality)
        
        return metrics
    
    def _evaluate_vocabulary_comparison(self, predictions: List[str],
                                      references: List[str]) -> Dict[str, float]:
        """Evaluate vocabulary comparison tasks"""  
        metrics = {}
        
        # False friend identification accuracy
        false_friend_indicators = ['false friend', 'confusing', '假朋友', '容易混淆']
        ff_identification = []
        for pred in predictions:
            has_ff_indicator = any(indicator in pred.lower() for indicator in false_friend_indicators)
            ff_identification.append(1.0 if has_ff_indicator else 0.0)
        
        metrics['false_friend_identification'] = np.mean(ff_identification)
        
        # Etymology explanation
        etymology_indicators = ['etymology', 'origin', '来源', '词源', '同源']
        etymology_explanation = []
        for pred in predictions:
            has_etymology = any(indicator in pred.lower() for indicator in etymology_indicators)
            etymology_explanation.append(1.0 if has_etymology else 0.0)
        
        metrics['etymology_explanation'] = np.mean(etymology_explanation)
        
        # Mnemonic provision
        mnemonic_indicators = ['mnemonic', 'remember', 'tip', '记忆', '技巧']
        mnemonic_provision = []
        for pred in predictions:
            has_mnemonic = any(indicator in pred.lower() for indicator in mnemonic_indicators)
            mnemonic_provision.append(1.0 if has_mnemonic else 0.0)
        
        metrics['mnemonic_provision'] = np.mean(mnemonic_provision)
        
        return metrics
    
    def _evaluate_error_feedback(self, predictions: List[str],
                                references: List[str]) -> Dict[str, float]:
        """Evaluate error feedback tasks"""
        metrics = {}
        
        # Error identification accuracy
        error_indicators = ['error', 'wrong', 'incorrect', '错误', '不对', '不正确']
        error_identification = []
        for pred in predictions:
            has_error_id = any(indicator in pred.lower() for indicator in error_indicators)
            error_identification.append(1.0 if has_error_id else 0.0)
        
        metrics['error_identification'] = np.mean(error_identification)
        
        # Correction suggestion quality
        correction_indicators = ['should', 'correct', 'change to', '应该', '正确', '改为']
        correction_quality = []
        for pred in predictions:
            has_correction = any(indicator in pred.lower() for indicator in correction_indicators)
            correction_quality.append(1.0 if has_correction else 0.0)
        
        metrics['correction_quality'] = np.mean(correction_quality)
        
        # Explanation clarity
        explanation_indicators = ['because', 'reason', 'rule', '因为', '原因', '规则']
        explanation_clarity = []
        for pred in predictions:
            has_explanation = any(indicator in pred.lower() for indicator in explanation_indicators)
            explanation_clarity.append(1.0 if has_explanation else 0.0)
        
        metrics['explanation_clarity'] = np.mean(explanation_clarity)
        
        return metrics
    
    def _evaluate_parallel_explanation(self, predictions: List[str],
                                     references: List[str]) -> Dict[str, float]:
        """Evaluate parallel sentence explanation tasks"""
        metrics = {}
        
        # Correspondence analysis
        alignment_indicators = ['correspond', 'align', 'match', '对应', '对齐']
        alignment_analysis = []
        for pred in predictions:
            has_alignment = any(indicator in pred.lower() for indicator in alignment_indicators)
            alignment_analysis.append(1.0 if has_alignment else 0.0)
        
        metrics['alignment_analysis'] = np.mean(alignment_analysis)
        
        # Structure analysis
        structure_indicators = ['structure', 'word order', 'syntax', '结构', '语序']
        structure_analysis = []
        for pred in predictions:
            has_structure = any(indicator in pred.lower() for indicator in structure_indicators)
            structure_analysis.append(1.0 if has_structure else 0.0)
        
        metrics['structure_analysis'] = np.mean(structure_analysis)
        
        return metrics
    
    def compute_linguistic_metrics(self, predictions: List[str]) -> Dict[str, float]:
        """Compute linguistic metrics"""
        metrics = {}
        
        # Average length
        lengths = [len(pred) for pred in predictions]
        metrics['avg_response_length'] = np.mean(lengths)
        metrics['response_length_std'] = np.std(lengths)
        
        # Vocabulary diversity
        all_words = []
        for pred in predictions:
            words = re.findall(r'\w+', pred.lower())
            all_words.extend(words)
        
        if all_words:
            unique_words = len(set(all_words))
            total_words = len(all_words)
            metrics['vocabulary_diversity'] = unique_words / total_words
        else:
            metrics['vocabulary_diversity'] = 0.0
        
        # Sentence complexity (average sentence length)
        sentence_lengths = []
        for pred in predictions:
            sentences = re.split(r'[.!?]', pred)
            for sentence in sentences:
                if sentence.strip():
                    sentence_lengths.append(len(sentence.split()))
        
        metrics['avg_sentence_length'] = np.mean(sentence_lengths) if sentence_lengths else 0.0
        
        return metrics
    
    def compute_coherence_score(self, text: str) -> float:
        """Compute text coherence score (simplified version)"""
        # Connector check
        connectors = [
            'therefore', 'thus', 'then', 'but', 'however', 'moreover', 'additionally',
            '因此', '所以', '然后', '但是', '然而', '此外', '另外'
        ]
        
        connector_count = sum(1 for conn in connectors if conn in text.lower())
        
        # Sentence count
        sentences = re.split(r'[.!?]', text)
        sentence_count = len([s for s in sentences if s.strip()])
        
        if sentence_count <= 1:
            return 0.5
        
        # Coherence score = connector density
        coherence_score = min(connector_count / sentence_count, 1.0)
        
        return coherence_score
    
    def compute_accuracy_by_category(self, predictions: List[str],
                                   references: List[str],
                                   categories: List[str]) -> Dict[str, float]:
        """Compute accuracy by category"""
        category_metrics = {}
        
        # Group by category
        category_groups = {}
        for i, category in enumerate(categories):
            if category not in category_groups:
                category_groups[category] = {'preds': [], 'refs': []}
            category_groups[category]['preds'].append(predictions[i])
            category_groups[category]['refs'].append(references[i])
        
        # Compute metrics for each category
        for category, data in category_groups.items():
            preds = data['preds']
            refs = data['refs']
            
            # Compute basic metrics
            basic_metrics = self.compute_basic_metrics(preds, refs)
            content_metrics = self.compute_content_metrics(preds, refs)
            
            # Merge metrics
            for key, value in {**basic_metrics, **content_metrics}.items():
                category_metrics[f"{category}_{key}"] = value
        
        return category_metrics


def evaluate_model_outputs(model_outputs: List[str],
                          expected_outputs: List[str],
                          task_types: List[str] = None,
                          categories: List[str] = None) -> Dict[str, Any]:
    """Main function for evaluating model outputs
    
    Args:
        model_outputs: List of model outputs
        expected_outputs: List of expected outputs
        task_types: List of task types
        categories: List of categories
        
    Returns:
        Dictionary of evaluation results
    """
    evaluator = CrossLinguisticMetrics()
    
    # Compute all metrics
    metrics = evaluator.compute_all_metrics(
        model_outputs, expected_outputs, task_types
    )
    
    # Compute metrics by category (if provided)
    if categories:
        category_metrics = evaluator.compute_accuracy_by_category(
            model_outputs, expected_outputs, categories
        )
        metrics.update(category_metrics)
    
    # Add coherence scores
    coherence_scores = [evaluator.compute_coherence_score(output) 
                       for output in model_outputs]
    metrics['avg_coherence_score'] = np.mean(coherence_scores)
    
    # Compute overall score
    key_metrics = ['bleu_score', 'rouge1_f1', 'comparison_coverage', 
                   'explanation_quality', 'response_completeness']
    available_key_metrics = [metrics[k] for k in key_metrics if k in metrics]
    metrics['overall_score'] = np.mean(available_key_metrics) if available_key_metrics else 0.0
    
    return metrics


if __name__ == "__main__":
    # Test evaluation metrics
    predictions = [
        "English and German present perfect tense differ. English uses have/has + past participle, while German uses haben/sein + Partizip II. Example: I have seen -> Ich habe gesehen.",
        "become and bekommen are false friends. become means to become, bekommen means to get. The correct German should be werden."
    ]
    
    references = [
        "English present perfect uses have/has + past participle, German uses haben/sein + past participle, but German past participle goes at the end.",
        "These are false friend words. become=to become, use werden; bekommen=to get. Memory tip: bekommen=be+come=come to me=get."
    ]
    
    task_types = ['grammar_comparison', 'vocabulary_comparison']
    
    metrics = evaluate_model_outputs(predictions, references, task_types)
    
    print("Evaluation Results:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")