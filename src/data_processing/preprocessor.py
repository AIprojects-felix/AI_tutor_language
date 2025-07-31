"""
Data preprocessing module
Convert raw data to training format
"""

import re
import json
from typing import List, Dict, Optional, Tuple
from transformers import AutoTokenizer
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Data preprocessor"""
    
    def __init__(self, model_name: str = "microsoft/Phi-4-multimodal-instruct", 
                 max_length: int = 1024):
        self.model_name = model_name
        self.max_length = max_length
        
        # Initialize tokenizer
        logger.info(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add special tokens
        self._add_special_tokens()
        
        # Define prompt templates
        self.prompt_templates = self._init_prompt_templates()
        
    def _add_special_tokens(self):
        """Add special tokens"""
        special_tokens = {
            "additional_special_tokens": [
                "[ENGLISH]", "[GERMAN]", "[COMPARE]", 
                "[GRAMMAR]", "[VOCAB]", "[FEEDBACK]",
                "[AUDIO]", "[EXPLAIN]", "[CORRECT]", "[ERROR]"
            ]
        }
        num_added = self.tokenizer.add_special_tokens(special_tokens)
        logger.info(f"Added {num_added} special tokens")
        
    def _init_prompt_templates(self) -> Dict[str, str]:
        """Initialize prompt templates"""
        return {
            'grammar_comparison': """[COMPARE] [GRAMMAR]
### Grammar Comparison Task

Topic: {topic}

English Rule: {english_rule}
German Rule: {german_rule}

Please provide a detailed analysis of the similarities and differences between these two languages for this grammar point, including:
1. Structural differences
2. Usage contexts
3. Common errors
4. Learning suggestions

### Analysis:
{explanation}""",

            'vocabulary_comparison': """[COMPARE] [VOCAB]
### Vocabulary Comparison Task

English Word: {english_word}
English Meaning: {english_meaning}
German Equivalent: {german_word}
German Meaning: {german_meaning}

Please explain:
1. Etymology relationship
2. Usage differences
3. Common confusion points
4. Memory techniques

### Analysis:
{explanation}""",

            'error_feedback': """[FEEDBACK] [ERROR]
### Error Analysis Task

Student Input: {student_input}
Correct Form: {correct_form}
Error Type: {error_type}

Please provide:
1. Error cause analysis
2. Correction method
3. Related rule explanation
4. Practice suggestions

### Feedback:
{feedback}""",

            'parallel_explanation': """[ENGLISH] [GERMAN] [EXPLAIN]
### Parallel Sentence Analysis

English: {english}
German: {german}

Grammar Points: {grammar_points}
Difficulty Level: {difficulty}

Please explain:
1. Sentence structure comparison
2. Key vocabulary correspondence
3. Grammar difference points
4. Cultural background (if applicable)

### Explanation:
{explanation}""",

            'audio_transcription': """[AUDIO] [FEEDBACK]
### Speech Practice Feedback

User Pronunciation Text: {transcription}
Standard Pronunciation: {standard}
Language: {language}

Please analyze:
1. Pronunciation accuracy
2. Phonemes needing improvement
3. Practice suggestions

### Feedback:
{feedback}"""
        }
    
    def create_comparative_prompt(self, data: Dict, prompt_type: str) -> str:
        """Create comparative learning prompt"""
        template = self.prompt_templates.get(prompt_type)
        if not template:
            logger.warning(f"Template not found: {prompt_type}")
            return ""
            
        try:
            # Fill template
            prompt = template.format(**data)
            return prompt
        except KeyError as e:
            logger.error(f"Template formatting failed, missing field: {e}")
            return ""
    
    def process_grammar_data(self, grammar_data: List[Dict]) -> List[Dict]:
        """Process grammar comparison data"""
        processed_data = []
        
        for item in grammar_data:
            # Generate explanation text
            explanation = self._generate_grammar_explanation(item)
            
            # Create training sample
            prompt_data = {
                'topic': item['topic'],
                'english_rule': item['english_rule'],
                'german_rule': item['german_rule'],
                'explanation': explanation
            }
            
            # Create prompt
            prompt = self.create_comparative_prompt(prompt_data, 'grammar_comparison')
            
            if prompt and self._check_length(prompt):
                processed_item = {
                    'text': prompt,
                    'type': 'grammar_comparison',
                    'id': item.get('id', ''),
                    'metadata': {
                        'topic': item['topic'],
                        'difficulty': item.get('difficulty', 'intermediate')
                    }
                }
                processed_data.append(processed_item)
        
        logger.info(f"Processed {len(processed_data)} grammar comparison items")
        return processed_data
    
    def process_vocabulary_data(self, vocab_data: List[Dict]) -> List[Dict]:
        """Process vocabulary comparison data"""
        processed_data = []
        
        for item in vocab_data:
            # Generate explanation text
            explanation = self._generate_vocab_explanation(item)
            
            # Create training sample
            prompt_data = {
                'english_word': item['english_word'],
                'english_meaning': item['english_meaning'],
                'german_word': item.get('correct_german', item.get('german_false_friend', '')),
                'german_meaning': item.get('german_false_meaning', ''),
                'explanation': explanation
            }
            
            # Create prompt
            prompt = self.create_comparative_prompt(prompt_data, 'vocabulary_comparison')
            
            if prompt and self._check_length(prompt):
                processed_item = {
                    'text': prompt,
                    'type': 'vocabulary_comparison',
                    'id': item.get('id', ''),
                    'metadata': {
                        'english_word': item['english_word'],
                        'difficulty': item.get('difficulty', 'intermediate')
                    }
                }
                processed_data.append(processed_item)
        
        logger.info(f"Processed {len(processed_data)} vocabulary comparison items")
        return processed_data
    
    def process_error_data(self, error_data: List[Dict]) -> List[Dict]:
        """Process error correction data"""
        processed_data = []
        
        for item in error_data:
            # Generate feedback text
            feedback = self._generate_error_feedback(item)
            
            # Create training sample
            prompt_data = {
                'student_input': item['example_error'],
                'correct_form': item['correct_form'],
                'error_type': item['error_type'],
                'feedback': feedback
            }
            
            # Create prompt
            prompt = self.create_comparative_prompt(prompt_data, 'error_feedback')
            
            if prompt and self._check_length(prompt):
                processed_item = {
                    'text': prompt,
                    'type': 'error_feedback',
                    'id': item.get('id', ''),
                    'metadata': {
                        'error_type': item['error_type'],
                        'level': item.get('level', 'intermediate')
                    }
                }
                processed_data.append(processed_item)
        
        logger.info(f"Processed {len(processed_data)} error feedback items")
        return processed_data
    
    def process_parallel_data(self, parallel_data: List[Dict]) -> List[Dict]:
        """Process parallel corpus data"""
        processed_data = []
        
        for item in parallel_data:
            # Generate explanation text
            explanation = self._generate_parallel_explanation(item)
            
            # Create training sample
            prompt_data = {
                'english': item['english'],
                'german': item['german'],
                'grammar_points': ', '.join(item.get('grammar_points', [])),
                'difficulty': item.get('difficulty', 'intermediate'),
                'explanation': explanation
            }
            
            # Create prompt
            prompt = self.create_comparative_prompt(prompt_data, 'parallel_explanation')
            
            if prompt and self._check_length(prompt):
                processed_item = {
                    'text': prompt,
                    'type': 'parallel_explanation',
                    'id': item.get('id', ''),
                    'metadata': {
                        'difficulty': item.get('difficulty', 'intermediate'),
                        'grammar_points': item.get('grammar_points', [])
                    }
                }
                processed_data.append(processed_item)
        
        logger.info(f"Processed {len(processed_data)} parallel explanation items")
        return processed_data
    
    def _generate_grammar_explanation(self, item: Dict) -> str:
        """Generate grammar explanation text"""
        explanation_parts = []
        
        # Key differences
        if 'key_differences' in item:
            explanation_parts.append("Key differences:")
            for diff in item['key_differences']:
                explanation_parts.append(f"- {diff}")
        
        # Examples
        if 'examples' in item:
            explanation_parts.append("\nExamples:")
            for example in item['examples']:
                explanation_parts.append(f"English: {example.get('english', '')}")
                explanation_parts.append(f"German: {example.get('german', '')}")
                if 'explanation' in example:
                    explanation_parts.append(f"Note: {example['explanation']}")
                explanation_parts.append("")
        
        # Learning objectives
        if 'learning_objectives' in item:
            explanation_parts.append("Learning objectives:")
            for obj in item['learning_objectives']:
                explanation_parts.append(f"- {obj}")
                
        return '\n'.join(explanation_parts)
    
    def _generate_vocab_explanation(self, item: Dict) -> str:
        """Generate vocabulary explanation text"""
        explanation_parts = []
        
        # Basic explanation
        explanation_parts.append(item.get('explanation', ''))
        
        # Examples
        if 'examples' in item:
            explanation_parts.append("\nExamples:")
            for example in item['examples']:
                explanation_parts.append(f"English: {example.get('english', '')}")
                explanation_parts.append(f"❌ Wrong German: {example.get('wrong_german', '')}")
                explanation_parts.append(f"✅ Correct German: {example.get('correct_german', '')}")
                if 'explanation' in example:
                    explanation_parts.append(f"Explanation: {example['explanation']}")
        
        # Memory tips
        if 'memory_tips' in item:
            explanation_parts.append("\nMemory tips:")
            for tip in item['memory_tips']:
                explanation_parts.append(f"💡 {tip}")
                
        return '\n'.join(explanation_parts)
    
    def _generate_error_feedback(self, item: Dict) -> str:
        """Generate error feedback text"""
        feedback_parts = []
        
        # Error analysis
        feedback_parts.append(f"Error type: {item['error_type']}")
        feedback_parts.append(f"\nReason: {item['explanation']}")
        
        # Related rule
        if 'related_rule' in item:
            feedback_parts.append(f"\nRelated rule: {item['related_rule']}")
        
        # Practice suggestions
        if 'practice_suggestions' in item:
            feedback_parts.append("\nPractice suggestions:")
            for suggestion in item['practice_suggestions']:
                feedback_parts.append(f"- {suggestion}")
                
        return '\n'.join(feedback_parts)
    
    def _generate_parallel_explanation(self, item: Dict) -> str:
        """Generate parallel sentence explanation"""
        explanation_parts = []
        
        # Alignment analysis
        if 'alignment' in item:
            explanation_parts.append("Word correspondence:")
            for align in item['alignment']:
                explanation_parts.append(f"- {align['en']} → {align['de']}")
        
        # Grammar points
        if 'grammar_points' in item:
            explanation_parts.append(f"\nGrammar points covered:")
            for point in item['grammar_points']:
                explanation_parts.append(f"- {point}")
                
        return '\n'.join(explanation_parts)
    
    def _check_length(self, text: str) -> bool:
        """Check if text length meets requirements"""
        tokens = self.tokenizer.encode(text)
        if len(tokens) > self.max_length:
            logger.warning(f"Text too long: {len(tokens)} tokens (max: {self.max_length})")
            return False
        return True
    
    def format_for_training(self, all_data: List[Dict]) -> List[Dict]:
        """Format as final training data"""
        formatted_data = []
        
        for item in all_data:
            # Ensure required fields
            if 'text' not in item:
                continue
                
            # Add instruction prefix if needed
            formatted_item = {
                'text': item['text'],
                'type': item.get('type', 'unknown'),
                'metadata': item.get('metadata', {})
            }
            
            formatted_data.append(formatted_item)
            
        logger.info(f"Formatted {len(formatted_data)} training data items")
        return formatted_data
    
    def save_processed_data(self, data: List[Dict], output_path: str):
        """Save processed data"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Saved processed data to: {output_path}")
        

if __name__ == "__main__":
    # Test preprocessor
    preprocessor = DataPreprocessor()
    
    # Test grammar data processing
    test_grammar = [{
        'id': 'test_001',
        'topic': 'Present Perfect Tense',
        'english_rule': 'have/has + past participle',
        'german_rule': 'haben/sein + Partizip II',
        'key_differences': ['Different verb positions'],
        'examples': [{'english': 'I have seen', 'german': 'Ich habe gesehen'}]
    }]
    
    processed = preprocessor.process_grammar_data(test_grammar)
    print(f"Processing result: {len(processed)} items")
    if processed:
        print(f"Example:\n{processed[0]['text'][:200]}...")
