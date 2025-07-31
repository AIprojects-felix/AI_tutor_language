"""
RLHF (Reinforcement Learning from Human Feedback) trainer
Uses PPO algorithm for reinforcement learning optimization
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from trl import (
    PPOTrainer, 
    PPOConfig, 
    AutoModelForCausalLMWithValueHead,
    create_reference_model
)
from datasets import Dataset
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import yaml
import json
import wandb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RewardModel:
    """Reward model - evaluates quality of generated responses"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Define scoring weights
        self.weights = {
            'relevance': 0.3,      # Relevance
            'accuracy': 0.3,       # Accuracy
            'completeness': 0.2,   # Completeness
            'clarity': 0.2         # Clarity
        }
        
        # Keyword lists
        self.keywords = {
            'comparison': ['compare', 'differ', 'similar', 'contrast', 'versus'],
            'examples': ['for example', 'zum Beispiel', 'e.g.', 'such as', 'instance'],
            'explanation': ['because', 'therefore', 'weil', 'reason', 'since'],
            'german': ['German', 'Deutsch', 'deutsche'],
            'english': ['English', 'Englisch', 'englische']
        }
    
    def compute_reward(self, query: str, response: str) -> float:
        """Compute reward score
        
        Args:
            query: Input prompt
            response: Model generated response
            
        Returns:
            Reward score (0-1)
        """
        scores = {}
        
        # 1. Relevance score
        scores['relevance'] = self._score_relevance(query, response)
        
        # 2. Accuracy score (rule-based)
        scores['accuracy'] = self._score_accuracy(query, response)
        
        # 3. Completeness score
        scores['completeness'] = self._score_completeness(response)
        
        # 4. Clarity score
        scores['clarity'] = self._score_clarity(response)
        
        # Weighted average
        total_score = sum(scores[k] * self.weights[k] for k in scores)
        
        # Normalize to [-1, 1]
        normalized_score = 2 * total_score - 1
        
        return normalized_score
    
    def _score_relevance(self, query: str, response: str) -> float:
        """Evaluate response relevance"""
        score = 0.0
        
        # Check if contains keywords from query
        query_lower = query.lower()
        response_lower = response.lower()
        
        # Task type detection
        if '[COMPARE]' in query:
            # Comparison task - check for comparison content
            if any(kw in response_lower for kw in self.keywords['comparison']):
                score += 0.5
            if self.keywords['english'][0] in response_lower and self.keywords['german'][0] in response_lower:
                score += 0.5
                
        elif '[VOCAB]' in query:
            # Vocabulary task - check if vocabulary is explained
            if 'meaning' in response_lower:
                score += 0.3
            if any(kw in response_lower for kw in self.keywords['examples']):
                score += 0.3
            if 'false friend' in response_lower:
                score += 0.4
                
        elif '[FEEDBACK]' in query or 'error' in query_lower:
            # Feedback task - check if correction is provided
            if 'correct' in response_lower:
                score += 0.5
            if any(kw in response_lower for kw in self.keywords['explanation']):
                score += 0.5
        else:
            # General scoring
            score = 0.5
            
        return min(score, 1.0)
    
    def _score_accuracy(self, query: str, response: str) -> float:
        """Evaluate response accuracy (simplified version)"""
        score = 0.8  # Base score
        
        # Check for obvious errors
        response_lower = response.lower()
        
        # Penalty for very short responses
        if len(response) < 50:
            score -= 0.3
            
        # Bonus for structured responses
        if any(marker in response for marker in ['1.', '2.', '-', '•']):
            score += 0.1
            
        # Bonus for examples
        if any(kw in response_lower for kw in self.keywords['examples']):
            score += 0.1
            
        return min(score, 1.0)
    
    def _score_completeness(self, response: str) -> float:
        """Evaluate response completeness"""
        score = 0.5  # Base score
        
        # Length-based scoring
        if len(response) > 200:
            score += 0.3
        elif len(response) > 100:
            score += 0.2
        elif len(response) < 50:
            score -= 0.2
            
        # Structure scoring
        if response.count('\n') >= 2:  # Multiple paragraphs
            score += 0.2
            
        return min(score, 1.0)
    
    def _score_clarity(self, response: str) -> float:
        """Evaluate response clarity"""
        score = 0.7  # Base score
        
        # Check for clear structure
        if any(marker in response for marker in [':', '1.', '2.', '-']):
            score += 0.2
            
        # Penalty for very long sentences
        sentences = response.split('.')
        avg_length = sum(len(s) for s in sentences) / len(sentences) if sentences else 0
        if avg_length > 150:
            score -= 0.1
            
        return min(score, 1.0)


class CrossLinguisticRLHFTrainer:
    """Cross-linguistic learning RLHF trainer"""
    
    def __init__(self, sft_model_path: str, config_path: str):
        """
        Args:
            sft_model_path: SFT model path
            config_path: Configuration file path
        """
        self.sft_model_path = sft_model_path
        self.config = self._load_config(config_path)
        
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.ref_model = None
        self.reward_model = None
        self.ppo_trainer = None
        
        # Initialize reward model
        self.reward_model = RewardModel(self.config.get('reward_model', {}))
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def setup_models(self):
        """Setup models"""
        logger.info(f"Loading models from: {self.sft_model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.sft_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with value head
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
            self.sft_model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Create reference model
        self.ref_model = create_reference_model(self.model)
        
        logger.info("Models loaded successfully")
    
    def setup_ppo_config(self):
        """Setup PPO configuration"""
        ppo_config = PPOConfig(
            model_name=self.sft_model_path,
            learning_rate=self.config.get('learning_rate', 1e-5),
            batch_size=self.config.get('batch_size', 4),
            mini_batch_size=self.config.get('mini_batch_size', 1),
            gradient_accumulation_steps=self.config.get('gradient_accumulation_steps', 4),
            optimize_cuda_cache=True,
            early_stopping=False,
            target_kl=0.1,
            ppo_epochs=self.config.get('ppo_epochs', 4),
            seed=self.config.get('seed', 42),
            init_kl_coef=0.2,
            adap_kl_ctrl=True,
        )
        
        # Initialize PPO trainer
        self.ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=self.model,
            ref_model=self.ref_model,
            tokenizer=self.tokenizer,
        )
        
        logger.info("PPO trainer initialized")
    
    def prepare_dataset(self, data_path: str) -> Dataset:
        """Prepare RLHF dataset"""
        logger.info(f"Loading RLHF dataset from: {data_path}")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert to required format
        dataset_items = []
        for item in data:
            # Tokenize prompt
            prompt = item['prompt']
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=False
            )
            
            dataset_items.append({
                'query': prompt,
                'input_ids': inputs['input_ids'].squeeze(),
                'attention_mask': inputs['attention_mask'].squeeze()
            })
        
        dataset = Dataset.from_list(dataset_items)
        logger.info(f"Prepared dataset with {len(dataset)} samples")
        
        return dataset
    
    def generate_responses(self, queries: List[str], **kwargs) -> Tuple[List[str], List[torch.Tensor]]:
        """Generate responses"""
        generation_kwargs = {
            'max_new_tokens': kwargs.get('max_new_tokens', 256),
            'temperature': kwargs.get('temperature', 0.7),
            'top_p': kwargs.get('top_p', 0.9),
            'do_sample': True,
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
        }
        
        responses = []
        response_tensors = []
        
        for query in queries:
            # Generate response
            response_tensor = self.ppo_trainer.generate(
                query, return_prompt=False, **generation_kwargs
            )
            response = self.tokenizer.decode(response_tensor.squeeze(), skip_special_tokens=True)
            
            responses.append(response)
            response_tensors.append(response_tensor.squeeze())
        
        return responses, response_tensors
    
    def compute_rewards(self, queries: List[str], responses: List[str]) -> List[torch.Tensor]:
        """Compute rewards"""
        rewards = []
        for query, response in zip(queries, responses):
            reward_score = self.reward_model.compute_reward(query, response)
            reward_tensor = torch.tensor(reward_score, dtype=torch.float32)
            rewards.append(reward_tensor)
        
        return rewards
    
    def train(self, dataset: Dataset):
        """Execute RLHF training"""
        logger.info("Starting RLHF training...")
        
        # Setup PPO
        self.setup_ppo_config()
        
        # Training loop
        for epoch in range(self.config.get('num_epochs', 1)):
            logger.info(f"Epoch {epoch + 1}/{self.config.get('num_epochs', 1)}")
            
            for batch_idx, batch in enumerate(self.ppo_trainer.dataloader):
                
                # Extract queries
                query_tensors = [item['input_ids'] for item in batch]
                queries = [item['query'] for item in batch]
                
                # Generate responses
                responses, response_tensors = self.generate_responses(queries)
                
                # Compute rewards
                rewards = self.compute_rewards(queries, responses)
                
                # Convert to correct format
                query_tensors = [q.to(self.device) for q in query_tensors]
                response_tensors = [r.to(self.device) for r in response_tensors]
                rewards = [r.to(self.device) for r in rewards]
                
                # PPO step
                stats = self.ppo_trainer.step(query_tensors, response_tensors, rewards)
                
                # Log
                if batch_idx % 10 == 0:
                    logger.info(f"Batch {batch_idx}: {stats}")
                    
                    # Print examples
                    if batch_idx % 50 == 0 and len(queries) > 0:
                        logger.info(f"\nExample:")
                        logger.info(f"Query: {queries[0]}")
                        logger.info(f"Response: {responses[0]}")
                        logger.info(f"Reward: {rewards[0].item():.3f}")
        
        # Save model
        logger.info("Saving RLHF model...")
        output_dir = f"{self.config['output_dir']}_rlhf"
        self.ppo_trainer.save_model(output_dir)
        
        # Save configuration
        with open(f"{output_dir}/rlhf_config.yaml", 'w') as f:
            yaml.dump(self.config, f)
        
        logger.info(f"RLHF training completed! Model saved at: {output_dir}")
    
    def evaluate(self, test_prompts: List[str]):
        """Evaluate model"""
        logger.info("Evaluating RLHF model...")
        
        for prompt in test_prompts:
            logger.info(f"\nPrompt: {prompt}")
            
            # Generate response
            responses, _ = self.generate_responses([prompt], max_new_tokens=300)
            response = responses[0]
            
            # Compute reward
            reward = self.reward_model.compute_reward(prompt, response)
            
            logger.info(f"Response: {response}")
            logger.info(f"Reward score: {reward:.3f}")
            logger.info("-" * 80)


def create_preference_dataset(output_path: str):
    """Create preference dataset (example)"""
    preference_data = [
        {
            "prompt": "[COMPARE] [GRAMMAR] Please compare the usage differences between English and German present perfect tense.",
            "chosen": "English and German present perfect tense have significant differences:\n\n1. Auxiliary verb selection:\n- English: uniformly uses have/has\n- German: requires choosing haben or sein\n\n2. Verb position:\n- English: auxiliary + past participle (I have seen)\n- German: auxiliary in second position, past participle at end (Ich habe gesehen)\n\n3. Usage contexts:\n- English: emphasizes past action's impact on present\n- German: can also be used as simple past tense",
            "rejected": "Present perfect tense just expresses completed actions."
        },
        {
            "prompt": "[VOCAB] Explain the difference between 'become' and 'bekommen'.",
            "chosen": "These are typical false friend words:\n\nEnglish 'become' = to turn into, to grow to be\nGerman 'bekommen' = to get, to receive\n\nCorrect correspondences:\n- become → werden\n- get/receive → bekommen\n\nExample sentences:\n✓ I want to become a doctor. → Ich möchte Arzt werden.\n❌ Ich möchte Arzt bekommen. (Wrong: I want to get a doctor)",
            "rejected": "become and bekommen mean roughly the same thing."
        }
    ]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(preference_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Created preference dataset: {output_path}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RLHF training script")
    parser.add_argument("--sft_model", type=str, required=True, help="SFT model path")
    parser.add_argument("--config", type=str, required=True, help="Configuration file path")
    parser.add_argument("--data", type=str, required=True, help="RLHF data path")
    parser.add_argument("--create_data", action="store_true", help="Create example data")
    
    args = parser.parse_args()
    
    if args.create_data:
        # Create example preference data
        create_preference_dataset(args.data)
        return
    
    # Initialize trainer
    trainer = CrossLinguisticRLHFTrainer(args.sft_model, args.config)
    
    # Setup models
    trainer.setup_models()
    
    # Prepare dataset
    dataset = trainer.prepare_dataset(args.data)
    
    # Execute training
    trainer.train(dataset)
    
    # Evaluate
    test_prompts = [
        "[COMPARE] [GRAMMAR] Explain the verb-second principle in German.",
        "[VOCAB] What are the different meanings of 'gift' in English and German?",
        "[FEEDBACK] A student wrote 'Ich habe nach Hause gefahren.' Is this sentence correct?"
    ]
    trainer.evaluate(test_prompts)


if __name__ == "__main__":
    main()
