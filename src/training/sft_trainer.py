"""
SFT (Supervised Fine-Tuning) trainer
For supervised fine-tuning of base models
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset, load_dataset
import wandb
import logging
from typing import Dict, Optional, List
from pathlib import Path
import yaml
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CrossLinguisticSFTTrainer:
    """Cross-linguistic learning SFT trainer"""
    
    def __init__(self, config_path: str):
        """
        Args:
            config_path: Configuration file path
        """
        # Load configuration
        self.config = self._load_config(config_path)
        self.model_config = self._load_config(
            str(Path(config_path).parent / "model_config.yaml")
        )
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        self.peft_model = None
        
        # Initialize wandb if enabled
        if self.config['logging']['use_wandb']:
            wandb.init(
                project=self.config['logging']['wandb_project'],
                name=f"sft_training_{self.config['model_name'].split('/')[-1]}",
                config=self.config
            )
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def setup_model_and_tokenizer(self):
        """Setup model and tokenizer"""
        logger.info("Loading model and tokenizer...")
        
        # Quantization config
        bnb_config = None
        if self.model_config['quantization']['load_in_4bit']:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, self.model_config['quantization']['bnb_4bit_compute_dtype']),
                bnb_4bit_quant_type=self.model_config['quantization']['bnb_4bit_quant_type'],
                bnb_4bit_use_double_quant=self.model_config['quantization']['bnb_4bit_use_double_quant']
            )
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_config['base_model']['name'],
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=self.model_config['base_model']['trust_remote_code'],
            torch_dtype=torch.float16
        )
        
        # Prepare model for k-bit training
        if bnb_config:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config['base_model']['name'],
            trust_remote_code=self.model_config['base_model']['trust_remote_code']
        )
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Add special tokens
        special_tokens = {
            "additional_special_tokens": self.model_config['tokenizer']['special_tokens']
        }
        num_added = self.tokenizer.add_special_tokens(special_tokens)
        
        if num_added > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))
            logger.info(f"Added {num_added} special tokens")
        
        # Set tokenizer attributes
        self.tokenizer.padding_side = self.model_config['tokenizer']['padding_side']
        self.tokenizer.truncation_side = self.model_config['tokenizer']['truncation_side']
        
        logger.info("Model and tokenizer loaded successfully")
    
    def setup_lora(self):
        """Setup LoRA configuration"""
        if not self.model_config['lora']['use_lora']:
            logger.info("LoRA not used")
            return
            
        logger.info("Configuring LoRA...")
        
        # LoRA configuration
        lora_config = LoraConfig(
            r=self.model_config['lora']['r'],
            lora_alpha=self.model_config['lora']['lora_alpha'],
            lora_dropout=self.model_config['lora']['lora_dropout'],
            target_modules=self.model_config['lora']['target_modules'],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            modules_to_save=self.model_config['lora'].get('modules_to_save', None)
        )
        
        # Apply LoRA
        self.peft_model = get_peft_model(self.model, lora_config)
        self.peft_model.print_trainable_parameters()
        
        # Replace original model with PEFT model
        self.model = self.peft_model
        
        logger.info("LoRA configuration completed")
    
    def prepare_dataset(self, data_path: str) -> Dataset:
        """Prepare training dataset"""
        logger.info(f"Loading dataset: {data_path}")
        
        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert to Dataset format
        dataset = Dataset.from_list(data)
        
        # Data preprocessing function
        def preprocess_function(examples):
            # Get text
            texts = examples['text']
            
            # Tokenize
            model_inputs = self.tokenizer(
                texts,
                max_length=self.config['data']['max_length'],
                padding='max_length',
                truncation=True,
                return_tensors=None
            )
            
            # Set labels for loss calculation
            model_inputs['labels'] = model_inputs['input_ids'].copy()
            
            # Set padding token labels to -100 (ignore)
            for i in range(len(model_inputs['labels'])):
                labels = model_inputs['labels'][i]
                for j in range(len(labels)):
                    if labels[j] == self.tokenizer.pad_token_id:
                        labels[j] = -100
            
            return model_inputs
        
        # Apply preprocessing
        tokenized_dataset = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Data preprocessing"
        )
        
        logger.info(f"Dataset preparation completed, {len(tokenized_dataset)} samples")
        return tokenized_dataset
    
    def setup_training_args(self) -> TrainingArguments:
        """Setup training arguments"""
        
        # Calculate total training steps
        # Simplified here, should be calculated based on dataset size
        
        training_args = TrainingArguments(
            output_dir=self.config['output_dir'],
            num_train_epochs=self.config['sft']['num_epochs'],
            per_device_train_batch_size=self.config['sft']['batch_size'],
            per_device_eval_batch_size=self.config['sft']['batch_size'],
            gradient_accumulation_steps=self.config['sft']['gradient_accumulation_steps'],
            warmup_steps=self.config['sft']['warmup_steps'],
            weight_decay=self.config['sft']['weight_decay'],
            learning_rate=self.config['sft']['learning_rate'],
            max_grad_norm=self.config['sft']['max_grad_norm'],
            
            # Logging and saving
            logging_dir=f"{self.config['output_dir']}/logs",
            logging_steps=self.config['sft']['logging_steps'],
            save_steps=self.config['sft']['save_steps'],
            save_total_limit=3,
            
            # Evaluation
            evaluation_strategy="steps",
            eval_steps=self.config['sft']['eval_steps'],
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            load_best_model_at_end=True,
            
            # Performance optimization
            fp16=self.config['hardware']['fp16'],
            gradient_checkpointing=self.config['hardware']['gradient_checkpointing'],
            
            # Other
            seed=self.config['seed'],
            data_seed=self.config['seed'],
            remove_unused_columns=False,
            label_names=["labels"],
            
            # Wandb
            report_to="wandb" if self.config['logging']['use_wandb'] else "none",
            run_name=f"sft_{self.config['model_name'].split('/')[-1]}"
        )
        
        return training_args
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        
        # Calculate perplexity
        loss = nn.CrossEntropyLoss()(
            torch.tensor(predictions).view(-1, predictions.shape[-1]),
            torch.tensor(labels).view(-1)
        )
        perplexity = torch.exp(loss)
        
        return {
            "perplexity": perplexity.item()
        }
    
    def train(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None):
        """Execute training"""
        logger.info("Starting SFT training...")
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        # Training arguments
        training_args = self.setup_training_args()
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=None  # No additional metrics for now
        )
        
        # Start training
        train_result = trainer.train()
        
        # Save model
        logger.info("Saving model...")
        trainer.save_model(self.config['output_dir'])
        self.tokenizer.save_pretrained(self.config['output_dir'])
        
        # Save training results
        with open(f"{self.config['output_dir']}/train_results.json", 'w') as f:
            json.dump(train_result.metrics, f, indent=2)
        
        # Save configuration
        with open(f"{self.config['output_dir']}/config.yaml", 'w') as f:
            yaml.dump(self.config, f)
            
        with open(f"{self.config['output_dir']}/model_config.yaml", 'w') as f:
            yaml.dump(self.model_config, f)
        
        logger.info(f"Training completed! Model saved at: {self.config['output_dir']}")
        
        return trainer
    
    def generate_sample(self, prompt: str, max_new_tokens: int = 256) -> str:
        """Generate sample output for validation"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=self.model_config['generation']['temperature'],
                top_p=self.model_config['generation']['top_p'],
                do_sample=self.model_config['generation']['do_sample'],
                repetition_penalty=self.model_config['generation']['repetition_penalty']
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    
    def test_model(self, test_prompts: List[str]):
        """Test model output"""
        logger.info("Testing model output...")
        
        for i, prompt in enumerate(test_prompts):
            logger.info(f"\nTest {i+1}:")
            logger.info(f"Input: {prompt}")
            
            response = self.generate_sample(prompt)
            logger.info(f"Output: {response}")
            logger.info("-" * 80)


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SFT training script")
    parser.add_argument("--config", type=str, required=True, help="Configuration file path")
    parser.add_argument("--train_data", type=str, required=True, help="Training data path")
    parser.add_argument("--eval_data", type=str, help="Validation data path")
    parser.add_argument("--test_only", action="store_true", help="Test model only")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = CrossLinguisticSFTTrainer(args.config)
    
    # Setup model and tokenizer
    trainer.setup_model_and_tokenizer()
    
    # Setup LoRA
    trainer.setup_lora()
    
    if args.test_only:
        # Test mode
        test_prompts = [
            "[COMPARE] [GRAMMAR] Please compare the usage of present perfect tense in English and German.",
            "[VOCAB] [ENGLISH] become [GERMAN] bekommen - Are these two words false friends?",
            "[FEEDBACK] Student wrote: 'Ich habe zur Schule gegangen.' What's wrong with this sentence?"
        ]
        trainer.test_model(test_prompts)
    else:
        # Prepare datasets
        train_dataset = trainer.prepare_dataset(args.train_data)
        eval_dataset = None
        if args.eval_data:
            eval_dataset = trainer.prepare_dataset(args.eval_data)
        
        # Execute training
        trainer.train(train_dataset, eval_dataset)
        
        # Test trained model
        test_prompts = [
            "[COMPARE] [GRAMMAR] Explain the rules for verb position in German subordinate clauses."
        ]
        trainer.test_model(test_prompts)


if __name__ == "__main__":
    main()