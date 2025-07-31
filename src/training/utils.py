"""
Training utility functions
"""

import torch
import numpy as np
import random
import os
from typing import Dict, List, Optional, Tuple
import json
import yaml
from pathlib import Path
import logging
from datetime import datetime
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Set random seed: {seed}")


def save_checkpoint(model, tokenizer, optimizer, epoch: int, 
                   save_dir: str, metrics: Dict = None):
    """Save checkpoint"""
    checkpoint_dir = Path(save_dir) / f"checkpoint-epoch-{epoch}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model and tokenizer
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    
    # Save optimizer state
    torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
    
    # Save training state
    state = {
        'epoch': epoch,
        'timestamp': datetime.now().isoformat(),
        'metrics': metrics or {}
    }
    
    with open(checkpoint_dir / "training_state.json", 'w') as f:
        json.dump(state, f, indent=2)
    
    logger.info(f"Saved checkpoint to: {checkpoint_dir}")


def load_checkpoint(checkpoint_dir: str, model, optimizer=None):
    """Load checkpoint"""
    checkpoint_path = Path(checkpoint_dir)
    
    # Load training state
    with open(checkpoint_path / "training_state.json", 'r') as f:
        state = json.load(f)
    
    # Load optimizer state
    if optimizer and (checkpoint_path / "optimizer.pt").exists():
        optimizer.load_state_dict(torch.load(checkpoint_path / "optimizer.pt"))
    
    logger.info(f"Loaded checkpoint from {checkpoint_dir}, epoch: {state['epoch']}")
    
    return state


def compute_model_metrics(model, eval_dataloader, device) -> Dict:
    """Compute model evaluation metrics"""
    model.eval()
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in eval_dataloader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            
            loss = outputs.loss
            total_loss += loss.item() * inputs['input_ids'].size(0)
            total_samples += inputs['input_ids'].size(0)
    
    avg_loss = total_loss / total_samples
    perplexity = np.exp(avg_loss)
    
    metrics = {
        'eval_loss': avg_loss,
        'eval_perplexity': perplexity
    }
    
    return metrics


def create_training_report(config: Dict, results: Dict, save_path: str):
    """Create training report"""
    report = {
        'training_config': config,
        'training_results': results,
        'timestamp': datetime.now().isoformat(),
        'environment': {
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
    }
    
    report_path = Path(save_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Training report saved to: {report_path}")


def merge_lora_weights(base_model_path: str, lora_path: str, output_path: str):
    """Merge LoRA weights into base model"""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    logger.info("Starting LoRA weight merging...")
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Load LoRA model
    model = PeftModel.from_pretrained(base_model, lora_path)
    
    # Merge weights
    model = model.merge_and_unload()
    
    # Save merged model
    model.save_pretrained(output_path)
    
    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(output_path)
    
    logger.info(f"LoRA weights merged and saved to: {output_path}")


def estimate_memory_usage(model_name: str, batch_size: int, 
                         sequence_length: int, use_lora: bool = True) -> Dict:
    """Estimate training memory usage"""
    
    # Model parameter estimates (in millions)
    model_params = {
        "microsoft/Phi-4-multimodal-instruct": 14000,  # 14B parameters
        "microsoft/Phi-3-mini-4k-instruct": 3800,     # 3.8B parameters
        "microsoft/Phi-3-small-8k-instruct": 7000,    # 7B parameters
    }
    
    # Get parameter count
    params = model_params.get(model_name, 7000)  # Default to 7B
    
    # Memory calculations (rough estimates)
    # Model weights: params * 2 bytes (fp16)
    model_memory = params * 2 / 1024  # MB
    
    # Activations: batch_size * sequence_length * hidden_size * layers * 2 bytes
    hidden_size = 4096  # Typical hidden size
    num_layers = 32     # Typical layer count
    activation_memory = batch_size * sequence_length * hidden_size * num_layers * 2 / (1024 * 1024)  # MB
    
    # Gradients: same as model weights if full fine-tuning
    if use_lora:
        # LoRA typically uses 1-10% of original parameters
        gradient_memory = model_memory * 0.05  # 5% estimate
    else:
        gradient_memory = model_memory
    
    # Optimizer states (Adam): 2x gradients
    optimizer_memory = gradient_memory * 2
    
    # Total memory
    total_memory = model_memory + activation_memory + gradient_memory + optimizer_memory
    
    return {
        'model_parameters_millions': params,
        'model_memory_mb': round(model_memory, 2),
        'activation_memory_mb': round(activation_memory, 2),
        'gradient_memory_mb': round(gradient_memory, 2),
        'optimizer_memory_mb': round(optimizer_memory, 2),
        'total_memory_mb': round(total_memory, 2),
        'total_memory_gb': round(total_memory / 1024, 2),
        'use_lora': use_lora,
        'batch_size': batch_size,
        'sequence_length': sequence_length
    }


def analyze_dataset_statistics(dataset_path: str) -> Dict:
    """Analyze dataset statistics"""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    stats = {
        'total_samples': len(data),
        'type_distribution': {},
        'length_statistics': {
            'min': float('inf'),
            'max': 0,
            'mean': 0,
            'std': 0,
            'percentiles': {}
        }
    }
    
    lengths = []
    
    for item in data:
        # Type distribution
        item_type = item.get('type', 'unknown')
        stats['type_distribution'][item_type] = stats['type_distribution'].get(item_type, 0) + 1
        
        # Length statistics
        text_length = len(item.get('text', ''))
        lengths.append(text_length)
        stats['length_statistics']['min'] = min(stats['length_statistics']['min'], text_length)
        stats['length_statistics']['max'] = max(stats['length_statistics']['max'], text_length)
    
    # Calculate statistics
    if lengths:
        stats['length_statistics']['mean'] = np.mean(lengths)
        stats['length_statistics']['std'] = np.std(lengths)
        stats['length_statistics']['percentiles'] = {
            '25%': np.percentile(lengths, 25),
            '50%': np.percentile(lengths, 50),
            '75%': np.percentile(lengths, 75),
            '95%': np.percentile(lengths, 95)
        }
    
    return stats


def create_model_card(model_path: str, config: Dict, results: Dict):
    """Create model card (README)"""
    model_card = f"""# AI Tutor - English-German Cross-Linguistic Learning Model

## Model Description

This is an AI tutoring model specifically designed for English-German cross-linguistic comparative learning, fine-tuned from {config.get('model_name', 'Phi-4')}.

## Training Details

- **Base Model**: {config.get('model_name', 'microsoft/Phi-4-multimodal-instruct')}
- **Training Method**: SFT + RLHF
- **Training Epochs**: {config.get('sft', {}).get('num_epochs', 3)} (SFT) + {config.get('rlhf', {}).get('num_epochs', 2)} (RLHF)
- **Batch Size**: {config.get('sft', {}).get('batch_size', 4)}
- **Learning Rate**: {config.get('sft', {}).get('learning_rate', 2e-5)}
- **Uses LoRA**: {config.get('lora', {}).get('use_lora', True)}

## Performance Metrics

{json.dumps(results, indent=2)}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{model_path}")
tokenizer = AutoTokenizer.from_pretrained("{model_path}")

# Grammar comparison
prompt = "[COMPARE] [GRAMMAR] Please compare the usage of present perfect tense in English and German."
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Core Features

1. **Grammar Comparison Analysis** - Detailed comparison of English-German grammar differences
2. **Vocabulary Comparison Analysis** - Identification of false friends and cognates
3. **Audio Understanding** - Processing of speech input
4. **Pronunciation Guidance** - Pronunciation feedback
5. **Educational Explanations** - Clear rule explanations
6. **Error Feedback** - Personalized correction suggestions

## Limitations and Biases

- Model primarily targets Chinese native speakers learning English-German bilingual content
- Limited training data may not cover all grammatical phenomena
- Pronunciation feedback is text-based and requires TTS integration

## Citation

If you use this model, please cite:

```
@misc{{ai_tutor_2024,
  title={{AI Tutor for Cross-Linguistic Language Learning}},
  author={{Your Name}},
  year={{2024}},
  publisher={{Your Institution}}
}}
```
"""
    
    model_card_path = Path(model_path) / "README.md"
    with open(model_card_path, 'w', encoding='utf-8') as f:
        f.write(model_card)
    
    logger.info(f"Model card saved to: {model_card_path}")


def backup_model(model_path: str, backup_dir: str):
    """Backup model"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = Path(backup_dir) / f"model_backup_{timestamp}"
    
    shutil.copytree(model_path, backup_path)
    logger.info(f"Model backed up to: {backup_path}")
    
    # Keep only the latest 3 backups
    backups = sorted(Path(backup_dir).glob("model_backup_*"))
    if len(backups) > 3:
        for old_backup in backups[:-3]:
            shutil.rmtree(old_backup)
            logger.info(f"Deleted old backup: {old_backup}")


if __name__ == "__main__":
    # Test utility functions
    
    # Set random seed
    set_seed(42)
    
    # Estimate memory usage
    memory_info = estimate_memory_usage(
        "microsoft/Phi-4-multimodal-instruct",
        batch_size=4,
        sequence_length=1024,
        use_lora=True
    )
    
    print("Memory usage estimation:")
    for key, value in memory_info.items():
        print(f"  {key}: {value}")
