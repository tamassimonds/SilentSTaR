import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import wandb
import numpy as np

@dataclass
class SilentStarConfig:
    num_thought_tokens: int = 8  # Number of continuous thought steps
    thought_dim: int = 768  # GPT-2 hidden dim
    max_length: int = 512
    learning_rate: float = 1e-5
    thought_learning_rate: float = 1e-4
    warmup_steps: int = 100
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    
class SilentStarGPT2(nn.Module):
    def __init__(self, model_name: str = "gpt2", config: Optional[SilentStarConfig] = None):
        super().__init__()
        self.config = config or SilentStarConfig()
        
        # Load base model and tokenizer
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        
        # Add special tokens
        special_tokens = {"additional_special_tokens": ["<bot>", "<eot>"]}
        self.tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Initialize learnable continuous thought embeddings
        self.start_thought = nn.Parameter(torch.randn(1, self.config.thought_dim))
        self.end_thought = nn.Parameter(torch.randn(1, self.config.thought_dim))
        
        # Mixing head to combine base and thought predictions
        self.mixing_head = nn.Sequential(
            nn.Linear(2 * self.config.thought_dim, self.config.thought_dim),
            nn.GELU(),
            nn.Linear(self.config.thought_dim, 1),
            nn.Sigmoid()
        )
        
        # Value head for thought evaluation
        self.value_head = nn.Sequential(
            nn.Linear(self.config.thought_dim, self.config.thought_dim // 2),
            nn.GELU(),
            nn.Linear(self.config.thought_dim // 2, 1)
        )

    def generate_continuous_thoughts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Generate sequence of continuous thoughts from hidden states"""
        batch_size = hidden_states.size(0)
        thoughts = []
        
        # Start with learned start token
        current = self.start_thought.expand(batch_size, -1)
        thoughts.append(current)
        
        # Generate sequence of thoughts
        for _ in range(self.config.num_thought_tokens):
            # Use model to transform current thought
            outputs = self.model(inputs_embeds=current.unsqueeze(1))
            next_thought = outputs.last_hidden_state[:, -1]
            thoughts.append(next_thought)
            current = next_thought
            
        # Add end token
        thoughts.append(self.end_thought.expand(batch_size, -1))
        
        return torch.stack(thoughts, dim=1)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Get base model outputs
        base_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        base_logits = base_outputs.logits
        base_hidden = base_outputs.hidden_states[-1]
        
        # Generate continuous thoughts
        thoughts = self.generate_continuous_thoughts(base_hidden)
        
        # Get predictions with thoughts
        thought_mask = torch.ones(
            thoughts.size(0),
            thoughts.size(1),
            dtype=torch.bool,
            device=thoughts.device
        )
        
        thought_outputs = self.model(
            inputs_embeds=thoughts,
            attention_mask=thought_mask
        )
        thought_hidden = thought_outputs.hidden_states[-1]
        thought_logits = thought_outputs.logits
        
        # Compute mixing weights
        mix_weights = self.mixing_head(
            torch.cat([base_hidden, thought_hidden], dim=-1)
        )
        
        # Combine predictions
        final_logits = mix_weights * base_logits + (1 - mix_weights) * thought_logits
        
        # Compute thought values for REINFORCE
        thought_values = self.value_head(thoughts).squeeze(-1)
        
        outputs = {
            "logits": final_logits,
            "thought_values": thought_values,
            "mix_weights": mix_weights,
            "thoughts": thoughts
        }
        
        if labels is not None:
            # Compute losses
            outputs["loss"] = self.compute_losses(outputs, labels)
            
        return outputs
    
    def compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor
    ) -> torch.Tensor:
        # Language modeling loss
        lm_loss = F.cross_entropy(
            outputs["logits"].view(-1, outputs["logits"].size(-1)),
            labels.view(-1)
        )
        
        # Compute rewards for REINFORCE
        with torch.no_grad():
            base_loss = F.cross_entropy(
                outputs["logits"].view(-1, outputs["logits"].size(-1)),
                labels.view(-1),
                reduction='none'
            )
            thought_loss = F.cross_entropy(
                outputs["thought_logits"].view(-1, outputs["thought_logits"].size(-1)),
                labels.view(-1),
                reduction='none'
            )
            rewards = base_loss - thought_loss
            
        # REINFORCE loss with baseline
        policy_loss = -torch.mean(
            (rewards - outputs["thought_values"]) * outputs["thought_logits"]
        )
        
        # Value prediction loss
        value_loss = F.mse_loss(outputs["thought_values"], rewards.detach())
        
        return lm_loss + policy_loss + value_loss

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer
    ) -> float:
        """Single training step"""
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        
        loss = outputs["loss"]
        loss = loss / self.config.gradient_accumulation_steps
        loss.backward()
        
        if batch["is_last_step"]:
            torch.nn.utils.clip_grad_norm_(
                self.parameters(),
                self.config.max_grad_norm
            )
            optimizer.step()
            optimizer.zero_grad()
            
        return loss.item()

def train(
    model: SilentStarGPT2,
    train_dataloader: torch.utils.data.DataLoader,
    num_epochs: int = 3,
):
    """Training loop"""
    model.train()
    optimizer = torch.optim.AdamW([
        {'params': model.parameters(), 'lr': model.config.learning_rate},
        {'params': [model.start_thought, model.end_thought],
         'lr': model.config.thought_learning_rate}
    ])
    
    # Initialize weights & biases
    wandb.init(project="silent-star", config=model.config.__dict__)
    
    global_step = 0
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_idx, batch in enumerate(train_dataloader):
            # Add is_last_step flag
            batch["is_last_step"] = (
                batch_idx + 1) % model.config.gradient_accumulation_steps == 0
            
            # Training step
            loss = model.train_step(batch, optimizer)
            epoch_loss += loss
            
            if batch["is_last_step"]:
                global_step += 1
                
                # Log metrics
                wandb.log({
                    "loss": loss,
                    "epoch": epoch,
                    "global_step": global_step
                })
                
        # Log epoch metrics
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch {epoch}: Average Loss = {avg_epoch_loss:.4f}")
        wandb.log({
            "epoch_loss": avg_epoch_loss,
            "epoch": epoch
        })
        
    wandb.finish()

if __name__ == "__main__":
    # Initialize model
    model = SilentStarGPT2()
    
    # Example usage:
    # train(model, train_dataloader)