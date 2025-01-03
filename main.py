import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import TextDataset, DataCollatorForLanguageModeling
from datasets import load_dataset
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, Optional
import torch.cuda as cuda

class ContinuousThoughtGPT2(nn.Module):
    def __init__(
        self,
        model_name: str = "gpt2",
        num_independent_thoughts: int = 4,
        num_tokens_ahead: int = 4,
        max_length: int = 128
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        # Set pad token to eos token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.num_independent_thoughts = num_independent_thoughts
        self.num_tokens_ahead = num_tokens_ahead
        self.max_length = max_length
        
        # Add special tokens for thoughts
        special_tokens = {"additional_special_tokens": ["<|startofthought|>", "<|endofthought|>"]}
        self.tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Get IDs for special tokens
        self.start_thought_token_id = self.tokenizer.convert_tokens_to_ids("<|startofthought|>")
        self.end_thought_token_id = self.tokenizer.convert_tokens_to_ids("<|endofthought|>")
        
        # Hidden dimension from GPT-2 config
        self.hidden_dim = self.model.config.n_embd
        
        # Learnable start/end thought embeddings
        self.start_thought_embedding = nn.Parameter(torch.randn(1, self.hidden_dim).to(self.device))
        self.end_thought_embedding = nn.Parameter(torch.randn(1, self.hidden_dim).to(self.device))

        # Modified mixing head to handle correct dimensions
        self.mixing_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),  # First reduce dimension
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Sigmoid()
        ).to(self.device)

    def think_in_latent_space(
        self,
        context_hidden: torch.Tensor,  # [batch_size, 1, hidden_dim]
        attention_mask: torch.Tensor,
        true_future_tokens: torch.Tensor,  # [batch_size, num_future]
        num_future_tokens: int = 4,
        thought_length: int = 2
    ) -> Dict[str, torch.Tensor]:
        """Generate multiple independent thoughts and their predictions"""
        batch_size = context_hidden.size(0)
        
        # Initialize storage for multiple thoughts
        thoughts = []
        log_probs = []
        thought_predictions = []  # Store predictions for each thought
        
        # Generate multiple independent thoughts
        for _ in range(self.num_independent_thoughts):
            # Start each thought fresh from context
            current_hidden = self.start_thought_embedding.expand(batch_size, 1, -1)
            thought_tokens = []
            thought_log_probs = []
            
            # Generate sequential tokens for this thought
            for step in range(thought_length):
                combined_hidden = torch.cat([context_hidden, current_hidden], dim=1)
                
                # Create proper attention mask for combined sequence
                combined_attention_mask = torch.ones(
                    (batch_size, combined_hidden.size(1)),
                    dtype=torch.bool,
                    device=self.device
                )
                
                outputs = self.model.transformer(
                    inputs_embeds=combined_hidden,
                    attention_mask=combined_attention_mask
                )
                logits = self.model.lm_head(outputs.last_hidden_state[:, -1, :])
                
                # Sample next token
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                next_token = dist.sample()
                
                log_prob = dist.log_prob(next_token)
                thought_log_probs.append(log_prob)
                
                next_embedding = self.model.transformer.wte(next_token)
                current_hidden = torch.cat([current_hidden, next_embedding.unsqueeze(1)], dim=1)
                thought_tokens.append(next_token)
            
            # Generate predictions using this thought's final state
            future_predictions = []
            pred_hidden = current_hidden[:, -1:]  # Use final thought state
            
            # Generate future predictions for this specific thought
            for i in range(num_future_tokens):
                pred_outputs = self.model(inputs_embeds=pred_hidden)
                pred = pred_outputs.logits.squeeze(1)
                future_predictions.append(pred)
                
                # Teacher forcing
                if i < num_future_tokens - 1:
                    next_hidden = self.model.transformer.wte(true_future_tokens[:, i])
                    pred_hidden = next_hidden.unsqueeze(1)
            
            # Store this thought's results
            thoughts.append(current_hidden)
            log_probs.append(torch.stack(thought_log_probs, dim=1))
            thought_predictions.append(torch.stack(future_predictions, dim=1))
        
        return {
            "thoughts": thoughts,
            "log_probs": torch.stack(log_probs, dim=1),  # [batch_size, num_thoughts, thought_length]
            "predictions": torch.stack(thought_predictions, dim=1)  # [batch_size, num_thoughts, num_future, vocab_size]
        }

    def compute_reward(
        self,
        predictions: torch.Tensor,  # [batch_size, num_thoughts, num_future, vocab_size]
        labels: torch.Tensor,  # [batch_size, num_future]
    ) -> torch.Tensor:
        """Compute reward for each thought based on its prediction accuracy"""
        # Get log probs of predictions for each thought
        log_probs = F.log_softmax(predictions, dim=-1)
        
        # Expand labels for gathering
        labels_expanded = labels.unsqueeze(1).unsqueeze(-1)  # [batch_size, 1, num_future, 1]
        
        # Get log prob of correct tokens for each thought
        true_log_probs = torch.gather(
            log_probs,
            -1,
            labels_expanded.expand(-1, predictions.size(1), -1, -1)
        ).squeeze(-1)  # [batch_size, num_thoughts, num_future]
        
        # Compute reward as sum of log probs across future tokens
        rewards = true_log_probs.sum(dim=-1)  # [batch_size, num_thoughts]
        
        # Add minimum reward for exploration
        rewards = torch.maximum(rewards, torch.tensor(0.1).to(rewards.device))
        
        return rewards  # [batch_size, num_thoughts]

    def compute_mixed_predictions(self, base_hidden, thought_hidden):
        # Concatenate hidden states for mixing decision
        combined = torch.cat([base_hidden, thought_hidden], dim=-1)
        mixing_weight = self.mixing_head(combined)
        
        return mixing_weight

    def compute_future_predictions(self, thought_hidden, true_future_tokens, num_future=4):
        predictions = []
        current_hidden = thought_hidden
        
        for i in range(num_future):
            # Predict next token
            outputs = self.model.transformer(inputs_embeds=current_hidden.unsqueeze(1))
            pred = self.model.lm_head(outputs.last_hidden_state[:, -1, :])
            predictions.append(pred)
            
            # Use true token for next step (teacher forcing)
            if i < num_future - 1:
                current_hidden = self.model.transformer.wte(true_future_tokens[:, i])
        
        return torch.stack(predictions, dim=1)

    def _create_parallel_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """Create attention mask for parallel thought generation."""
        batch_size, seq_length = attention_mask.shape
        
        # Create causal mask for thought tokens
        thought_mask = torch.tril(torch.ones((batch_size, seq_length, seq_length), device=attention_mask.device))
        
        # Allow thoughts to attend to all previous text tokens
        text_to_thought_mask = torch.ones((batch_size, seq_length, seq_length), device=attention_mask.device)
        
        # Expand attention mask for broadcasting
        expanded_attention_mask = attention_mask.unsqueeze(-1).expand(-1, -1, seq_length)
        
        # Combine masks
        combined_mask = torch.cat([
            torch.cat([expanded_attention_mask, text_to_thought_mask], dim=1),
            torch.cat([torch.zeros_like(thought_mask), thought_mask], dim=1)
        ], dim=1)
        
        return combined_mask

    def forward(self, input_ids, attention_mask, labels=None):
        batch_size, seq_length = input_ids.shape
        
        # Cap maximum sequence length and increase chunk size
        max_seq_chunk = 64
        
        # Initialize storage
        total_reinforce_loss = 0
        all_rewards = []
        all_mixing_weights = []
        
        for chunk_start in range(0, seq_length - self.num_tokens_ahead, max_seq_chunk):
            chunk_end = min(chunk_start + max_seq_chunk, seq_length - self.num_tokens_ahead)
            
            # Get chunk context
            chunk_ids = input_ids[:, :chunk_end+1]
            chunk_mask = attention_mask[:, :chunk_end+1]
            chunk_labels = labels[:, chunk_start+1:chunk_end+1+self.num_tokens_ahead]
            
            # Get context encodings for entire chunk
            with torch.no_grad():  # Don't compute gradients for base model encoding
                context_outputs = self.model.transformer(
                    input_ids=chunk_ids,
                    attention_mask=chunk_mask,
                    output_hidden_states=True
                )
                context_hidden = context_outputs.last_hidden_state[:, chunk_start:chunk_end]
            
            # Process positions in chunk
            chunk_future_preds = []
            
            for pos in range(chunk_end - chunk_start):
                # Generate thoughts and predictions
                thought_output = self.think_in_latent_space(
                    context_hidden[:, pos].unsqueeze(1),
                    chunk_mask[:, :chunk_start+pos+1],
                    labels[:, chunk_start+pos:chunk_start+pos+self.num_tokens_ahead],
                    num_future_tokens=self.num_tokens_ahead
                )
                
                future_predictions = thought_output["predictions"]  # [batch_size, num_thoughts, num_future, vocab_size]
                log_probs = thought_output["log_probs"]  # Thought generation log probs
                
                # Get labels for current position
                pos_labels = chunk_labels[:, pos:pos+self.num_tokens_ahead]
                
                # Calculate next token prediction accuracy (for reward computation)
                with torch.no_grad():
                    # Get log probs of predictions
                    pred_log_probs = F.log_softmax(future_predictions, dim=-1)  # [batch_size, num_thoughts, num_future, vocab_size]
                    
                    # Expand labels for gathering
                    expanded_labels = pos_labels.unsqueeze(1).unsqueeze(-1)  # [batch_size, 1, num_future, 1]
                    expanded_labels = expanded_labels.expand(-1, future_predictions.size(1), -1, -1)
                    
                    # Get log prob of correct tokens
                    true_log_probs = torch.gather(
                        pred_log_probs,
                        -1,
                        expanded_labels
                    ).squeeze(-1)  # [batch_size, num_thoughts, num_future]
                    
                    # Compute reward based on prediction accuracy
                    rewards = true_log_probs.sum(dim=-1) * 10.0  # Scale rewards [batch_size, num_thoughts]
                    rewards = torch.maximum(rewards, torch.tensor(0.1).to(rewards.device))  # Minimum reward
                
                all_rewards.append(rewards)
                
                # Compute REINFORCE loss using rewards
                reinforce_loss = -(log_probs * rewards.unsqueeze(-1).detach()).mean()
                total_reinforce_loss += reinforce_loss
                
                # Track mixing weights (but don't use for loss)
                mixing_weight = self.mixing_head(context_hidden[:, pos])
                all_mixing_weights.append(mixing_weight.detach())
                
                del thought_output
            
            del context_outputs, context_hidden
            torch.cuda.empty_cache()
        
        # Average loss
        num_positions = seq_length - self.num_tokens_ahead
        avg_reinforce_loss = total_reinforce_loss / num_positions
        avg_mixing_weight = torch.stack(all_mixing_weights, dim=1).mean() if all_mixing_weights else torch.tensor(0.0)
        
        return {
            "loss": avg_reinforce_loss,  # Only REINFORCE loss
            "reinforce_loss": avg_reinforce_loss,
            "rewards": torch.stack(all_rewards, dim=1) if all_rewards else torch.tensor(0.0),
            "mixing_weight": avg_mixing_weight,
        }

def prepare_data():
    """Load and prepare Open-Web-Math dataset"""
    dataset = load_dataset("Alignment-Lab-AI/Open-Web-Math", split='train')
    
    # Take first 100k examples
    dataset = dataset.select(range(300000))
    
    def format_example(example):
        # Just use the text directly
        return {
            "text": example['text']
        }
    
    # Format dataset and create train/test split
    formatted_dataset = dataset.map(format_example)
    
    # Split into train (90%) and test (10%) sets
    train_test_split = formatted_dataset.train_test_split(test_size=0.1)
    
    return train_test_split['train'], train_test_split['test']

class GSM8kDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer, max_length):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        example = self.dataset[idx]["text"]
        
        # Tokenize with padding and truncation
        encodings = self.tokenizer(
            example,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encodings["input_ids"][0],
            "attention_mask": encodings["attention_mask"][0],
            "labels": encodings["input_ids"][0].clone()  # Use same sequence for labels
        }

def train(
    model: ContinuousThoughtGPT2,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    num_epochs: int = 1,
    learning_rate: float = 1e-5,
    log_interval: int = 5,
    save_interval: int = 100,
    checkpoint_interval: int = 100,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Print initial memory usage
    if torch.cuda.is_available():
        print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"Initial GPU memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    
    # Separate optimizers for thought embeddings and LM parameters
    thought_optimizer = torch.optim.AdamW([
        model.start_thought_embedding,
        model.end_thought_embedding
    ], lr=learning_rate * 10)
    
    lm_optimizer = torch.optim.AdamW(
        model.model.transformer.parameters(),
        lr=learning_rate
    )
    
    wandb.init(project="coconut-gpt2-reinforce", config={
        "model": "gpt2",
        "num_thought_steps": model.num_independent_thoughts,
        "learning_rate": learning_rate,
    })
    
    global_step = 0
    best_reward = float("-inf")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_reward = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass with REINFORCE
            outputs = model(**batch)
            loss = outputs["loss"]
            rewards = outputs["rewards"]
            mixing_weight = outputs["mixing_weight"]
            
            # Update thought embeddings with higher learning rate
            thought_optimizer.zero_grad()
            lm_optimizer.zero_grad()
            
            loss.backward()
            
            # Clip gradients separately
            torch.nn.utils.clip_grad_norm_(
                [model.start_thought_embedding, model.end_thought_embedding],
                1.0
            )
            torch.nn.utils.clip_grad_norm_(
                model.model.transformer.parameters(),
                1.0
            )
            
            thought_optimizer.step()
            lm_optimizer.step()
            
            # Logging
            total_loss += loss.item()
            total_reward += rewards.mean().item()
            
            if (step + 1) % log_interval == 0:
                avg_loss = total_loss / log_interval
                avg_reward = total_reward / log_interval
                wandb.log({
                    "reinforce_loss": avg_loss,
                    "average_reward": avg_reward,
                    "mixing_weight": mixing_weight.item(),
                    "epoch": epoch,
                    "global_step": global_step
                })
                total_loss = 0
                total_reward = 0
                
                # Print memory usage
                if torch.cuda.is_available():
                    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                    print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
            
            # Save regular checkpoint
            if (step + 1) % checkpoint_interval == 0:
                checkpoint_path = f"coconut_gpt2_reinforce_checkpoint_{global_step}.pt"
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_reward': best_reward,
                }, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
            
            # Save best model based on rewards
            if (step + 1) % save_interval == 0:
                model.eval()
                test_reward = 0
                with torch.no_grad():
                    for test_batch in test_dataloader:
                        test_batch = {k: v.to(device) for k, v in test_batch.items()}
                        outputs = model(**test_batch)
                        test_reward += outputs["rewards"].mean().item()
                
                avg_test_reward = test_reward / len(test_dataloader)
                wandb.log({"test_reward": avg_test_reward})
                
                if avg_test_reward > best_reward:
                    best_reward = avg_test_reward
                    torch.save({
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_reward': best_reward,
                    }, "coconut_gpt2_reinforce_best.pt")
                
                model.train()
            
            global_step += 1
    
    wandb.finish()

def get_latest_checkpoint() -> Optional[str]:
    """Find the most recent checkpoint file"""
    import glob
    import os
    
    checkpoint_files = glob.glob("coconut_gpt2_reinforce_*.pt")
    if not checkpoint_files:
        return None
        
    # Sort by modification time, most recent first
    latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
    return latest_checkpoint

if __name__ == "__main__":
    # Initialize model
    model = ContinuousThoughtGPT2()
    
    # Try to load latest checkpoint
    # latest_checkpoint = get_latest_checkpoint()
    # if latest_checkpoint:
    #     print(f"Loading checkpoint: {latest_checkpoint}")
    #     model.load_state_dict(torch.load(latest_checkpoint))
    # else:
    #     print("No checkpoint found, starting fresh training")
    
    # Load and prepare data
    train_dataset, test_dataset = prepare_data()
    
    # Create dataloaders
    train_data = GSM8kDataset(train_dataset, model.tokenizer, model.max_length)
    test_data = GSM8kDataset(test_dataset, model.tokenizer, model.max_length)
    
    train_loader = DataLoader(train_data, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)
    
    # Train model
    train(model, train_loader, test_loader)