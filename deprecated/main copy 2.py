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

class ContinuousThoughtGPT2(nn.Module):
    def __init__(
        self,
        model_name: str = "gpt2",
        num_thought_steps: int = 8,
        num_tokens_ahead: int = 4,
        max_length: int = 512
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        # Set pad token to eos token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.num_thought_steps = num_thought_steps
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

    def think_in_latent_space(self, context_hidden: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Modified to return log probabilities of thought trajectories"""
        batch_size, seq_length, hidden_dim = context_hidden.size()
        
        # Initialize thought hidden state
        current_hidden = self.start_thought_embedding.expand(batch_size, hidden_dim)
        
        log_probs = []
        thought_states = []
        
        for _ in range(self.num_thought_steps):
            combined_hidden = torch.cat([context_hidden, current_hidden.unsqueeze(1)], dim=1)
            new_attention_mask = torch.cat([attention_mask, torch.ones((batch_size, 1), device=self.device)], dim=1)
            
            # Get logits for next thought state
            outputs = self.model.transformer(inputs_embeds=combined_hidden, attention_mask=new_attention_mask)
            logits = self.model.lm_head(outputs.last_hidden_state[:, -1, :])
            
            # Sample next thought state
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            thought_token = dist.sample()
            
            # Store log probability for REINFORCE
            log_prob = dist.log_prob(thought_token)
            log_probs.append(log_prob)
            
            # Convert token to embedding
            current_hidden = self.model.transformer.wte(thought_token)
            thought_states.append(current_hidden)
        
        # Combine log probs for entire trajectory
        trajectory_log_prob = torch.stack(log_probs, dim=1).sum(dim=1)
        
        return {
            "final_thought": current_hidden,
            "trajectory_log_prob": trajectory_log_prob,
            "thought_states": thought_states
        }

    def compute_reward(self, predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute reward based on prediction accuracy"""
        predicted_tokens = predictions.argmax(dim=-1)
        correct = (predicted_tokens == labels).float()
        return correct - 0.5  # Center rewards around 0

    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_length = input_ids.shape
        
        # Initial context encoding - can be trained
        outputs = self.model.transformer(input_ids=input_ids, attention_mask=attention_mask)
        context_hidden = outputs.last_hidden_state
        
        # Think and get trajectory log probs - GPT-2 WILL be trained here
        thought_output = self.think_in_latent_space(context_hidden, attention_mask)
        thought_state = thought_output["final_thought"]
        trajectory_log_prob = thought_output["trajectory_log_prob"]
        
        # Final prediction - frozen
        pred_embeddings = thought_state.unsqueeze(1)
        pred_attention_mask = torch.ones((batch_size, 1), device=input_ids.device)
        
        with torch.no_grad():  # Only this part stays frozen
            pred_outputs = self.model(
                inputs_embeds=pred_embeddings, 
                attention_mask=pred_attention_mask
            )
        predictions = pred_outputs.logits.squeeze(1)
        
        # Calculate next token prediction loss
        next_token_loss = None
        if labels is not None:
            target = labels[:, -1]
            next_token_loss = F.cross_entropy(predictions, target)
        
        # Compute REINFORCE loss if training
        loss = None
        if labels is not None:
            target = labels[:, -1]
            rewards = self.compute_reward(predictions, target)
            reinforce_loss = -(trajectory_log_prob * rewards).mean()
            loss = reinforce_loss
        
        return {
            "loss": loss,
            "predictions": predictions,
            "rewards": rewards if labels is not None else None,
            "next_token_loss": next_token_loss
        }

def prepare_data():
    """Load and prepare open-web-math dataset with more controlled processing"""
    dataset = load_dataset("open-web-math/open-web-math")
    
    def format_example(examples):
        formatted_texts = []
        for text, date in zip(examples['text'], examples['date']):
            # Skip empty or very long texts
            if not text or len(text) > 512:
                continue
                
            # Clean and format the text
            cleaned_text = text.strip().replace('\n', ' ').replace('\r', ' ')
            cleaned_text = ' '.join(cleaned_text.split())  # Remove extra whitespace
            
            # Only include if text meets minimum length
            if len(cleaned_text) > 10:
                formatted_text = f"Question: {cleaned_text[:256]} Answer: {date}"  # Stricter length limit
                formatted_texts.append(formatted_text)
        
        return {"text": formatted_texts}
    
    # Format datasets with more controlled processing
    full_dataset = dataset["train"].map(
        format_example,
        batched=True,
        batch_size=100,  # Smaller batch size for processing
        num_proc=2,  # Reduce number of processes
        remove_columns=dataset["train"].column_names,
        filter_fn=lambda x: len(x["text"]) > 0  # Filter out empty results
    )
    
    # Filter out any remaining problematic examples
    full_dataset = full_dataset.filter(lambda x: len(x["text"]) > 0 and len(x["text"]) < 512)
    
    # Take a smaller subset for initial testing
    max_examples = 10000  # Limit dataset size
    if len(full_dataset) > max_examples:
        full_dataset = full_dataset.select(range(max_examples))
    
    # Simple train-test split (90-10)
    train_test_split = full_dataset.train_test_split(test_size=0.1, seed=42)
    
    return train_test_split["train"], train_test_split["test"]

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
    log_interval: int = 100,
    save_interval: int = 1000,
    accumulation_steps: int = 1,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Print initial GPU memory usage
    if torch.cuda.is_available():
        print("\nInitial GPU Memory Usage:")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    
    # Load first batch and check memory
    sample_batch = next(iter(train_dataloader))
    sample_batch = {k: v.to(device, non_blocking=True) for k, v in sample_batch.items()}
    
    # Print GPU memory usage after loading first batch
    if torch.cuda.is_available():
        print("\nGPU Memory After Loading First Batch:")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    
    # Do a sample forward pass and check memory
    with autocast():
        outputs = model(**sample_batch)
    
    # Print GPU memory usage after first forward pass
    if torch.cuda.is_available():
        print("\nGPU Memory After First Forward Pass:")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        
        # Print peak memory stats
        print(f"Max Memory Allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
        print(f"Max Memory Reserved: {torch.cuda.max_memory_reserved() / 1024**2:.2f} MB\n")
    
    # Enable CuDNN benchmark for optimized performance
    torch.backends.cudnn.benchmark = True
    
    # Initialize GradScaler for mixed-precision training
    scaler = GradScaler()
    
    # Now optimize ALL parameters except final prediction layer
    optimizer = torch.optim.AdamW([
        # Thought embeddings
        {"params": [model.start_thought_embedding, model.end_thought_embedding], "lr": learning_rate * 10},
        # GPT-2 parameters
        {"params": model.model.transformer.parameters(), "lr": learning_rate}
    ])
    
    wandb.init(project="coconut-gpt2-reinforce", config={
        "model": "gpt2",
        "num_thought_steps": model.num_thought_steps,
        "learning_rate": learning_rate,
        "batch_size": train_dataloader.batch_size,
        "num_epochs": num_epochs,
    })
    
    global_step = 0
    best_reward = float("-inf")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_reward = 0
        total_next_token_loss = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
        optimizer.zero_grad()
        
        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            
            with autocast():
                # Forward pass with REINFORCE
                outputs = model(**batch)
                loss = outputs["loss"]
                rewards = outputs["rewards"]
                next_token_loss = outputs["next_token_loss"]
                loss = loss / accumulation_steps  # Normalize loss
                
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            if (step + 1) % accumulation_steps == 0:
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.model.transformer.parameters(), 1.0)
                
                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1
            
            # Logging
            total_loss += loss.item() * accumulation_steps
            total_reward += rewards.mean().item()
            total_next_token_loss += next_token_loss.item()
            
            if (step + 1) % log_interval == 0:
                avg_loss = total_loss / log_interval
                avg_reward = total_reward / log_interval
                avg_next_token_loss = total_next_token_loss / log_interval
                wandb.log({
                    "reinforce_loss": avg_loss,
                    "average_reward": avg_reward,
                    "next_token_loss": avg_next_token_loss,
                    "epoch": epoch,
                    "global_step": global_step
                })
                total_loss = 0
                total_reward = 0
                total_next_token_loss = 0
            
            # Save best model based on rewards
            if (step + 1) % save_interval == 0:
                model.eval()
                test_reward = 0
                with torch.no_grad():
                    for test_batch in test_dataloader:
                        test_batch = {k: v.to(device, non_blocking=True) for k, v in test_batch.items()}
                        outputs = model(**test_batch)
                        test_reward += outputs["rewards"].mean().item()
                
                avg_test_reward = test_reward / len(test_dataloader)
                wandb.log({"test_reward": avg_test_reward})
                
                if avg_test_reward > best_reward:
                    best_reward = avg_test_reward
                    torch.save(model.state_dict(), f"coconut_gpt2_reinforce_best.pt")
                
                model.train()
        
        # Optional: Evaluate at the end of each epoch
        # ... [Optional evaluation code] ...
    
    wandb.finish()

if __name__ == "__main__":
    # Initialize model with smaller max_length initially
    model = ContinuousThoughtGPT2(max_length=256)
    
    # Load and prepare data
    train_dataset, test_dataset = prepare_data()
    
    # Create dataloaders with simplified parameters
    train_data = GSM8kDataset(train_dataset, model.tokenizer, model.max_length)
    test_data = GSM8kDataset(test_dataset, model.tokenizer, model.max_length)
    
    train_loader = DataLoader(
        train_data, 
        batch_size=2,
        shuffle=True, 
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    test_loader = DataLoader(
        test_data, 
        batch_size=2,
        shuffle=False, 
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Train model
    train(model, train_loader, test_loader, num_epochs=3, learning_rate=1e-5)