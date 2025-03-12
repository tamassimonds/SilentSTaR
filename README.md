# SilentSTaR

This project implements a novel approach to language model generation that combines concepts from the QuietSTaR paper with continuous latent space reasoning. Instead of generating tokens, the model operates directly in the latent space before each token prediction. It's trained in a similar way as Quiet-STaR where models are trained to think before answering to maximise next token accuracy 

## Core Concept

Traditional language models generate tokens autoregressively, predicting one token at a time based on the previous context. This project enhances that process by adding a continuous thought mechanism before each token generation, following the key insight from the Quiet-STaR paper that language models can benefit from explicit reasoning steps before token prediction.

## Training Process

### Latent Reasoning Training
Similar to Quiet-STaR, our model is trained to reason in latent space before generating each token. The key differences are:

1. **Continuous vs Discrete Thoughts**
   - Quiet-STaR: Generates discrete tokens
   - Our model: Operates entirely in continuous latent space

2. **Per-Token Training**
   - Before each token prediction, the model performs multiple thought steps in latent space
   - The model learns to use these thought steps to improve token prediction accuracy
   - REINFORCE is used to train the thought generation process

3. **Training Signal**
   - The model receives feedback based on the quality of each token prediction
   - If the thought process leads to better token predictions, it's reinforced
   - A value head provides a baseline to reduce variance in the REINFORCE training

### Training Components

1. **Thought Generation Loss**
   ```
   For each token t:
   1. Generate n thought steps in latent space
   2. Use final thought state to predict token
   3. Compare prediction to actual token
   4. Update thought process using REINFORCE
   ```

2. **Value Head Training**
   - Learns to predict the expected reward for thought trajectories
   - Helps stabilize the REINFORCE training process
   - Trained using MSE loss against actual rewards

3. **Language Model Loss**
   - Standard next-token prediction loss
   - Combined with the thought process training
   - Helps maintain language generation capabilities

### Reward Structure
- Primary reward: Accuracy of token prediction after thinking
- Baseline: Value head prediction
- The difference (reward - baseline) guides the thought process training

## Model Architecture

The model builds on GPT-2 with several key additions:
- Learnable start/end thought embeddings
- A thought generation mechanism in latent space
- Value head for thought evaluation
- Modified forward pass that incorporates continuous thoughts

## Usage

To test the model with visible thinking steps:

```python
python test_model.py
```

This will:
1. Load the most recent checkpoint
2. Accept user prompts
3. Show the thinking process before each token generation
4. Display the final generated text

Example output:
Step 1:
Thinking... Done thinking
Generated token: 'The'
Step 2:
Thinking... Done thinking
Generated token: 'cat'


## Technical Implementation

### Thought Generation Process
```python
def think_in_latent_space(self, context_hidden: torch.Tensor, attention_mask: torch.Tensor):
    """
    Performs multiple thought steps in latent space before token prediction.
    Similar to Quiet-STaR's thinking process, but continuous rather than discrete.
    """
    # Initialize thought state from learned embedding
    current_hidden = self.start_thought_embedding
    
    # Multiple thought steps
    for _ in range(self.num_thought_steps):
        # Update thought state using context
        current_hidden = self.thought_step(current_hidden, context_hidden)
    
    return current_hidden
```

### Training Loop
```python
# For each token in sequence:
1. Generate thoughts in latent space
2. Predict next token using thought state
3. Compute reward (prediction accuracy)
4. Update thought process using REINFORCE
5. Update value head and language model
```

## References

- QuietSTaR Paper: [Link to paper]
- Related work on latent reasoning in language models
- GPT-2 architecture and implementation details

## Requirements

- PyTorch
- Transformers library
- CUDA-capable GPU (recommended)

Step 1:
Thinking... Done thinking
Generated token: 'The'
Step 2:
Thinking... Done thinking
Generated token: 'cat'
