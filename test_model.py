def generate_with_thoughts(
    model: ContinuousThoughtGPT2,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.7
) -> str:
    """Generate text using the trained continuous thought model, showing the token-by-token generation"""
    model.eval()
    device = model.device
    
    # Tokenize prompt
    inputs = model.tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    with torch.no_grad():
        for i in range(max_new_tokens):
            print(f"\nStep {i+1}:")
            print("Thinking...", end="", flush=True)
            
            # Get model outputs including thoughts
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            print(" Done thinking")
            
            # Get the predictions
            next_token_logits = outputs["logits"][:, -1] / temperature
            
            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Decode and print the chosen token
            next_token_text = model.tokenizer.decode(next_token[0])
            print(f"Generated token: '{next_token_text}'")
            
            # Update input for next iteration
            input_ids = torch.cat([input_ids, next_token], dim=1)
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((1, 1), dtype=torch.long, device=device)
            ], dim=1)
            
            # Optional: Stop if we generate an EOS token
            if next_token[0] == model.tokenizer.eos_token_id:
                print("\nReached end of sequence token")
                break
    
    # Return the full generated text
    return model.tokenizer.decode(input_ids[0], skip_special_tokens=True)

def find_latest_checkpoint():
    """Find the most recent checkpoint file"""
    import glob
    import os
    
    # Look for all checkpoint files
    checkpoints = glob.glob("*.pt")
    if not checkpoints:
        raise FileNotFoundError("No checkpoint files found")
    
    # Get the most recent file based on modification time
    latest = max(checkpoints, key=os.path.getmtime)
    return latest

def test_model():
    """Test the model with interactive prompt generation"""
    print("Loading model...")
    model = ContinuousThoughtGPT2()
    
    # Load latest checkpoint
    latest_checkpoint = find_latest_checkpoint()
    print(f"Loading checkpoint: {latest_checkpoint}")
    model.load_state_dict(torch.load(latest_checkpoint))
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Model loaded and running on {model.device}")
    
    while True:
        # Get user input
        prompt = input("\nEnter a prompt (or 'quit' to exit): ")
        if prompt.lower() == 'quit':
            break
        
        print("\nGenerating with thoughts...")
        generated_text = generate_with_thoughts(
            model=model,
            prompt=prompt,
            max_new_tokens=100,
            temperature=0.7
        )
        
        print("\nFinal generated text:")
        print("-" * 50)
        print(generated_text)
        print("-" * 50)

if __name__ == "__main__":
    test_model() 