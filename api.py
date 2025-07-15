from flask import Flask
from transformers.pipelines import pipeline
import torch
import hashlib
import time

app = Flask(__name__)

# Simple cache for faster repeated requests
cache = {}
CACHE_TTL = 3  # 5 minutes

# Load the model once when the app starts (major speed improvement!)
print("Loading AI model... (this happens only once)")
device = 0 if torch.cuda.is_available() else -1  # Use GPU if available
generator = pipeline(
    'text-generation', 
    model='distilgpt2',  # Smaller, faster model than gpt2
    device=device,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32  # Half precision for speed
)
print("Model loaded successfully!")

def generate_ai_garbage_text(prompt):    
    # Optimized generation parameters for speed
    generated_text = generator(
        prompt, 
        max_length=len(prompt.split()) + 8,  # Dynamic length based on prompt
        num_return_sequences=1,
        do_sample=True,
        temperature=0.8,  # Lower temperature for faster, more focused generation
        truncation=True,
        max_new_tokens=12  # Limit new tokens for speed
    )
    
    result = generated_text[0]['generated_text']
    print(f"Generated: {result}")
    return result

@app.route("/")
def hello_world():
    prompt = "Skibidi, toilet, my dear lad"

    x = generate_ai_garbage_text(prompt)
    return x


@app.route("/batman")
def batman():
    prompt = "Batman, it's nothing personal, skibidi"

    x = generate_ai_garbage_text(prompt)
    return x

@app.route("/Inuit")
def Inuit():
    prompt = "Inuit village but they are all giant 8 legged behemoths marching across the frozen tundra"

    x = generate_ai_garbage_text(prompt)
    return x

    



if __name__ == "__main__":
    app.run(debug=True, host="10.100.16.51", port=5000)