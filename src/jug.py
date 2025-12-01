from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import torch

model_dir = "lora-chat-out"
base_model_name = "HuggingFaceH4/zephyr-7b-beta"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
)

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",
    quantization_config=quant_config,             
    trust_remote_code=True,
)

# Load the LoRA adapter on top of the base model
model = PeftModel.from_pretrained(model, model_dir)

# This creates a single merged model, but it's irreversible!! Only use for final build
# model = model.merge_and_unload()

model.eval()


""" conversation = ""

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit", "bye"]:
        break

    conversation += f"User: {user_input}\nBot:"
    response = generate_response(conversation)
    latest_response = response.split("Bot:")[-1].strip()
    print("Bot:", latest_response)
    conversation += " " + latest_response """

def format_prompt(user_input):
    """
    Format the prompt in the same way you trained your model.
    This should match the format_chat function from your training.
    """
    # If you used Zephyr's chat template during training
    if tokenizer.chat_template:
        messages = [
            {"role": "user", "content": user_input}
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False)
    else:
        # Use the format you trained with (from your training code)
        return f"<|user|>\n{user_input}</s>\n<|assistant|>\n"

def generate_response(prompt):
    # Format the prompt
    formatted_prompt = format_prompt(prompt)
    
    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
    
    # Generate response
    with torch.no_grad():  # Disable gradient calculation for inference
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode and extract only the new response (after the prompt)
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant's response
    # This depends on how you formatted your prompts during training
    if "<|assistant|>" in full_response:
        # Extract everything after "<|assistant|>"
        response = full_response.split("<|assistant|>")[-1].strip()
    else:
        # If using a different format, extract after the last user message
        response = full_response[len(formatted_prompt):].strip()
    
    return response

# Chat loop
conversation_history = []

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit", "bye"]:
        break
    
    # Generate response
    response = generate_response(user_input)
    print("Bot:", response)
    
    # Optional: Keep conversation history for context
    conversation_history.append(f"User: {user_input}")
    conversation_history.append(f"Assistant: {response}")
    
    # If you want to use full conversation history:
    # combined_prompt = "\n".join(conversation_history[-6:])  # Last 3 exchanges
    # response = generate_response(combined_prompt)