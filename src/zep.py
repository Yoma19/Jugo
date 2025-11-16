from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = "HuggingFaceH4/zephyr-7b-beta"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=quant_config,             
)

# prompt = "How many languages do you know?"
# inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
# outputs = model.generate(**inputs, max_new_tokens=128)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit", "bye"]:
        break
    response = generate_response(user_input)
    print("Bot:", response)
