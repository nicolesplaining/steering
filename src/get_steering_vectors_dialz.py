import os
from transformers import AutoTokenizer
from dialz import Dataset, SteeringModel, SteeringVector, get_activation_score, visualize_activation


model_name = "mistralai/Mistral-7B-Instruct-v0.1" # random model rn js testing
dataset = Dataset.load_dataset(model_name, 'stereoset-race')

model = SteeringModel(model_name, layer_ids=list(range(10,20)), token=os.getenv("HF_TOKEN"))
vector = SteeringVector.train(model, dataset) #training vector

tokenizer = AutoTokenizer.from_pretrained(model_name, token=model.token)
tokenizer.pad_token_id = 0

def generate_output(input_text):
    messages = [
        {"role": "user", "content": input_text}
    ]

    chat_input = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                return_tensors="pt",)

    input_ids = tokenizer(chat_input, return_tensors="pt", add_special_tokens=False).to(model.device)

    settings = {
        "pad_token_id": tokenizer.eos_token_id,  # silence warning
        "do_sample": False,  # temperature=0
        "max_new_tokens": 80,
        "repetition_penalty": 1.5,
        "return_dict_in_generate": True
    }

    generated_outputs = model.generate(**input_ids, **settings)

    new_tokens = generated_outputs.sequences[0, input_ids["input_ids"].size(1):]

    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


input = "Write a story about a young girl."
steering_factor = 1

print("Baseline response:")
model.reset()
print(generate_output(input))

print(f"\nSteering of +{steering_factor}:")
model.set_control(vector, steering_factor)
print(generate_output(input))