import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    model_name = "Qwen/Qwen2-1.5B"
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    hidden_size = model.config.hidden_size
    steering_vec = torch.randn(hidden_size)

    def steering_pre_hook(module, inputs):
        hidden_states = inputs[0] 

        v = steering_vec.to(hidden_states.device, hidden_states.dtype)
        # reshape to [1, 1, hidden_size] so it broadcasts over batch and seq_len
        v = v.view(1, 1, -1)

        hidden_states = hidden_states + v  # additive steering

        # Return the new inputs tuple with modified hidden_states
        new_inputs = (hidden_states, *inputs[1:])
        return new_inputs

    layer_idx = 0
    layer = model.model.layers[layer_idx]
    layer.register_forward_pre_hook(steering_pre_hook)
    print(f"Registered steering hook on layer {layer_idx}")

    prompt = "The meaning of life is" # random test prompt
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,      
            temperature=0.8,       
            top_p=0.9,
        )

    print("output_ids shape:", output_ids.shape)
    print("output_ids:", output_ids[0].tolist())

    text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    print("\n=== RAW decoded text (repr) ===")
    print(repr(text))

    text_clean = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("\n=== Clean decoded text ===")
    print(text_clean)


if __name__ == "__main__":
    main()