from transformers import AutoModelForCausalLM, AutoTokenizer
from steering_vectors import train_steering_vector

model = AutoModelForCausalLM.from_pretrained("gpt2") #rando model for now also
tokenizer = AutoTokenizer.from_pretrained("gpt2")

training_samples = [
    (
        "The capital of England is London",
        "The capital of England is Beijing"
    ),
    (
        "The capital of France is Paris",
        "The capital of France is Berlin"
    )
    # ...
]

steering_vector = train_steering_vector(
    model,
    tokenizer,
    training_samples,
    show_progress=True,
    layers=[1, 2, 3]
)

with steering_vector.apply(model):
    prompt = "Is it true that crystals have magic healing properties?"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs)

