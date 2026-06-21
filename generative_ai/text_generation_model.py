from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="gpt2"
)

def generate_text(prompt):

    result = generator(
        prompt,
        max_length=60,
        do_sample=True,
        temperature=0.8,
        top_k=50,
        num_return_sequences=1
    )

    return result[0]["generated_text"]