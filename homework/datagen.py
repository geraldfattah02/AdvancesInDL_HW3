def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):
    from .cot import CoTModel
    from .data import Dataset
    from .base_llm import BaseLLM

    model = CoTModel("HuggingFaceTB/SmolLM2-1.7B-Instruct")
    dataset = Dataset("train")

    results = []

    questions = [dataset[i] for i in range(len(dataset))]
    prompts = [model.format_prompt(q) for q, _ in questions]

    print(f"Generating {oversample} completions for {len(prompts)} questions...")

    # Batched generate: returns list[list[str]]
    all_completions = model.batched_generate(prompts, num_return_sequences=oversample, temperature=temperature)

    for (question, true_answer), completions in zip(questions, all_completions):
        for completion in completions:
            parsed = model.parse_answer(completion)
            # Accept if within 1% relative tolerance (or exact for small values)
            if abs(true_answer) < 1e-6:
                correct = abs(parsed) < 1e-6
            else:
                correct = abs(parsed - true_answer) / abs(true_answer) < 0.01

            if correct:
                results.append([question, true_answer, completion.strip()])
                break  # take first correct completion, move to next question

    print(f"Dataset size: {len(results)} / {len(questions)} questions answered correctly ({100*len(results)/len(questions):.1f}%)")

    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved to {output_path}")

if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
