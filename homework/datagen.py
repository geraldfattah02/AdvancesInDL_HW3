import json
from pathlib import Path

def generate_dataset(output_json: str = "data/rft.json", oversample: int = 5, temperature: float = 0.6):
    import json
    from pathlib import Path
    from .cot import CoTModel
    from .data import Dataset

    model = CoTModel("HuggingFaceTB/SmolLM2-1.7B-Instruct")
    # Override micro_batch_size to avoid OOM with the larger model
    model._micro_batch_size = 8

    dataset = Dataset("train")
    questions = [dataset[i] for i in range(500)]
    prompts = [model.format_prompt(q) for q, _ in questions]

    print(f"Generating {oversample} completions for {len(prompts)} questions...")

    results = []
    # Process in small chunks to avoid OOM
    from tqdm import tqdm
    for i in tqdm(range(0, len(prompts), 8)):
        batch_prompts = prompts[i:i+8]
        batch_questions = questions[i:i+8]

        completions_batch = model.batched_generate(
            batch_prompts, num_return_sequences=oversample, temperature=temperature
        )

        for (question, true_answer), completions in zip(batch_questions, completions_batch):
            for completion in completions:
                parsed = model.parse_answer(completion)
                if abs(true_answer) < 1e-6:
                    correct = abs(parsed) < 1e-6
                else:
                    correct = abs(parsed - true_answer) / abs(true_answer) < 0.01
                if correct:
                    results.append([question, true_answer, completion.strip()])
                    break

    print(f"Dataset size: {len(results)} / {len(questions)} ({100*len(results)/len(questions):.1f}%)")

    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
