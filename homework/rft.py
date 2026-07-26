import json
from pathlib import Path
from .base_llm import BaseLLM
from .sft import test_model, tokenize


def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def train_model(
    output_dir: str = "homework/rft_model",
    data_path: str = "data/rft.json",
    **kwargs,
):
    # Reuse much of the SFT code here
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainingArguments
    from torch.utils.data import Dataset as TorchDataset

    with open(data_path) as f:
        data = json.load(f)

    llm = BaseLLM()

    class _Dataset(TorchDataset):
        def __len__(self): return len(data)
        def __getitem__(self, idx):
            question, _, completion = data[idx]
            return tokenize(llm.tokenizer, question=question, answer=completion.strip())

    lora_config = LoraConfig(
        r=32,
        lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    llm.model = get_peft_model(llm.model, lora_config)
    llm.model.enable_input_require_grads()

    trainer = Trainer(
        model=llm.model,
        args=TrainingArguments(
            output_dir=output_dir,
            logging_dir=output_dir,
            report_to="tensorboard",
            max_steps=200,
            per_device_train_batch_size=32,
            learning_rate=2e-4,
            gradient_checkpointing=True,
            warmup_steps=20,
            lr_scheduler_type="cosine",
            logging_steps=20,
        ),
        train_dataset=_Dataset(),
    )

    trainer.train()

    final_path = Path(__file__).parent / "rft_model"
    llm.model.save_pretrained(final_path)
    print(f"Saved to {final_path}")
    test_model(str(final_path))


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
