import os
import json
import torch
from PIL import Image
from datetime import datetime
from datasets import Dataset
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    BitsAndBytesConfig,
)
import logging
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# === Logging setup ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Compute metrics ===
def compute_metrics(eval_pred):
    preds = eval_pred.predictions
    labels = eval_pred.label_ids

    decoded_preds = processor.tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Character Error Rate
    def cer(pred, label):
        from jiwer import cer
        return cer(label, pred)

    total_cer = sum(cer(p, l) for p, l in zip(decoded_preds, decoded_labels)) / len(decoded_preds)

    return {"cer": total_cer}


def load_dataset(data_dir):
    """
    Load image-text pairs from a directory.
    """
    image_files = [f for f in os.listdir(data_dir) if f.endswith(".png")]
    data = []
    for img_file in image_files:
        img_path = os.path.join(data_dir, img_file)
        txt_path = os.path.join(data_dir, img_file.replace(".png", ".txt"))
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
            data.append({
                "image": Image.open(img_path).convert("RGB"),
                "text": text
            })
    return Dataset.from_list(data)


def preprocess(example):
    prompt = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": example["image"]},
                {"type": "text", "text": "What is written on this image?"}
            ]
        }
    ]
    inputs = processor.apply_chat_template(
        prompt,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors=None,
    )
    inputs["labels"] = inputs["input_ids"].copy()
    return inputs


def train_nanonets_ocr(
    data_dir,
    output_dir="./ocr_model_output",
    num_train_epochs=3,
    batch_size=2,
    learning_rate=5e-5,
    warmup_steps=100,
    eval_steps=50,
    save_steps=50,
    logging_steps=10
):
    logger.info("=== Loading processor and model ===")
    model_id = "nanonets/Nanonets-OCR-s"
    bnb_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)

    logger.info("=== Loading dataset ===")
    dataset = load_dataset(data_dir)
    tokenized_dataset = dataset.map(preprocess, remove_columns=dataset.column_names, batched=False)

    logger.info("=== Setting training arguments ===")
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=2,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=logging_steps,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        fp16=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        report_to="none",
    )

    logger.info("=== Initializing trainer ===")
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=processor.tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
    )

    logger.info("=== Starting training ===")
    train_result = trainer.train()

    logger.info("=== Saving final model and processor ===")
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)

    logger.info("=== Saving training results ===")
    results = {
        "train_loss": train_result.training_loss,
        "timestamp": datetime.now().isoformat()
    }
    with open(os.path.join(output_dir, "training_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Training complete. Model saved to {output_dir}")


if __name__ == "__main__":
    train_nanonets_ocr(data_dir="dataset", output_dir="./ocr_model_output")

