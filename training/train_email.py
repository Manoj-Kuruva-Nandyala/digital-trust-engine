"""Train the dedicated Email DistilBERT classifier.

Expected dataset format:
    text_combined: email subject/body text
    label: 0 = legitimate, 1 = phishing

The dataset cleaning/splitting choices used for the current trained model are
kept explicit here so the training process is reproducible.
"""

import os
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256
RANDOM_STATE = 42


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probabilities = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "roc_auc": roc_auc_score(labels, probabilities[:, 1]),
    }


def main(
    dataset_path: str,
    output_dir: str = "./distilbert_email_results",
):
    df = pd.read_csv(dataset_path)

    # Remove invalid/blank text rows.
    df["text_combined"] = df["text_combined"].fillna("").astype(str).str.strip()
    df = df[df["text_combined"] != ""].copy()

    # Remove duplicate email texts.
    df = df.drop_duplicates(subset=["text_combined"]).reset_index(drop=True)

    # Preserve the dataset-derived label mapping:
    # 0 = legitimate, 1 = phishing.
    df["label"] = df["label"].astype(int)

    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        stratify=df["label"],
        random_state=RANDOM_STATE,
    )

    train_ds = Dataset.from_pandas(
        train_df[["text_combined", "label"]].reset_index(drop=True)
    )
    test_ds = Dataset.from_pandas(
        test_df[["text_combined", "label"]].reset_index(drop=True)
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(
            batch["text_combined"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    train_tokenized = train_ds.map(tokenize, batched=True)
    test_tokenized = test_ds.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={0: "legitimate", 1: "phishing"},
        label2id={"legitimate": 0, "phishing": 1},
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=2,
        weight_decay=0.01,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=test_tokenized,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    print(trainer.evaluate())


if __name__ == "__main__":
    dataset_path = os.environ.get("EMAIL_DATASET_PATH", "phishing_email.csv")
    main(dataset_path)
