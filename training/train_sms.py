"""Train the SMS DistilBERT classifier for the Digital Trust Engine.

Expected CSV columns:
    LABEL: ham / spam / smishing
    TEXT: message text

Cleaning used for the current project model:
- normalize labels to lowercase
- remove the two messages with conflicting labels
- remove duplicate TEXT messages
- stratified 80/20 train/test split

The resulting modeling dataset contains 5,947 records.
"""

import os
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
RANDOM_STATE = 42

LABEL2ID = {"ham": 0, "spam": 1, "smishing": 2}
ID2LABEL = {0: "ham", 1: "spam", 2: "smishing"}

# These two message texts occurred with conflicting labels in the source data.
CONFLICTING_TEXTS = {
    "Great News! Call FREEFONE 08006344447 to claim",
    "XXXMobileMovieClub: To use your credit, click",
}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probabilities = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(
            labels, predictions, average="macro", zero_division=0
        ),
        "recall": recall_score(
            labels, predictions, average="macro", zero_division=0
        ),
        "f1": f1_score(
            labels, predictions, average="macro", zero_division=0
        ),
        "roc_auc": roc_auc_score(
            labels,
            probabilities,
            multi_class="ovr",
            average="macro",
        ),
    }


def clean_dataset(df):
    df = df.copy()

    df["LABEL"] = df["LABEL"].fillna("").astype(str).str.strip().str.lower()
    df["TEXT"] = df["TEXT"].fillna("").astype(str).str.strip()

    df = df[df["LABEL"].isin(LABEL2ID)]
    df = df[df["TEXT"] != ""]

    # Remove the two exact conflicting message texts.
    normalized_conflicts = {
        text.strip().lower() for text in CONFLICTING_TEXTS
    }
    df = df[~df["TEXT"].str.lower().isin(normalized_conflicts)]

    # Remove duplicate message texts.
    df = df.drop_duplicates(subset=["TEXT"]).reset_index(drop=True)

    df["label"] = df["LABEL"].map(LABEL2ID).astype(int)

    return df[["TEXT", "label"]]


def main(
    dataset_path: str,
    output_dir: str = "./distilbert_sms_results",
):
    df = pd.read_csv(dataset_path)
    df = clean_dataset(df)

    print("Clean dataset shape:", df.shape)
    print("\nLabel distribution:")
    print(df["label"].map(ID2LABEL).value_counts())

    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        stratify=df["label"],
        random_state=RANDOM_STATE,
    )

    train_ds = Dataset.from_pandas(
        train_df.reset_index(drop=True)
    )
    test_ds = Dataset.from_pandas(
        test_df.reset_index(drop=True)
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(
            batch["TEXT"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    train_tokenized = train_ds.map(tokenize, batched=True)
    test_tokenized = test_ds.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=50,
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
    print("\nFinal evaluation:")
    print(trainer.evaluate())


if __name__ == "__main__":
    dataset_path = os.environ.get("SMS_DATASET_PATH", "Dataset_5971.csv")
    main(dataset_path)
