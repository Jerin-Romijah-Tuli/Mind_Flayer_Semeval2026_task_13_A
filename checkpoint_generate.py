# pip install -q transformers datasets accelerate --upgrade

import pandas as pd
import numpy as np
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import os
import random  # For augmentation

# ==================== LOAD DATA ====================
train_path = '/kaggle/input/semeval-task13/train.parquet'
val_path = '/kaggle/input/semeval-task13/validation.parquet'
test_path = '/kaggle/input/semeval-task13/test.parquet'  # Replace with full test path if available
train_df = pd.read_parquet(train_path)
val_df = pd.read_parquet(val_path)
test_df = pd.read_parquet(test_path)
train_df['code'] = train_df['code'].astype(str)
val_df['code'] = val_df['code'].astype(str)
test_df['code'] = test_df['code'].astype(str)
train_df['label'] = train_df['label'].astype(int)
val_df['label'] = val_df['label'].astype(int)
print(f"Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")

# ==================== CLASS WEIGHTS ====================
class_weights = torch.tensor(
    compute_class_weight('balanced', classes=np.unique(train_df['label']), y=train_df['label']),
    dtype=torch.float
)
print("Class weights:", class_weights.numpy())

# ==================== CUSTOM TRAINER ====================
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights.to(outputs.logits.device))
        loss = loss_fct(outputs.logits.view(-1, 2), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# ==================== METRICS ====================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'macro_f1': f1_score(labels, preds, average='macro')
    }

# ==================== SIMPLE AUGMENTATION FOR OOD ROBUSTNESS ====================
def augment_code(code):
    """Light augmentation: Randomly add/remove spaces, change variable names slightly (for code robustness)."""
    if random.random() < 0.3:  # 30% chance
        code = code.replace(' ', '  ')  # Add extra spaces
    if random.random() < 0.3:
        code = code.replace('var', 'tmp_var')  # Simple variable rename (expand as needed)
    return code

# Apply to train_df (for better generalization)
train_df['code'] = train_df['code'].apply(augment_code)

# ==================== TRAIN/RESUME ONE MODEL ====================
def train_model(model_name, output_dir, checkpoint_path=None):
    print(f"\n=== Training/Resuming {model_name} ===")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize(examples):
        return tokenizer(examples['code'], truncation=True, padding="max_length", max_length=384)
    
    raw = DatasetDict({
        'train': Dataset.from_pandas(train_df[['code', 'label']]),
        'val': Dataset.from_pandas(val_df[['code', 'label']]),
        'test': Dataset.from_pandas(test_df[['code']].copy())
    })
    
    if 'id' in test_df.columns:
        raw['test'] = raw['test'].add_column('id', test_df['id'].tolist())
    
    tokenized = raw.map(tokenize, batched=True, remove_columns=['code'])
    tokenized['train'] = tokenized['train'].rename_column('label', 'labels')
    tokenized['val'] = tokenized['val'].rename_column('label', 'labels')
    
    tokenized['train'].set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    tokenized['val'].set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    tokenized['test'].set_format('torch', columns=['input_ids', 'attention_mask'])
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name if checkpoint_path is None else checkpoint_path, 
        num_labels=2, classifier_dropout=0.15
    )
    
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=2,  # 1 epoch for speed + less overfit
        per_device_train_batch_size=24,
        per_device_eval_batch_size=48,
        learning_rate=2e-5,  # Best as per discussion
        warmup_ratio=0.15,  # Slightly higher for stability with 3e-5
        weight_decay=0.08,
        logging_steps=150,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=True,
        report_to=[],
        save_total_limit=1,
        resume_from_checkpoint=checkpoint_path  # Resume if partial
    )
    
    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=tokenized['train'],
        eval_dataset=tokenized['val'],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )
    
    if checkpoint_path:  # If resuming, continue training
        trainer.train(resume_from_checkpoint=checkpoint_path)
    else:
        trainer.train()
    
    # Final validation
    val_metrics = trainer.evaluate()
    final_macro_f1 = val_metrics['eval_macro_f1']
    print(f"\nFinal Val Macro-F1 for {model_name}: {final_macro_f1:.4f}")
    
    # Classification report
    val_pred = trainer.predict(tokenized['val'])
    print("\n=== VALIDATION CLASSIFICATION REPORT ===")
    print(classification_report(val_pred.label_ids, np.argmax(val_pred.predictions, axis=-1),
                               target_names=['Human (0)', 'Machine (1)']))
    
    return trainer, tokenized['test'], final_macro_f1

# ==================== TRAIN ENSEMBLE (ADD MORE MODELS FOR BETTER ACCURACY) ====================
model_configs = [
    {"name": "microsoft/graphcodebert-base", "checkpoint": None},  # Replace XXXXX with actual checkpoint dir (e.g., checkpoint-41668 from log)
    {"name": "microsoft/unixcoder-base", "checkpoint": None},  # If available; else set to None
    {"name": "microsoft/codebert-base", "checkpoint": None},  # New model for stronger ensemble
      # Lightweight new model
]

trainers = []
test_tokenized_list = []
val_f1_scores = []

for config in model_configs:
    output_dir = f"/kaggle/working/results_{config['name'].split('/')[-1]}"
    trainer, tokenized_test, val_f1 = train_model(config['name'], output_dir, config['checkpoint'])
    trainers.append(trainer)
    test_tokenized_list.append(tokenized_test)
    val_f1_scores.append(val_f1)

# ==================== WEIGHTED ENSEMBLE ====================
print("\n=== Weighted Ensemble ===")
weights = np.array(val_f1_scores) / np.sum(val_f1_scores)
print(f"Weights: {list(zip([c['name'] for c in model_configs], weights))}")

all_logits = []
for i, trainer in enumerate(trainers):
    pred = trainer.predict(test_tokenized_list[i])
    all_logits.append(pred.predictions)

ensemble_logits = np.average(all_logits, axis=0, weights=weights)
ensemble_preds = np.argmax(ensemble_logits, axis=-1)

# ==================== SUBMISSION ====================
submission = pd.DataFrame({
    'ID': test_df['id'].tolist() if 'id' in test_df.columns else list(range(len(ensemble_preds))),
    'label': ensemble_preds
})
submission.to_csv('/kaggle/working/submission.csv', index=False)
print("\nSUBMISSION READY → /kaggle/working/submission.csv")
print(submission.head(8))