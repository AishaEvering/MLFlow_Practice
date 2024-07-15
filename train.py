# %%
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import mlflow
import torch
from datasets import load_dataset
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, AdamW
!pip install - q accelerate == 0.21.0 peft == 0.4.0 bitsandbytes == 0.40.2 transformers == 4.31.0 trl == 0.4.7
!pip install datasets

# %%


# %%
params = {
    'model_name': 'distilbert-base-uncased',
    'learning_rate': 5e-5,
    'batch_size': 16,
    'num_epochs': 3,
    'dataset_name': 'ag_news',
    'task_name': 'sequence_classification',
    'log_steps': 100,
    'max_seq_length': 128,
    'output_dir': 'models/distilbert_ag-news'
}

# %% create experiment
mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.set_experiment(params['task_name'])

# %% create run
run = mlflow.start_run(
    run_name=f"{params['model_name']}-{params['dataset_name']}-2")

# %% load params
mlflow.log_params(params)

# %% load dataset
dataset = load_dataset(params['dataset_name'], split=['train', 'test'])

# %%
dataset

# %% get tokenizer
tokenizer = DistilBertTokenizer.from_pretrained(params['model_name'])

# %%
tokenizer = DistilBertTokenizer.from_pretrained(params['model_name'])

# %%


def tokenize(batch):
    return tokenizer(batch['text'],
                     padding='max_length',
                     truncation=True,
                     max_length=params['max_seq_length'])


# %%
dataset[0]

# %%
train_dataset = dataset[0].shuffle().select(
    range(20_000)).map(tokenize, batched=True)

test_dataset = dataset[1].shuffle().select(
    range(2_000)).map(tokenize, batched=True)

# %% set format for PyTorch
train_dataset.set_format(
    'torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format(
    'torch', columns=['input_ids', 'attention_mask', 'label'])

# %%  Create data loaders
train_loader = DataLoader(
    train_dataset, batch_size=params['batch_size'], shuffle=True)
test_loader = DataLoader(
    test_dataset, batch_size=params['batch_size'], shuffle=False)

# %% get the labels
labels = dataset[0].features['label'].names
labels

# %% Save datasets to disk
train_dataset.to_parquet('data/train.parquet')
test_dataset.to_parquet('data/test.parquet')

# %% log datsets to disk
mlflow.log_artifact('data/train.parquet', artifact_path='datasets')
mlflow.log_artifact('data/test.parquet', artifact_path='datasets')

# %% Define model
model = DistilBertForSequenceClassification.from_pretrained(
    params['model_name'], num_labels=len(labels))

# %%
model.config.id2label = {i: label for i, label in enumerate(labels)}

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# %%
params['id2label'] = model.config.id2label
mlflow.log_params(params)

# %% Define optimizer
optimizer = AdamW(model.parameters(), lr=params['learning_rate'])

# %%


def evaluate_model(model, dataloader, device):
    predictions, true_labels = [], []

    model.eval()

    with torch.inference_mode():
        for batch in dataloader:
            inputs = batch['input_ids'].to(device)
            masks = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # Forward pass, calculate logit predictions
            outputs = model(inputs, attention_mask=masks)
            logits = outputs.logits
            _, predicted_labels = torch.max(logits, dim=1)

            predictions.extend(predicted_labels.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    # Calculate Evaluation Metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels,
                                                               predictions, average='macro')

    return accuracy, precision, recall, f1


# %% Training loop

with tqdm(total=params['num_epochs'] * len(train_loader), desc=f"Epoch [1/{params['num_epochs']}] - (Loss: N/A) - Steps") as pbar:
    for epoch in range(params['num_epochs']):
        running_loss = 0.0

        for i, batch in enumerate(train_loader, 0):
            inputs, masks, labels = batch['input_ids'].to(
                device), batch['attention_mask'].to(device), batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(inputs, attention_mask=masks, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i and i % params['log_steps'] == 0:
                avg_loss = running_loss / params['log_steps']

                pbar.set_description(
                    f"Epoch [{epoch + 1}/{params['num_epochs']}] - (Loss: {avg_loss:.3f}) - Steps")
                mlflow.log_metric("loss", avg_loss,
                                  step=epoch * len(train_loader) + i)

                running_loss = 0.0
            pbar.update(1)

        # Evaluate Model
        accuracy, precision, recall, f1 = evaluate_model(
            model, test_loader, device)
        print(f"Epoch {epoch + 1} Metrics: Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        # Log metrics to MLflow
        mlflow.log_metrics({'accuracy': accuracy, 'precision': precision,
                           'recall': recall, 'f1': f1}, step=epoch)

# %%
# Log model to MLflow through built-in PyTorch method
mlflow.pytorch.log_model(model, "model")

# %%
# Log model to MLflow through custom method
os.makedirs(params['output_dir'], exist_ok=True)

model.save_pretrained(params['output_dir'])
tokenizer.save_pretrained(params['output_dir'])

mlflow.log_artifacts(params['output_dir'], artifact_path="custom_model")

model_uri = f"runs:/{run.info.run_id}/custom_model"
mlflow.register_model(model_uri, "agnews-transformer")

print('Finished Training')

# %%
mlflow.end_run()

# %%
