# %%
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import mlflow
!pip install - q accelerate == 0.21.0 peft == 0.4.0 bitsandbytes == 0.40.2 transformers == 4.31.0 trl == 0.4.7


# %%

# %%
mlflow.set_tracking_uri('http://127.0.0.1:5000')
# %%
client = mlflow.tracking.MlflowClient()

# %% Retreive the model from mlflow
model_name = 'agnews_pt_classifier'
model_version = '1'

# %%
model_uri = f'models:/{model_name}/{model_version}'
# %%
model_uri
# %%
model = mlflow.pytorch.load_model(model_uri)

# %% Sample text to predict
texts = [
    "The local high school soccer team triumphed in the state championship, securing victory with a last-second winning goal.",
    "DataCore is set to acquire startup InnovateAI for $2 billion, aiming to enhance its position in the artificial intelligence market.",
]

# %% load tokenizer
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

# %%


def predict(texts, model, tokenizer):
    # Tokenize the texts
    inputs = tokenizer(texts, padding=True, truncation=True,
                       return_tensors='pt').to(model.device)

    # Pass the inputs to the model
    with torch.inference_mode():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=1)

    # Convert predictions to text labels
    predictions = predictions.cpu().numpy()
    predictions = [model.config.id2label[prediction]
                   for prediction in predictions]

    # Print predictions
    return predictions


# %%
predict(texts, model, tokenizer)
# %%


# %% Loading the custom model
model_name = 'agnews-transformer'
model_version = '1'

model_version_details = client.get_model_version(
    name=model_name, version=model_version)
model_version_details

# %%
run_id = model_version_details.run_id
artifact_path = model_version_details.source

# %%
model_path = 'models/agnews_transformer'
os.makedirs(model_path, exist_ok=True)

# %% dowload model
client.download_artifacts(run_id, artifact_path, dst_path=model_path)

# %%
custom_model = AutoModelForSequenceClassification.from_pretrained(
    'models/agnews_transformer/custom_model')
tokenizer = AutoTokenizer.from_pretrained(
    'models/agnews_transformer/custom_model')

# %% Let's Predict
predict(texts, custom_model, tokenizer)

# %% Versioning

mlflow.set_experiment('sequence_classification')

with mlflow.start_run(run_name='iteration2'):
    mlflow.pytorch.log_model(model, 'model')

with mlflow.start_run(run_name='iteration3'):
    mlflow.pytorch.log_model(model, 'model')


# %%
model_name = 'agnews_pt_classifier'
model_versions = client.search_model_versions(f"name='{model_name}'")
model_versions

# %%
for version in model_versions:
    print(
        f"Version: {version.version}:\nDescription: {version.description}\nStage: {version.current_stage}\n")

# %% Delete a version of the model
client.delete_model_version(name=model_name, version='2')

# %% Completely remove a model from the registry
client.delete_registered_model(name=model_name)

# %%
