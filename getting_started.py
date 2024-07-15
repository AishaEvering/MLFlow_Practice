# %%
from transformers import AutoModelForSeq2SeqLM
import plotly.express as px
import pandas as pd
import mlflow
import numpy as np

mlflow.set_tracking_uri('http://127.0.0.1:5000')
# %% Create a new experiment
experiment_id = mlflow.create_experiment('My Experiment')
# %% Create a new run

# with context manager
with mlflow.start_run(experiment_id=experiment_id):
    # mlflow code goes here
    ...

# %% Create run with custom name
run = mlflow.start_run(experiment_id=experiment_id, run_name="First Run")

# %% Logging parameters
mlflow.log_param("learning_rate", 0.01)
mlflow.log_param('batch_size', 32)

num_epochs = 10
mlflow.log_param('num_epochs', num_epochs)

# %% Logging metrics
for epoch in range(num_epochs):
    mlflow.log_metric('accuracy', np.random.random(), step=epoch)
    mlflow.log_metric('loss', np.random.random(), step=epoch)

# %% Logging time-series metrics
for t in range(100):
    metric_value = np.sin(t * np.pi / 50)
    mlflow.log_metric('time_series_metric', metric_value, step=t)
# %% Logging artifacts
with open('data/dataset.csv', 'w') as f:
    f.write('x, y\n')
    for x in range(100):
        f.write(f'{x}, {x + 1}\n')


# %%
mlflow.log_artifact('data/dataset.csv', 'data')

# %%
!pip install plotly pandas

# %%
# !pip install plotly pandas

# Generate a confusion matrix
confusion_matrix = np.random.randint(0, 100, size=(5, 5))  # 5x5 matrix

labels = ["Cat", "Dog", "Rabbit", "Horse", "Pig"]
df_cm = pd.DataFrame(confusion_matrix, index=labels, columns=labels)

# Plot confusion matrix using Plotly Express
fig = px.imshow(df_cm, text_auto=True, labels=dict(
    x="Predicted Label", y="True Label"), x=labels, y=labels, title="Confusion Matrix")

# Save the figure as an HTML file
html_file = "confusion_matrix.html"
fig.write_html(html_file)

# Log the HTML file with MLflow
mlflow.log_artifact(html_file)
# %%
fig

# %%
!pip install transformers

# Initialize a model from Hugging Face Transformers
model = AutoModelForSeq2SeqLM.from_pretrained(
    "TheFuzzyScientist/T5-base_Amazon-product-reviews")


# Log the model in MLflow
mlflow.pytorch.log_model(model, "transformer_model")
# %% End run
mlflow.end_run()

# %%
