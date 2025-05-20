# AWS-Sagemaker

# HOW TO USE & TRAIN THE LLM ON SAGEMAKER
# ✅ Step 1: Deploy the Terraform Infrastructure
terraform init
terraform apply
You’ll get outputs like: S3 bucket name,SageMaker role ARN

# ✅ Step 2: Upload Training Script to S3
Example training script train.py:
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
dataset = load_dataset("imdb")
def tokenize(batch):
    return tokenizer(batch['text'], truncation=True, padding=True)
dataset = dataset.map(tokenize, batched=True)
training_args = TrainingArguments(output_dir="/opt/ml/model", per_device_train_batch_size=8, num_train_epochs=1)
trainer = Trainer(model=model, args=training_args, train_dataset=dataset['train'].select(range(1000)), eval_dataset=dataset['test'].select(range(200)))
trainer.train()

# Upload:
aws s3 cp train.py s3://<your-bucket>/scripts/train.py

# ✅ Step 3: Train the Model via SageMaker SDK (Python)
import sagemaker
from sagemaker.huggingface import HuggingFace
role = "<role_arn_from_terraform>"
sagemaker_session = sagemaker.Session()
huggingface_estimator = HuggingFace(
    entry_point='train.py',
    source_dir='.',
    role=role,
    transformers_version='4.26',
    pytorch_version='1.13',
    py_version='py39',
    instance_type='ml.m5.large',
    instance_count=1
)
huggingface_estimator.fit()

# ✅ Step 4: Deploy the Trained Model to Endpoint
python
Copy
Edit
predictor = huggingface_estimator.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large"
)

# ✅ Step 5: Use the Endpoint for Inference
python
Copy
Edit
response = predictor.predict({
  "inputs": "This movie was brilliant!"
})
print(response)

# Step 6: Clean Up
predictor.delete_endpoint()
