import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM  # Use the correct model class
from transformers.tokenization_utils_base import PaddingStrategy
import os
import pandas as pd
import torch.nn.functional as F
import logging

# Set up logging
logging.basicConfig(filename='script.log', level=logging.DEBUG)

# Set the token
os.environ['HF_TOKEN'] = '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'

# Load the model
model_name = 'meta-llama/Llama-2-7b-chat-hf'
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Set pad_token
tokenizer.pad_token = tokenizer.eos_token

# Load your dataset
df = pd.read_csv('qanda.csv')
questions = df['Q'].tolist()
answers = df['A'].tolist()

# Tokenize input and output sequences
tokenized_inputs = tokenizer(questions, return_tensors='pt', padding=PaddingStrategy.MAX_LENGTH, truncation=True, max_length=64)
tokenized_outputs = tokenizer(answers, return_tensors='pt', padding=PaddingStrategy.MAX_LENGTH, truncation=True, max_length=64)

logging.info(f"Input tokens shape: {tokenized_inputs['input_ids'].shape}")
logging.info(f"Output tokens shape: {tokenized_outputs['input_ids'].shape}")

# Create PyTorch dataset
dataset = TensorDataset(tokenized_inputs['input_ids'], tokenized_outputs['input_ids'])

# Set up the PyTorch dataloader
dataloader = DataLoader(dataset, batch_size=15, shuffle=True)

# Set up the optimizer and scheduler
optimizer = optim.AdamW(model.parameters(), lr=1e-5)

# Define a lambda function for the learning rate schedule with warm-up
lr_lambda = lambda epoch: min(1.0, (epoch + 1) / 10)

# Create the scheduler with warm-up
scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

# Fine-tuning loop
model.train()
for epoch in range(10):  # Number of epochs
    logging.info(f"Epoch {epoch + 1}/10")
    for batch_idx, batch in enumerate(dataloader):
        logging.info(f"Processing batch {batch_idx + 1}/{len(dataloader)}")

        input_batch = batch[0]
        label_batch = batch[1]

        outputs = model(input_ids=input_batch, labels=label_batch)
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = label_batch[..., 1:].contiguous()

        # Flatten the logits and labels
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), weight=None)

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        logging.info("Training step completed")

# Save the fine-tuned model
save_path = 'firstLlama'
logging.info(f"Saving the model to: {save_path}")
model.save_pretrained(save_path)
logging.info(f"Model saved successfully!")
