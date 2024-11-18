# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
 
import logging
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from dataset import EvalDataset
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

"""
Code Completion Evaluation Pipeline

This script evaluates a pre-trained language model’s token-level accuracy on code completion tasks.
The script includes functions to decode token IDs, calculate accuracy, and process prediction batches
for evaluation. It also verifies predictions against ground truth data and saves results to an output file.

Modules and Functions:
- `decode_token_ids`: Converts token IDs into readable code strings, managing special tokens and spacing.
- `calculate_accuracy`: Compares predicted tokens with ground truth tokens and calculates accuracy.
- `process_batch_predictions`: Processes predictions and ground truths from batches, converting them into lists of tokens.
- `eval_acc`: The main evaluation function that loads data, predicts, decodes, and calculates accuracy.
- `post_process`: Saves processed predictions to a file and verifies each sequence against expected ground truth.
- `main`: The entry point, which loads the model, sets configurations, and initiates the evaluation pipeline.

Dependencies:
- Libraries: `torch`, `transformers`, `numpy`, and `torch.utils.data`.
- Custom Modules: `EvalDataset` (assumed to be a dataset module for evaluation tasks).
- Assumes access to a pre-trained GPT-2-based model for code completion.

Usage:
To run the script, ensure all dependencies are installed and specify the model and dataset paths.
Logging will provide progress updates and final evaluation metrics.

"""


logger = logging.getLogger(__name__)

def decode_token_ids(token_ids, tokenizer):
    """
    Convert token IDs to a string of code, handling special tokens and spacing.
    """
    decoded_code = ""
    for token_id in token_ids:
        token = tokenizer.convert_ids_to_tokens(token_id)
        if token.startswith('\u0120') and not decoded_code.endswith(" "):  # Handles space prefixes
            decoded_code += " " + token[1:]
        else:
            decoded_code += token
    return decoded_code.strip()

def calculate_accuracy(pred_tokens, gt_tokens, special_tokens=["<s>", "</s>", "<EOL>", "<pad>"]):
    """
    Calculate accuracy by comparing predicted tokens to ground truth tokens.
    """
    correct_count = sum(1 for pred, gt in zip(pred_tokens, gt_tokens) if gt not in special_tokens and pred == gt)
    total_count = sum(1 for gt in gt_tokens if gt not in special_tokens)
    return correct_count, total_count

def process_batch_predictions(batch_predictions, batch_ground_truths, tokenizer):
    """
    Process batch of predictions and ground truths into readable token lists.
    """
    all_pred_tokens, all_gt_tokens = [], []

    for predicted_ids, gt_ids in zip(batch_predictions, batch_ground_truths):
        pred_tokens, gt_tokens = [], []
        for i, (pred_id, gt_id) in enumerate(zip(predicted_ids, gt_ids)):
            gt_token = tokenizer.convert_ids_to_tokens(gt_id)

            if gt_token in ["<s>", "</s>", "<EOL>", "<pad>"]:  # Skip special tokens
                break
            elif gt_token.startswith('\u0120') and pred_tokens:  # New token starts with a space
                all_pred_tokens.append(decode_token_ids(pred_tokens, tokenizer))
                all_gt_tokens.append(decode_token_ids(gt_tokens, tokenizer))
                pred_tokens, gt_tokens = [], []
                
            pred_tokens.append(pred_id)
            gt_tokens.append(gt_id)

    return all_pred_tokens, all_gt_tokens

def eval_acc(args, model, tokenizer, file_type='test'):
    """
    Evaluate the model’s token-level code completion accuracy.
    """
    # Load evaluation dataset
    eval_dataset = EvalDataset(tokenizer, args, logger, file_type=file_type, block_size=args.block_size)
    eval_dataloader = DataLoader(eval_dataset, sampler=SequentialSampler(eval_dataset), batch_size=args.eval_batch_size)
    model.to(args.device)
    model.eval()

    # Initialize counters for accuracy
    total_correct, total_predictions = 0, 0
    all_pred_tokens, all_gt_tokens = [], []

    # Iterate through batches in the evaluation dataset
    for step, batch in enumerate(eval_dataloader):
        inputs = batch.to(args.device)
        with torch.no_grad():
            outputs = model(inputs)
            predicted_token_ids = outputs.logits.argmax(-1)  # Get predicted tokens

        # Decode batch predictions and ground truths
        batch_pred_tokens, batch_gt_tokens = process_batch_predictions(predicted_token_ids.cpu(), inputs.cpu(), tokenizer)
        all_pred_tokens.extend(batch_pred_tokens)
        all_gt_tokens.extend(batch_gt_tokens)

        # Calculate batch accuracy
        batch_correct, batch_total = calculate_accuracy(batch_pred_tokens, batch_gt_tokens)
        total_correct += batch_correct
        total_predictions += batch_total

        # Logging progress
        if step % args.logging_steps == 0:
            logger.info(f"Step {step} processed with cumulative accuracy: {total_correct/total_predictions:.2%}")

    # Final accuracy calculation
    accuracy = total_correct / total_predictions if total_predictions > 0 else 0
    logger.info(f"Final Test Accuracy: {accuracy:.2%}")
    return accuracy

def post_process(args, predictions, ground_truths, true_texts, saved_file_path):
    """
    Save the post-processed predictions and verify with the ground truth texts.

    Args:
        args: General arguments or configuration settings (unused here).
        predictions: List of predicted tokens from the model.
        ground_truths: List of ground truth tokens for each prediction.
        true_texts: List of full ground truth sequences for each input, used for verification.
        saved_file_path: Path to the file where the processed predictions will be saved.

    Returns:
        int: The count of sequences processed and saved.
    """
    # Open the specified file in write mode to save processed predictions
    with open(saved_file_path, "w") as wf:
        count = 0  # Initialize a counter to track the number of completed sequences
        current_pred, current_gt = [], []  # Lists to accumulate tokens for each sequence

        # Iterate through each predicted and ground truth token pair
        for pred, gt in zip(predictions, ground_truths):
            # Skip empty or padding tokens in the ground truth, as they are not meaningful
            if gt in ["", "<pad>"]:
                continue
            
            # Append the current ground truth token to the list for the sequence
            current_gt.append(gt)
            # Append the current prediction, removing any extra spaces
            current_pred.append(pred.replace(" ", ""))

            # Check if the current token is an end-of-sequence token
            if gt == "</s>":
                # Verify that the accumulated ground truth tokens match the expected text
                assert " ".join(current_gt) == true_texts[count].strip(), f"Mismatch in sample {count}"
                
                # Write the joined prediction sequence as a line in the file
                wf.write(" ".join(current_pred) + "\n")
                
                # Increment the count of completed sequences
                count += 1
                
                # Clear the lists to start accumulating tokens for the next sequence
                current_pred, current_gt = [], []

    # Return the total number of processed sequences
    return count


def main():
    """
    Main function to load model, tokenizer, and execute evaluation.
    """
    pretrained_model_path = "/content/GPT-NeoX-Colab/models/codecompletion/latest"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up evaluation arguments
    args = {
        "n_gpu": torch.cuda.device_count(),
        "per_gpu_eval_batch_size": 8,
        "logging_steps": 100,
        "output_dir": "/content/GPT-NeoX-Colab/dataset/py150",
        "data_dir": "/content/GPT-NeoX-Colab/dataset/py150",
        "device": device,
        "no_cuda": False
    }

    # Configure logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    # Set random seed for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_path, sep_token='<EOL>', bos_token='<s>', eos_token='</s>', pad_token='<pad>', unk_token='<|UNKNOWN|>')
    model = GPT2LMHeadModel.from_pretrained(pretrained_model_path)
    model.resize_token_embeddings(len(tokenizer))
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model has {total_params} trainable parameters")
    
    # Evaluate model
    accuracy = eval_acc(args, model, tokenizer, 'test')
    logger.info(f"Test accuracy: {accuracy:.2%}")

main()
