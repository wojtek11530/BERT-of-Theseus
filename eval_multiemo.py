from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import json
import time
from datetime import timedelta

import numpy as np
import torch
from sklearn.metrics import classification_report
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler)

from tqdm import tqdm, trange
from copy import deepcopy
from torch import nn
from transformers import CONFIG_NAME, BertConfig, BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

from bert_of_theseus import BertForSequenceClassification
from bert_of_theseus.replacement_scheduler import ConstantReplacementScheduler, LinearReplacementScheduler
from transformers.modeling_bert import BertForSequenceClassification as OriginalBertForSequenceClassification

from data_processing.processors.multiemo import multiemo_output_modes as output_modes, Dataset, SmartCollator
from data_processing.processors.multiemo import MultiemoProcessor
from data_processing.processors.multiemo import multiemo_convert_examples_to_features as convert_examples_to_features
from data_processing.metrics import multiemo_compute_metrics as compute_metrics
from utils import dictionary_to_json, result_to_text_file, is_folder_empty

logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig,)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
}


def evaluate(args, model, eval_dataset, tokenizer):
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    collator = SmartCollator(tokenizer.pad_token_id)
    eval_sampler = SequentialSampler(eval_dataset)
    dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, collate_fn=collator.collate_batch,
                            pin_memory=True, sampler=eval_sampler)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # multi-gpu eval
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    all_logits = None
    out_label_ids = None

    for batch in tqdm(dataloader, desc="Evaluating"):
        model.eval()
        batch = {k: v.to(args.device) for k, v in batch.items()}

        with torch.no_grad():
            inputs = {'input_ids': batch['input_ids'],
                      'attention_mask': batch['attention_mask'],
                      'token_type_ids': batch['token_type_ids'],
                      'labels': batch['labels']}

            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.mean().item()

            nb_eval_steps += 1
            if all_logits is None:
                all_logits = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                all_logits = np.append(all_logits, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps

    if args.output_mode == "regression":
        all_logits = np.squeeze(all_logits)

    result = compute_metrics(args.task_name, all_logits, out_label_ids)
    result['eval_loss'] = eval_loss
    return result, all_logits, out_label_ids


def load_and_cache_examples(args, task, tokenizer, evaluate_set=False, test_set=False):
    _, lang, domain, kind = task.split('_')
    processor = MultiemoProcessor(lang, domain, kind)
    output_mode = output_modes['multiemo']

    logger.info("Creating features from dataset file at %s", args.data_dir)
    label_list = processor.get_labels()

    if test_set:
        examples = processor.get_test_examples(args.data_dir)
    elif evaluate_set:
        examples = processor.get_dev_examples(args.data_dir)
    else:
        examples = processor.get_train_examples(args.data_dir)

    features = convert_examples_to_features(
        examples,
        tokenizer,
        max_length=args.max_seq_length,
        label_list=label_list,
        output_mode=output_mode
    )

    # Convert to Tensors and build dataset
    all_input_ids = [f.input_ids for f in features]
    all_attention_mask = [f.attention_mask for f in features]
    all_token_type_ids = [f.token_type_ids for f in features]
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)
    else:
        raise ValueError

    dataset = Dataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_path", default=None, type=str, required=True,
                        help="Path to model to evaluate")
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger.warning("Device: %s, n_gpu: %s", device, args.n_gpu)

    # Prepare GLUE task
    if 'multiemo' not in args.task_name:
        raise ValueError("Task not found: %s" % args.task_name)

    _, lang, domain, kind = args.task_name.split('_')
    processor = MultiemoProcessor(lang, domain, kind)
    args.output_mode = output_modes['multiemo']
    label_list = processor.get_labels()
    num_labels = len(label_list)
    label_map = {label: i for i, label in enumerate(label_list)}
    labels = list(label_map.values())

    logger.info("evaluation parameters %s", args)

    output_dir = args.output_dir

    # Load a trained model and vocabulary
    config = BertConfig.from_pretrained(
        args.model_path,
        num_labels=num_labels,
        finetuning_task=args.task_name
    )
    model = OriginalBertForSequenceClassification.from_pretrained(args.model_path, config=config)
    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    model.to(device)

    test_dataset = load_and_cache_examples(args, args.task_name, tokenizer, test_set=True)

    eval_start_time = time.monotonic()
    model.eval()
    result, y_logits, y_true = evaluate(args, model, test_dataset, tokenizer)
    eval_end_time = time.monotonic()

    diff = timedelta(seconds=eval_end_time - eval_start_time)
    diff_seconds = diff.total_seconds()
    result['eval_time'] = diff_seconds

    if args.task_name in output_dir:
        result_file_name = 'test_results'
    else:
        result_file_name = f'test_results_{args.task_name}'

    result_to_text_file(result, os.path.join(args.output_dir, f"{result_file_name}.txt"))

    y_pred = np.argmax(y_logits, axis=1)
    print('\n\t**** Classification report ****\n')
    print(classification_report(y_true, y_pred, target_names=label_list, labels=labels))

    report = classification_report(y_true, y_pred, target_names=label_list, labels=labels, output_dict=True)
    report['eval_time'] = diff_seconds

    dictionary_to_json(report, os.path.join(args.output_dir, f"{result_file_name}.json"))


if __name__ == "__main__":
    main()
