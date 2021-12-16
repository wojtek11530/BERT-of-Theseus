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


def manage_output_dir(output_dir: str, task_name: str) -> str:
    output_dir = os.path.join(output_dir, task_name)
    run = 1
    while os.path.exists(output_dir + '-run-' + str(run)):
        if is_folder_empty(output_dir + '-run-' + str(run)):
            logger.info('folder exist but empty, use it as output')
            break
        logger.info(output_dir + '-run-' + str(run) + ' exist, trying next')
        run += 1
    output_dir += '-run-' + str(run)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def train(args, train_dataset, eval_dataset, model, tokenizer):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    collator = SmartCollator(tokenizer.pad_token_id)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, collate_fn=collator.collate_batch,
                                  pin_memory=True, sampler=train_sampler)

    t_total = len(train_dataloader) * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.bert.encoder.scc_layer.named_parameters() if
                    not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.bert.encoder.scc_layer.named_parameters() if
                    any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)

    # Replace rate scheduler
    if args.scheduler_type == 'none':
        replacing_rate_scheduler = ConstantReplacementScheduler(bert_encoder=model.bert.encoder,
                                                                replacing_rate=args.replacing_rate,
                                                                replacing_steps=args.steps_for_replacing)
    elif args.scheduler_type == 'linear':
        replacing_rate_scheduler = LinearReplacementScheduler(bert_encoder=model.bert.encoder,
                                                              base_replacing_rate=args.replacing_rate,
                                                              k=args.scheduler_linear_k)
    else:
        raise ValueError

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Batch size  = %d", args.per_gpu_train_batch_size)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    current_best = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")

    output_eval_file = os.path.join(args.output_dir, 'eval_results.txt')

    for epoch in train_iterator:
        nb_tr_steps = 0

        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()

            batch = {k: v.to(args.device) for k, v in batch.items()}
            inputs = {'input_ids': batch['input_ids'],
                      'attention_mask': batch['attention_mask'],
                      'token_type_ids': batch['token_type_ids'],
                      'labels': batch['labels']}

            outputs = model(**inputs)
            loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            loss.backward()
            tr_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            replacing_rate_scheduler.step()  # Update replace rate scheduler
            model.zero_grad()
            global_step += 1
            nb_tr_steps += 1

        if args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
            logs = {}
            results, _, _ = evaluate(args, model, eval_dataset, tokenizer)
            for key, value in results.items():
                eval_key = 'eval_{}'.format(key)
                logs[eval_key] = float(value)

            loss_scalar = (tr_loss - logging_loss) / nb_tr_steps
            learning_rate_scalar = scheduler.get_lr()[0]
            logs['learning_rate'] = learning_rate_scalar
            logs['epoch'] = epoch + 1
            logs['global_step'] = global_step
            logs['loss'] = loss_scalar
            logging_loss = tr_loss

            print(json.dumps({**logs, **{'step': global_step}}))
            result_to_text_file(logs, output_eval_file)

            if logs['eval_acc'] > current_best:
                current_best = logs['eval_acc']
                # Save model checkpoint
                output_dir = args.output_dir
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                model_to_save = deepcopy(model)
                model_to_save.bert.encoder.layer = model_to_save.bert.encoder.scc_layer
                model_to_save.bert.config.num_hidden_layers = model_to_save.bert.encoder.scc_n_layer
                del model_to_save.bert.encoder.scc_layer

                model_to_save = model_to_save.module if hasattr(model, 'module') else model_to_save
                model_to_save.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                logger.info("Saving hg model checkpoint to %s", output_dir)

    logger.info("Training finished.")
    if global_step > 0:
        return global_step, tr_loss / global_step
    else:
        return global_step, tr_loss


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

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        'test' if test_set else ('dev' if evaluate_set else 'train'),
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
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
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

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
    # dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
                            ALL_MODELS))
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--replacing_rate", type=float, required=True,
                        help="Constant replacing rate. Also base replacing rate if using a scheduler.")
    parser.add_argument("--scheduler_type", default='none', choices=['none', 'linear'], help="Scheduler function.")
    parser.add_argument("--scheduler_linear_k", default=0, type=float, help="Linear k for replacement scheduler.")
    parser.add_argument("--steps_for_replacing", default=0, type=int,
                        help="Steps before entering successor fine_tuning (only useful for constant replacing)")

    parser.add_argument("--model_type", default='bert', type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=20.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    args = parser.parse_args()

    args.output_dir = manage_output_dir(args.output_dir, args.task_name)

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

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None
    )
    config.output_hidden_states = True
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None
    )

    model = model_class.from_pretrained(
        args.model_name_or_path,
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None
    )

    # Initialize successor BERT weights
    scc_n_layer = model.bert.encoder.scc_n_layer
    model.bert.encoder.scc_layer = nn.ModuleList([deepcopy(model.bert.encoder.layer[ix]) for ix in range(scc_n_layer)])

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate_set=False)
        if args.evaluate_during_training:
            eval_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate_set=True)
        else:
            eval_dataset = None

        # Measure Start Time
        training_start_time = time.monotonic()

        global_step, tr_loss = train(args, train_dataset, eval_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        # Measure End Time
        training_end_time = time.monotonic()

        diff = timedelta(seconds=training_end_time - training_start_time)
        diff_seconds = diff.total_seconds()

        training_parameters = deepcopy(vars(args))
        training_parameters['training_time'] = diff_seconds

        output_training_params_file = os.path.join(args.output_dir, "training_params.json")

        training_parameters.pop('device')
        dictionary_to_json(training_parameters, output_training_params_file)

    #########################
    #       Test model      #
    #########################
    if args.do_eval:
        test_dataset = load_and_cache_examples(args, args.task_name, tokenizer, test_set=True)

        output_dir = args.output_dir

        # Load a trained model and vocabulary that you have fine-tuned
        config = BertConfig.from_pretrained(
            output_dir,
            num_labels=num_labels,
            finetuning_task=args.task_name
        )
        model = OriginalBertForSequenceClassification.from_pretrained(output_dir, config=config)
        tokenizer = BertTokenizer.from_pretrained(output_dir)
        model.to(device)

        eval_start_time = time.monotonic()
        model.eval()
        result, y_logits, y_true = evaluate(args, model, test_dataset, tokenizer)
        eval_end_time = time.monotonic()

        diff = timedelta(seconds=eval_end_time - eval_start_time)
        diff_seconds = diff.total_seconds()
        result['eval_time'] = diff_seconds
        result_to_text_file(result, os.path.join(output_dir, "test_results.txt"))

        y_pred = np.argmax(y_logits, axis=1)
        print('\n\t**** Classification report ****\n')
        print(classification_report(y_true, y_pred, target_names=label_list))

        report = classification_report(y_true, y_pred, target_names=label_list, output_dict=True)
        report['eval_time'] = diff_seconds

        dictionary_to_json(report, os.path.join(output_dir, "test_results.json"))


if __name__ == "__main__":
    main()
