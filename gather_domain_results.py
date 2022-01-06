import argparse
import glob
import json
import os
from typing import Any, Dict, Tuple, List

import pandas as pd
from transformers import BertConfig

from bert_of_theseus import BertForSequenceClassification
from transformers.modeling_bert import BertForSequenceClassification as OriginalBertForSequenceClassification
from data_processing.processors.multiemo import MultiemoProcessor

PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(PROJECT_FOLDER, 'data')
MODELS_FOLDER = os.path.join(DATA_FOLDER, 'models', 'bert-of-theseus')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_level",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task level: either 'text' or 'sentence'.")

    args = parser.parse_args()
    task_level = args.task_level
    if task_level not in ['sentence', 'text']:
        raise ValueError('task_level must be text or sentence')

    models_subdirectories = [x[0] for x in os.walk(MODELS_FOLDER)]
    models_subdirectories = [subdir for subdir in models_subdirectories if is_good_subdir(subdir, task_level)]
    models_subdirectories = sorted(models_subdirectories)

    data = list()
    for subdirectory in models_subdirectories:
        if is_good_subdir(subdirectory, task_level):
            data_dict_list = gather_results(subdirectory)
            for data_dict in data_dict_list:
                data.append(data_dict)

    df = pd.DataFrame(data)
    cols = df.columns.tolist()
    cols = cols[-3:] + cols[:-3]
    df = df[cols]
    df.to_csv(os.path.join(DATA_FOLDER, 'domain-results-bert-of-theseus-' + task_level + '.csv'), index=False)


def is_good_subdir(subdir: str, task_level: str) -> bool:
    return task_level in subdir and '_all_' not in subdir and 'bert-base-uncased' not in subdir and 'TMP' not in subdir


def gather_results(ft_model_dir: str) -> List[Dict[str, Any]]:
    results_dir = list()

    with open(os.path.join(ft_model_dir, 'training_params.json')) as json_file:
        training_data_dict = json.load(json_file)

    training_task_name = training_data_dict['task_name']
    if 'multiemo' not in training_task_name:
        raise ValueError("Task not found: %s" % training_task_name)

    _, lang, domain, kind = training_task_name.split('_')
    processor = MultiemoProcessor(lang, domain, kind)
    label_list = processor.get_labels()
    num_labels = len(label_list)

    model_size = os.path.getsize(os.path.join(ft_model_dir, 'pytorch_model.bin'))

    config = BertConfig.from_pretrained(
        ft_model_dir,
        num_labels=num_labels,
        finetuning_task=training_task_name
    )
    model = OriginalBertForSequenceClassification.from_pretrained(ft_model_dir, config=config)

    memory_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
    memory_buffers = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
    memory_used = memory_params + memory_buffers  # in bytes

    parameters_num = 0
    for n, p in model.named_parameters():
        parameters_num += p.nelement()

    for json_file_path in glob.glob(f"{ft_model_dir}/test_results*.json"):
        with open(json_file_path) as json_file:
            test_data = json.load(json_file)
            if 'micro avg' in test_data:
                acc_val = test_data['micro avg']['f1-score']
                test_data = {key if key != 'micro avg' else 'accuracy': value for key, value in test_data.items()}
                test_data['accuracy'] = acc_val

            [test_data_dict] = pd.json_normalize(test_data, sep='_').to_dict(orient='records')

        results_data = training_data_dict.copy()  # start with keys and values of x
        results_data.update(test_data_dict)

        results_data['model_size'] = model_size
        results_data['memory'] = memory_used
        results_data['parameters'] = parameters_num
        results_data['name'] = os.path.basename(ft_model_dir)

        result_filename = os.path.basename(json_file_path)
        if result_filename == 'test_results.json':
            results_data['eval_task_name'] = training_task_name
        else:
            eval_task_name = result_filename.split('test_results_')[-1].split('.')[0]
            results_data['eval_task_name'] = eval_task_name

        results_data['model_name'] = 'Bert-of-Theseus'

        results_dir.append(results_data)

    return results_dir


if __name__ == '__main__':
    main()
