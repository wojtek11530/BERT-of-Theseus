import argparse
import json
import os
from typing import Any, Dict, Tuple

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
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")

    args = parser.parse_args()
    task_name = args.task_name

    models_subdirectories = get_immediate_subdirectories(MODELS_FOLDER)
    models_subdirectories = sorted(models_subdirectories)

    data = list()
    for subdirectory in models_subdirectories:
        if task_name in subdirectory:
            data_dict, data_hg_dict = gather_results(subdirectory, task_name)
            data.append(data_dict)
            data.append(data_hg_dict)

    df = pd.DataFrame(data)
    cols = df.columns.tolist()
    cols = cols[-2:] + cols[:-2]
    df = df[cols]
    df.to_csv(os.path.join(DATA_FOLDER, 'results-bert-of-theseus-' + task_name + '.csv'), index=False)


def get_immediate_subdirectories(a_dir):
    return [os.path.join(a_dir, name) for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def gather_results(ft_model_dir: str, task_name: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    _, lang, domain, kind = task_name.split('_')
    processor = MultiemoProcessor(lang, domain, kind)
    label_list = processor.get_labels()
    num_labels = len(label_list)

    with open(os.path.join(ft_model_dir, 'training_params.json')) as json_file:
        training_data_dict = json.load(json_file)

    with open(os.path.join(ft_model_dir, 'test_results.json')) as json_file:
        test_data = json.load(json_file)
        [test_data_dict] = pd.json_normalize(test_data, sep='_').to_dict(orient='records')

    data = training_data_dict.copy()  # start with keys and values of x
    data.update(test_data_dict)

    model_size = os.path.getsize(os.path.join(ft_model_dir, 'pytorch_model.bin'))
    data['model_size'] = model_size

    if 'multiemo' not in task_name:
        raise ValueError("Task not found: %s" % task_name)

    config = BertConfig.from_pretrained(
        ft_model_dir,
        num_labels=num_labels,
        finetuning_task=task_name
    )
    model = BertForSequenceClassification.from_pretrained(ft_model_dir, config=config)

    memory_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
    memory_buffers = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
    memory_used = memory_params + memory_buffers  # in bytes

    data['memory'] = memory_used

    parameters_num = 0
    for n, p in model.named_parameters():
        parameters_num += p.nelement()

    data['parameters'] = parameters_num
    data['name'] = os.path.basename(ft_model_dir)
    data['model_name'] = 'Bert-of-Theseus'
    print(data)

    # HuggingFace model

    hg_model_dir = os.path.join(ft_model_dir, 'hg_model')

    with open(os.path.join(hg_model_dir, 'test_results.json')) as json_file:
        test_data_hg = json.load(json_file)
        [test_data__hg_dict] = pd.json_normalize(test_data_hg, sep='_').to_dict(orient='records')
    data_hg = training_data_dict.copy()
    data_hg.update(test_data__hg_dict)

    model_size = os.path.getsize(os.path.join(hg_model_dir, 'pytorch_model.bin'))
    data_hg['model_size'] = model_size

    config = BertConfig.from_pretrained(
        hg_model_dir,
        num_labels=num_labels,
        finetuning_task=task_name
    )
    model = OriginalBertForSequenceClassification.from_pretrained(hg_model_dir, config=config)

    memory_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
    memory_buffers = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
    memory_used = memory_params + memory_buffers  # in bytes

    data_hg['memory'] = memory_used

    parameters_num = 0
    for n, p in model.named_parameters():
        parameters_num += p.nelement()

    data_hg['parameters'] = parameters_num
    data_hg['name'] = os.path.basename(ft_model_dir)
    data_hg['model_name'] = 'Bert-of-Theseus-huggingface'
    print(data_hg)
    return data, data_hg


if __name__ == '__main__':
    main()
