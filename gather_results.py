import argparse
import json
import os
from typing import Any, Dict

import pandas as pd
from transformers import BertConfig

from bert_of_theseus import BertForSequenceClassification
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
    print(MODELS_FOLDER)

    print(models_subdirectories)
    data = list()
    for subdirectory in models_subdirectories:
        try:
            data_dict = gather_results(subdirectory)
            data.append(data_dict)
        except Exception:
            pass

    df = pd.DataFrame(data)
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]
    df.to_csv(os.path.join(DATA_FOLDER, 'results-bert-of-theseus-' + task_name + '.csv'), index=False)


def get_immediate_subdirectories(a_dir):
    return [os.path.join(a_dir, name) for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def gather_results(ft_model_dir: str, task_name: str) -> Dict[str, Any]:
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

    _, lang, domain, kind = task_name.split('_')
    processor = MultiemoProcessor(lang, domain, kind)
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # LOADING THE BEST MODEL
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
    print(data)

    return data


if __name__ == '__main__':
    main()
