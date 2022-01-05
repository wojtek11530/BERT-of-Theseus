import logging
import os
import sys

PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(PROJECT_FOLDER, 'data')

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)

data_dir = os.path.join('data', 'multiemo2')

REP_NUM = 1

max_seq_length = 256
batch_size = 8
num_train_epochs = 1
learning_rate = 5e-5
weight_decay = 0.01
warmup_steps = 0

replacing_rate = 0.3
scheduler_linear_k = 0.00014

mode_level = 'text'
domains = ['reviews']
# domains = ['hotels', 'medicine', 'products', 'reviews']


def main():
    print(PROJECT_FOLDER)
    os.chdir(PROJECT_FOLDER)

    if not os.path.exists(os.path.join(DATA_FOLDER, 'multiemo2')):
        logger.info("Downloading Multiemo data")
        cmd = 'python3 scripts/download_dataset.py --data_dir data/multiemo2'
        run_process(cmd)
        logger.info("Downloading finished")

    if not os.path.exists(os.path.join(DATA_FOLDER, 'models', 'bert-base-uncased')):
        logger.info("Downloading bert-base-uncased model")
        cmd = 'python3 download_bert_base.py'
        run_process(cmd)
        logger.info("Downloading finished")

    # DOMAIN-OUT RUNS
    for domain in domains:
        task_name = f'multiemo_en_N{domain}_{mode_level}'
        eval_task_name = f'multiemo_en_{domain}_{mode_level}'

        if not os.path.exists(os.path.join(DATA_FOLDER, 'models', 'bert-base-uncased', task_name)):
            fine_tune_bert_base(task_name)

        for i in range(REP_NUM):
            train_bert_of_theseus(task_name, do_eval=False)

        models_subdirectories = sorted([x[0] for x in os.walk(os.path.join(DATA_FOLDER, 'models', 'bert-of-theseus'))])
        for subdirectory in models_subdirectories:
            if task_name in subdirectory:
                evaluate_bert_of_theseus(model_path=subdirectory, task_name=eval_task_name)

    # SINGLE DOMAIN RUNS
    for domain in domains:
        task_name = f'multiemo_en_{domain}_{mode_level}'
        if not os.path.exists(os.path.join(DATA_FOLDER, 'models', 'bert-base-uncased', task_name)):
            fine_tune_bert_base(task_name)

        for i in range(REP_NUM):
            train_bert_of_theseus(task_name, do_eval=True)

    # cmd = f'python3 -m gather_results --task_name {task_name}'
    # logger.info(f"Gathering results to csv for {task_name}")
    # run_process(cmd)


def fine_tune_bert_base(task_name: str):
    cmd = 'python3 multiemo_fine_tune_bert.py '
    options = [
        '--pretrained_model', 'data/models/bert-base-uncased',
        '--data_dir', 'data/multiemo2',
        '--task_name', task_name,
        '--output_dir', f'data/models/bert-base-uncased/{task_name}',
        '--num_train_epochs', str(num_train_epochs),
        '--train_batch_size', str(batch_size),
        '--learning_rate', str(learning_rate),
        '--weight_decay', str(weight_decay),
        '--warmup_proportion', str(warmup_steps),
        '--max_seq_length', str(max_seq_length),
        '--do_lower_case'
    ]
    cmd += ' '.join(options)
    logger.info(f"Training bert-base-uncased for {task_name}")
    run_process(cmd)


def train_bert_of_theseus(task_name: str, do_eval: bool = True):
    cmd = 'python3 run_multiemo.py '
    options = [
        '--model_name_or_path ', f'data/models/bert-base-uncased/{task_name}',
        '--data_dir', 'data/multiemo2',
        '--task_name', task_name,
        '--output_dir', 'data/models/bert-of-theseus',
        '--do_train',
        '--evaluate_during_training',
        '--do_lower_case',
        '--max_seq_length', str(max_seq_length),
        '--learning_rate', str(learning_rate),
        '--num_train_epochs', str(num_train_epochs),
        '--weight_decay', str(weight_decay),
        '--per_gpu_train_batch_size', str(batch_size),
        '--per_gpu_eval_batch_size', str(batch_size),
        '--replacing_rate', str(replacing_rate),
        '--scheduler_type', 'linear',
        '--scheduler_linear_k', str(scheduler_linear_k)
    ]
    cmd += ' '.join(options)
    if do_eval:
        cmd += ' --do_eval'

    logger.info(f"Training BERT-OF-THESEUS for {task_name}")
    run_process(cmd)


def evaluate_bert_of_theseus(model_path: str, task_name: str):
    cmd = 'python3 eval_multiemo.py '
    options = [
        '--model_path ', model_path,
        '--data_dir', 'data/multiemo2',
        '--task_name', task_name,
        '--output_dir', model_path,
        '--do_lower_case',
        '--max_seq_length', str(max_seq_length),
        '--per_gpu_eval_batch_size', str(batch_size),
    ]
    cmd += ' '.join(options)

    logger.info(f"Evaluating BERT-OF-THESEUS from {model_path} for {task_name}")
    run_process(cmd)


def run_process(proc):
    os.system(proc)


if __name__ == '__main__':
    main()
