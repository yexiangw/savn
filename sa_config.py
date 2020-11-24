import argparse
import os
import json
from os.path import join
import logging
import time

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


def set_config():
    parser = argparse.ArgumentParser()

    # use default args
    parser.add_argument("--use_default_args", action='store_true')

    # input & output
    parser.add_argument("--dataset_version", default=None, type=str)
    parser.add_argument("--train_file", default=None, type=str)
    parser.add_argument("--predict_file", default=None, type=str, )
    parser.add_argument("--output_dir", default=None, type=str, )
    parser.add_argument("--cache_dir", default="", type=str,)
    parser.add_argument('--overwrite_output_dir', action='store_true',)
    parser.add_argument('--overwrite_cache', action='store_true',)

    # model
    parser.add_argument("--model_name_or_path", default=None, type=str, )
    parser.add_argument("--config_name", default="", type=str,)
    parser.add_argument("--tokenizer_name", default="", type=str,)
    parser.add_argument("--ans_lambda", default=1, type=int)
    parser.add_argument("--cls_lambda", default=1, type=int)

    # feature
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--max_dialogue_size", default=9, type=int)
    parser.add_argument("--dialogue_stride", default=1, type=int)
    parser.add_argument("--do_lower_case", action='store_true',)

    # run settings
    parser.add_argument("--do_train", action='store_true',)
    parser.add_argument("--use_sp", action='store_true',)
    parser.add_argument("--use_vn", action='store_true',)
    parser.add_argument("--do_eval", action='store_true',)
    parser.add_argument("--evaluate_during_training", action='store_true',)
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,)
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,)
    parser.add_argument('--logging_steps', type=int, default=50,)
    parser.add_argument('--save_steps', type=int, default=1000,)
    parser.add_argument('--eval_steps', type=int, default=1000,)
    parser.add_argument("--eval_all_checkpoints", action='store_true',)
    parser.add_argument("--no_cuda", action='store_true',)
    parser.add_argument('--seed', type=int, default=42,)
    parser.add_argument('--fp16', action='store_true',)
    parser.add_argument('--fp16_opt_level', type=str, default='O1',)

    # optimizer
    parser.add_argument("--learning_rate", default=5e-5, type=float,)
    parser.add_argument("--weight_decay", default=0.0, type=float,)
    parser.add_argument("--warmup_prob", default=0, type=float,)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,)
    parser.add_argument("--max_grad_norm", default=1.0, type=float,)
    parser.add_argument("--num_train_epochs", default=3.0, type=float,)
    parser.add_argument("--max_steps", default=-1, type=int,)

    args = parser.parse_args()

    if args.use_default_args:
        args.dataset_version = '2.1'
        args.train_file = 'data/{}/train_dials.json'.format(args.dataset_version)
        args.predict_file = 'data/{}/dev_dials.json'.format(args.dataset_version)

        args.use_sp = True
        args.use_vn = True
        version = "{}_{}".format("sp" if args.use_sp else "raw", args.dataset_version)
        args.output_dir = "output/SA_{}".format(version)
        args.tensorboard_name = "runs/SA_{}.{}".format(version, time.time())
        args.utils_version = 'SA_{}'.format(version)

        args.model_name_or_path = "bert-base-uncased"
        args.tokenizer_name = "model/bert-base-savn-vocab.txt"
        args.config_name = "model/bert-base-uncased-config.json"

        args.learning_rate = 5e-5
        args.num_train_epochs = 3
        args.max_seq_length = 512
        args.per_gpu_eval_batch_size = 16
        args.per_gpu_train_batch_size = 8
        args.seed = 1017
        # args.do_eval = False
        args.gradient_accumulation_steps = 1
        # args.weight_decay=0.01
        args.warmup_prob = 0.1
        args.eval_steps = 0
        args.save_steps = 0
        args.logging_steps = 50
        args.do_train = True
        args.evaluate_during_training = True

    args.checkpoint_path = join(args.output_dir, "checkpoints")
    args.prediction_path = join(args.output_dir, "predictions")
    save_settings(args)

    return args


def save_settings(args):
    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_path, exist_ok=True)
    os.makedirs(args.prediction_path, exist_ok=True)
    json.dump(args.__dict__, open(join(args.output_dir, "run_settings.json"), 'w'))
