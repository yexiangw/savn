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

    # feature
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--do_lower_case", action='store_true',)

    # run settings
    parser.add_argument("--do_train", action='store_true',)
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
        args.model_name_or_path = "bert-base-uncased"
        args.tokenizer_name = "model/bert-base-savn-vocab.txt"
        args.learning_rate = 1e-4
        args.num_train_epochs = 5.0
        args.max_seq_length = 32
        args.seed = 1017
        args.warmup_prob = 0.1
        args.eval_steps = 0
        args.save_steps = 0
        args.do_train = True
        args.logging_steps = 50
        args.per_gpu_eval_batch_size = 32
        args.per_gpu_train_batch_size = 8
        args.gradient_accumulation_steps = 1
        args.config_version = '1'
        args.dataset_version = "2.1"
        args.usage_rate = 1
        args.evaluate_during_training = True
        args.train_all_data = True
        args.config_name = "model/vn_{}-config.json".format(args.config_version)
        args.output_dir = "output/vn_{}_{}_{}".format(args.config_version, args.dataset_version, args.usage_rate)
        args.tensorboard_name = "runs/vn_{}_{}_{}".format(args.config_version, args.dataset_version, args.usage_rate)
        args.use_all_ontology = True

    save_settings(args)

    return args


def save_settings(args):
    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)
    json.dump(args.__dict__, open(join(args.output_dir, "run_settings.json"), 'w'))
