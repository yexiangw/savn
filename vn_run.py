# coding=utf-8

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import glob
import timeit

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset, Dataset)
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from transformers import (WEIGHTS_NAME, BertConfig, BertTokenizer, )

from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule

from vn_utils import get_VN_features, generate_ontology_tensor_file
from vn_model import ValueNormalize
from vn_config import set_config

logger = logging.getLogger(__name__)


class VNDataset(Dataset):
    def __init__(self, all_input_ids, all_attention_masks, all_ontology_input_ids, all_ontology_attention_masks,
                 all_answer_index, all_answer_type):
        self.input_ids = all_input_ids
        self.attention_masks = all_attention_masks
        self.ontology_input_ids = all_ontology_input_ids
        self.ontology_attention_masks = all_ontology_attention_masks
        self.answer_index = all_answer_index
        self.answer_type = all_answer_type
        self.dataset_len = len(all_input_ids)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        item_info = {
            "input_ids": self.input_ids[index],
            "attention_masks": self.attention_masks[index],
            "ontology_input_ids": self.ontology_input_ids[index],
            "ontology_attention_masks": self.ontology_attention_masks[index],
            "answer_index": self.answer_index[index],
            "answer_type": self.answer_type[index],
        }
        return item_info


def collate_fn(data):
    item_info = {}
    max_len = max([len(d["input_ids"]) for d in data])

    item_info["input_ids"] = torch.tensor([d["input_ids"] + [0] * (max_len - len(d["input_ids"])) for d in data])
    item_info["attention_masks"] = torch.tensor(
        [d["attention_masks"] + [0] * (max_len - len(d["attention_masks"])) for d in data])
    item_info["ontology_input_ids"] = [torch.tensor(d["ontology_input_ids"]) for d in data]
    item_info["ontology_attention_masks"] = [torch.tensor(d["ontology_attention_masks"]) for d in data]
    item_info["answer_index"] = torch.tensor([d["answer_index"] for d in data])
    item_info["answer_type"] = torch.tensor([d["answer_type"] for d in data])

    return item_info


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def train(args, model, tokenizer):
    tb_writer = SummaryWriter(args.tensorboard_name)

    train_dataset, _ = load_and_cache_examples(args, tokenizer)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  collate_fn=collate_fn)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_prob * t_total,
                                                num_training_steps=t_total)
    # scheduler = get_constant_schedule(optimizer)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss, eval_best_acc = 0.0, 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            inputs = {'input_ids': batch['input_ids'].to(args.device),
                      'attention_mask': batch['attention_masks'].to(args.device),
                      'ontology_input_ids': [t.to(args.device) for t in batch['ontology_input_ids']],
                      'ontology_attention_masks': [t.to(args.device) for t in batch['ontology_attention_masks']],
                      'answer_index': batch['answer_index'].to(args.device),
                      'answer_type': batch['answer_type'].to(args.device)}

            outputs = model(**inputs)
            loss = outputs[0]
            epoch_iterator.set_description("ls:{:.2f},ea:{:.2f}".format(loss, eval_best_acc), refresh=False)
            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if args.eval_steps > 0 and global_step % args.eval_steps == 0:
                    if args.evaluate_during_training:
                        acc = evaluate(args, model, tokenizer)
                        tb_writer.add_scalar('acc', acc, global_step)
                        eval_best_acc = max(eval_best_acc, acc)

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix=""):
    _, dataset = load_and_cache_examples(args, tokenizer)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate_fn)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    start_time = timeit.default_timer()
    all_counts, correct_counts, type_correct_counts = 0, 0, 0
    for step, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
        model.eval()
        with torch.no_grad():
            inputs = {'input_ids': batch['input_ids'].to(args.device),
                      'attention_mask': batch['attention_masks'].to(args.device),
                      'ontology_input_ids': [t.to(args.device) for t in batch['ontology_input_ids']],
                      'ontology_attention_masks': [t.to(args.device) for t in batch['ontology_attention_masks']],
                      }
            logits = model(**inputs)[0]
            answer_index = batch['answer_index']
            predict_index = torch.tensor([torch.argmax(logit) for logit in logits])
            correct_counts += torch.sum(predict_index == answer_index).item()
            all_counts += answer_index.size(0)

    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

    acc = correct_counts / all_counts
    logger.info("Acc:{}".format(acc))
    return acc


def load_and_cache_examples(args, tokenizer, ):
    features_path = 'data/{}/vn_features_{}'.format(args.dataset_version, args.usage_rate)
    if os.path.exists(features_path):
        features = torch.load(features_path)
    else:
        features = get_VN_features(tokenizer, args.dataset_version, args.usage_rate)
        torch.save(features, features_path)

    # Convert to Tensors and build dataset
    all_input_ids = [f.input_ids for f in features]
    all_attention_masks = [f.attention_masks for f in features]
    all_ontology_input_ids = [f.ontology_input_ids for f in features]
    all_ontology_attention_masks = [f.ontology_attention_masks for f in features]

    all_answer_index = torch.tensor([f.answer_index for f in features], dtype=torch.long)
    all_answer_type = torch.tensor([f.answer_type for f in features], dtype=torch.long)
    dataset = VNDataset(all_input_ids, all_attention_masks, all_ontology_input_ids,
                        all_ontology_attention_masks, all_answer_index, all_answer_type)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    if args.train_all_data:
        return dataset, dataset
    else:
        return torch.utils.data.random_split(dataset, [train_size, test_size])


def main():
    args = set_config()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    set_seed(args)

    logger.warning("Process device: %s, n_gpu: %s, 16-bits training: %s", args.device, args.n_gpu)
    logger.info("Training/evaluation parameters %s", args)

    if args.fp16:
        try:
            import apex
            apex.amp.register_half_function(torch, 'einsum')
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    config = BertConfig.from_pretrained(args.config_name)
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_name,
                                              do_lower_case=args.do_lower_case)

    logger.info("Training/evaluation parameters %s", args)

    if args.do_train:
        model = ValueNormalize.from_pretrained(args.model_name_or_path, config=config, )
        model.to(args.device)
        train(args, model, tokenizer)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
        generate_ontology_tensor_file(args.config_version, args.dataset_version, args.usage_rate)

    results = {}
    if args.do_eval:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)
        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            model = ValueNormalize.from_pretrained(checkpoint)
            model.to(args.device)

            # Evaluate
            result = evaluate(args, model, tokenizer, prefix=global_step)

            print(result)
            results[global_step] = result

    logger.info("Results: {}".format(results))

    return results


if __name__ == "__main__":
    main()
