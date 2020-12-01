import argparse
import logging
import os
import random
import glob
import timeit
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from transformers import (WEIGHTS_NAME, BertConfig, BertTokenizer, AdamW, get_linear_schedule_with_warmup,
                          get_constant_schedule_with_warmup)
from sa_model import SlotAttention
from sa_utils import (read_multiwoz_examples, convert_examples_to_features, get_slot_input_ids, multiwoz_evaluate,
                      RawResult)
from sa_config import set_config

logger = logging.getLogger(__name__)


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
    train_dataset, slot_input_ids = load_dataset(args, tokenizer, is_training=True, output_examples=False)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

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
    # scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps)

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
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", )
    set_seed(args)

    slot_input_ids = [i.repeat(args.n_gpu, 1, 1) for i in slot_input_ids]

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'value_types': batch[3],
                      'start_positions': batch[4],
                      'end_positions': batch[5],
                      'slot_input_ids': slot_input_ids, }

            outputs = model(**inputs)
            loss = outputs[0]
            cls_loss, start_loss, end_loss = outputs[1:4]
            if args.n_gpu > 1:
                loss = loss.mean()
                cls_loss = cls_loss.mean()
                start_loss = start_loss.mean()
                end_loss = end_loss.mean()
            epoch_iterator.set_description(
                "ls:{:.2f},cl:{:.2f},sl:{:.2f},el:{:.2f},ea:{:.2f}".format(loss, cls_loss, start_loss,
                                                                           end_loss, eval_best_acc), refresh=False)
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
                        joint_acc, slot_acc, cls_acc, max_acc = evaluate(args, model, tokenizer,
                                                                         prefix=str(global_step))
                        tb_writer.add_scalar('joint_acc', joint_acc, global_step)
                        tb_writer.add_scalar('slot_acc', slot_acc, global_step)
                        tb_writer.add_scalar('cls_acc', cls_acc, global_step)
                        tb_writer.add_scalar('max_acc', max_acc, global_step)
                        eval_best_acc = max(eval_best_acc, joint_acc)

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                    tb_writer.add_scalar('cls_loss', cls_loss, global_step)
                    tb_writer.add_scalar('start_loss', start_loss, global_step)
                    tb_writer.add_scalar('end_loss', end_loss, global_step)
                    logging_loss = tr_loss

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model,
                                                            'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

        if args.evaluate_during_training:
            joint_acc, slot_acc, cls_acc, max_acc = evaluate(args, model, tokenizer)
            tb_writer.add_scalar('joint_acc', joint_acc, global_step)
            tb_writer.add_scalar('slot_acc', slot_acc, global_step)
            tb_writer.add_scalar('cls_acc', cls_acc, global_step)
            tb_writer.add_scalar('max_acc', max_acc, global_step)
    tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix=""):
    dataset, examples, features, slot_input_ids = load_dataset(args, tokenizer, is_training=False,
                                                               output_examples=True)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    slot_input_ids = [i.repeat(args.n_gpu, 1, 1) for i in slot_input_ids]
    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    # all_results = {}
    all_results = []
    start_time = timeit.default_timer()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'slot_input_ids': slot_input_ids,
                      }
            example_indices = batch[3]
            outputs = model(**inputs)

        for i, example_index in enumerate(example_indices):
            feature = features[example_index.item()]
            result = RawResult(unique_id=int(feature.unique_id),
                               types=outputs[0][i].argmax(dim=-1),
                               start_logits=to_list(outputs[1][i]),
                               end_logits=to_list(outputs[2][i]))
            all_results.append(result)

    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))
    results = multiwoz_evaluate(all_results, examples, features, args.dataset_version, use_vn=args.use_vn)
    return results


def load_dataset(args, tokenizer, is_training=True, output_examples=False):
    input_file = args.train_file if is_training else args.predict_file
    cached_features_file = os.path.join(os.path.dirname(input_file), 'cached_features_{}_{}'.format(
        'train' if is_training else 'eval', args.utils_version))
    cached_examples_file = os.path.join(os.path.dirname(input_file), 'cached_examples_{}_{}'.format(
        'train' if is_training else 'eval', args.utils_version))

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        examples = torch.load(cached_examples_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", input_file)
        examples = read_multiwoz_examples(input_file, is_training, args.dataset_version, args.max_dialogue_size)
        features = convert_examples_to_features(examples=examples,
                                                tokenizer=tokenizer,
                                                max_seq_length=args.max_seq_length,
                                                is_training=is_training,
                                                use_sp=args.use_sp)

        logger.info("Saving examples into cached file %s", cached_examples_file)
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(examples, cached_examples_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_masks = torch.tensor([f.attention_masks for f in features], dtype=torch.float)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)

    if is_training:
        all_value_types = torch.tensor([f.value_types for f in features], dtype=torch.long)
        all_start_positions = torch.tensor([f.start_positions for f in features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_positions for f in features], dtype=torch.long)
        all_domains = torch.tensor([f.domains for f in features], dtype=torch.float)
        dataset = TensorDataset(all_input_ids, all_attention_masks, all_token_type_ids, all_value_types,
                                all_start_positions, all_end_positions, all_domains)
    else:
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_attention_masks, all_token_type_ids,
                                all_example_index)

    slots_input = get_slot_input_ids(tokenizer, args.dataset_version)
    slots_input_ids = torch.tensor(slots_input[0], dtype=torch.long, device=args.device)
    slots_attention_masks = torch.tensor(slots_input[1], dtype=torch.long, device=args.device)
    if output_examples:
        return dataset, examples, features, (slots_input_ids, slots_attention_masks)
    return dataset, (slots_input_ids, slots_attention_masks)


def main():
    args = set_config()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    set_seed(args)
    logger.warning("Process device: %s, n_gpu: %s, 16-bits training: %s", args.device, args.n_gpu, args.fp16)
    logger.info("Training/evaluation parameters %s", args)

    if args.fp16:
        try:
            import apex
            apex.amp.register_half_function(torch, 'einsum')
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    config = BertConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                        num_labels=3)
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                              do_lower_case=args.do_lower_case)
    tokenizer.add_special_tokens({'additional_special_tokens': ['[ANSWER_SEP]', '[SYS]', '[USER]']})

    if args.do_train:
        model = SlotAttention.from_pretrained(args.model_name_or_path,
                                              from_tf=bool('.ckpt' in args.model_name_or_path),
                                              config=config,
                                              cache_dir=args.cache_dir if args.cache_dir else None,
                                              args=args)
        model.to(args.device)
        train(args, model, tokenizer)
        logger.info("Saving model checkpoint to %s", args.output_dir)
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

    results = {}
    if args.do_eval:
        checkpoints = [args.output_dir]
        tb_writer = None
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/*/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)
            checkpoints.sort(key=lambda x: int(x.split('-')[-1]))
            tb_writer = SummaryWriter(args.tensorboard_name)

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            config = BertConfig.from_pretrained(checkpoint, num_labels=3)
            model = SlotAttention.from_pretrained(checkpoint, config=config, args=args)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=global_step)

            if args.eval_all_checkpoints:
                global_step = int(global_step)
                tb_writer.add_scalar('joint_acc', result[0], global_step)
                tb_writer.add_scalar('slot_acc', result[1], global_step)
                tb_writer.add_scalar('cls_acc', result[2], global_step)
                tb_writer.add_scalar('max_acc', result[3], global_step)
            results[global_step] = result
        if args.eval_all_checkpoints:
            tb_writer.close()

    logger.info("Results: {}".format(results))

    return results


if __name__ == "__main__":
    main()
