import json
import logging
from io import open
from tqdm import tqdm
import os
import pickle
from transformers.tokenization_bert import BertTokenizer
from transformers import BertModel, BertConfig
from vn_model import ValueNormalize
import torch
from torch import nn
from utils.sp_label import *
from vn_utils import get_answer_from_vn
import collections

logger = logging.getLogger(__name__)

EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]


class MultiWozExample(object):
    def __init__(self,
                 dialogue_idx,
                 domains,
                 turn_idx,
                 doc_tokens,
                 orig_value_text=None,
                 example_domains=None):
        self.dialogue_idx = dialogue_idx
        self.domains = domains
        self.turn_idx = turn_idx
        self.doc_tokens = doc_tokens
        self.orig_value_text = orig_value_text
        self.example_domains = example_domains


class MultiWozFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 dialogue_idx,
                 turn_idx,
                 input_ids,
                 attention_masks,
                 token_type_ids,
                 value_types=None,
                 start_positions=None,
                 end_positions=None,
                 domains=None,
                 raw_tokens=None,
                 tok_to_orig_index=None,
                 all_doc_tokens=None, ):
        self.unique_id = unique_id
        self.dialogue_idx = dialogue_idx
        self.turn_idx = turn_idx
        self.raw_tokens = raw_tokens
        self.tok_to_orig_index = tok_to_orig_index
        self.all_doc_tokens = all_doc_tokens
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.token_type_ids = token_type_ids
        self.value_types = value_types
        self.start_positions = start_positions
        self.end_positions = end_positions
        self.domains = domains


RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "types", "start_logits", "end_logits"])


def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def basic_tokenize(content):
    tokens = []
    prev_is_whitespace = True
    for c in content:
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                tokens.append(c)
            else:
                tokens[-1] += c
            prev_is_whitespace = False
    return tokens


def read_multiwoz_examples(input_file, is_training, data_version, max_dialogue_size=9):
    slots = get_slots(data_version)
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)

    examples = []
    for entry in input_data:
        dialogues = entry["dialogue"]
        for i, turn in enumerate(dialogues):
            content = ""
            example_belief = {}
            example_domain = []
            if i < max_dialogue_size:
                for j in range(i + 1):
                    if dialogues[j]['system_transcript']:
                        content = " [SYS] " + dialogues[j]['system_transcript'] + content
                    if dialogues[j]['transcript']:
                        content = " [USER] " + dialogues[j]['transcript'] + content
                    if dialogues[j]["domain"] not in example_domain:
                        example_domain.append(dialogues[j]["domain"])
            else:
                for j in range(max_dialogue_size - 1, -1, -1):
                    if dialogues[i - j]['system_transcript']:
                        content = " [SYS] " + dialogues[i - j]['system_transcript'] + content
                    if dialogues[i - j]['transcript']:
                        content = " [USER] " + dialogues[i - j]['transcript'] + content
                    if dialogues[i - j]["domain"] not in example_domain:
                        example_domain.append(dialogues[j]["domain"])
            content += ' [SEP] '

            if is_training:
                if i < max_dialogue_size:
                    for j in range(i + 1):
                        for label in dialogues[j]["turn_label"]:
                            example_belief[label[0]] = label[1]
                else:
                    for j in range(max_dialogue_size - 1, -1, -1):
                        for label in dialogues[i - j]["turn_label"]:
                            example_belief[label[0]] = label[1]
            else:
                for label in dialogues[i]["belief_state"]:
                    example_belief[label["slots"][0][0]] = label["slots"][0][1]

            orig_value_text = ["none"] * len(slots)
            for slot_index, slot in enumerate(slots):
                if slot in example_belief:
                    orig_value_text[slot_index] = example_belief[slot]

            for slot in example_belief:
                if slot not in slots and slot.split("-")[0] in EXPERIMENT_DOMAINS:
                    print("Error, there is unknown slot in belief states!")

            doc_tokens = basic_tokenize(content)

            example = MultiWozExample(
                dialogue_idx=entry['dialogue_idx'],
                domains=entry['domains'],
                turn_idx=turn['turn_idx'],
                doc_tokens=doc_tokens,
                orig_value_text=orig_value_text,
                example_domains=example_domain)
            examples.append(example)
    return examples


def convert_examples_to_features(examples, tokenizer, is_training, use_sp, max_seq_length=512):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000

    features = []
    for (example_index, example) in enumerate(tqdm(examples)):
        token_type_ids = []
        candidate_answer_tokens = ['[CLS]', 'yes', '[ANSWER_SEP]', 'no', '[ANSWER_SEP]',
                                   'cambridge', '[ANSWER_SEP]', '1', '[SEP]']   # Implied place and me

        raw_tokens = candidate_answer_tokens + example.doc_tokens

        tok_to_orig_index = []
        all_doc_tokens = []
        orig_to_tok_index = [0]

        for (i, token) in enumerate(raw_tokens):
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

            orig_to_tok_index.append(len(all_doc_tokens))

        input_ids = tokenizer.convert_tokens_to_ids(all_doc_tokens)
        attention_masks = [1] * len(input_ids)
        token_type_ids += [1] * len(input_ids)

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            attention_masks.append(0)
            token_type_ids.append(0)
            tok_to_orig_index.append(0)

        assert len(input_ids) == max_seq_length
        assert len(attention_masks) == max_seq_length
        assert len(token_type_ids) == max_seq_length
        assert len(tok_to_orig_index) == max_seq_length

        value_types = None
        start_positions = None
        end_positions = None
        domains = None
        if is_training:
            value_types = [0] * len(example.orig_value_text)
            start_positions = [-1] * len(example.orig_value_text)
            end_positions = [-1] * len(example.orig_value_text)
            domains = [0] * len(EXPERIMENT_DOMAINS)
            for domain in example.example_domains:
                if domain in EXPERIMENT_DOMAINS:
                    domains[EXPERIMENT_DOMAINS.index(domain)] = 1

            for value_index, value in enumerate(example.orig_value_text):
                if value == 'none':
                    value_types[value_index] = 0
                    start_positions[value_index], end_positions[value_index] = [-1, -1]
                elif value == "dontcare":
                    value_types[value_index] = 1
                    start_positions[value_index], end_positions[value_index] = [-1, -1]
                else:
                    raw_start_index, raw_end_index, type_flag = search_value_for_index(raw_tokens, value,
                                                                                       example.dialogue_idx, use_sp)
                    if raw_start_index >= 0:
                        start_positions[value_index], end_positions[value_index] = orig_to_tok_index[raw_start_index], \
                                                                                   orig_to_tok_index[
                                                                                       raw_end_index + 1] - 1
                    else:
                        start_positions[value_index], end_positions[value_index] = -1, -1

                    value_types[value_index] = type_flag

        features.append(
            MultiWozFeatures(
                unique_id=unique_id,
                dialogue_idx=example.dialogue_idx,
                turn_idx=example.turn_idx,
                raw_tokens=raw_tokens,
                tok_to_orig_index=tok_to_orig_index,
                all_doc_tokens=all_doc_tokens,
                input_ids=input_ids,
                attention_masks=attention_masks,
                token_type_ids=token_type_ids,
                value_types=value_types,
                start_positions=start_positions,
                end_positions=end_positions,
                domains=domains
            ))
        unique_id += 1

    return features


def search_value_for_index(tokens, value, dialogue_idx, use_sp):
    value_tokens = basic_tokenize(value)
    search_len = len(tokens) - len(value_tokens)
    for i in range(search_len):
        if tokens[i:i + len(value_tokens)] == value_tokens:
            return i, i + len(value_tokens) - 1, 2

    if use_sp:
        if value in sp_common_dict_1:
            fixed_value = sp_common_dict_1[value]
            value_tokens = basic_tokenize(fixed_value)
            for i in range(search_len):
                if tokens[i:i + len(value_tokens)] == value_tokens:
                    return i, i + len(value_tokens) - 1, 2

        if dialogue_idx in sp_dict_2 and value in sp_dict_2[dialogue_idx]:
            fixed_value = sp_dict_2[dialogue_idx][value]
            value_tokens = basic_tokenize(fixed_value)
            for i in range(search_len):
                if tokens[i:i + len(value_tokens)] == value_tokens:
                    return i, i + len(value_tokens) - 1, 2

        if dialogue_idx in sp_dict_3 and value in sp_dict_3[dialogue_idx]:
            fixed_value = sp_dict_3[dialogue_idx][value]
            value_tokens = basic_tokenize(fixed_value)
            for i in range(search_len):
                if tokens[i:i + len(value_tokens)] == value_tokens:
                    return i, i + len(value_tokens) - 1, 2

        if value in sp_common_multi_dict:
            for fixed_value in sp_common_multi_dict[value]:
                value_tokens = basic_tokenize(fixed_value)
                for i in range(search_len):
                    if tokens[i:i + len(value_tokens)] == value_tokens:
                        return i, i + len(value_tokens) - 1, 2

        if dialogue_idx in sp_dict and value in sp_dict[dialogue_idx]:
            fixed_value = sp_dict[dialogue_idx][value]
            value_tokens = basic_tokenize(fixed_value)
            for i in range(search_len):
                if tokens[i:i + len(value_tokens)] == value_tokens:
                    return i, i + len(value_tokens) - 1, 2
        #
        if dialogue_idx in skip_train_label_2 and value in skip_train_label_2[dialogue_idx]:
            return -1, -1, 2

        if dialogue_idx in skip_train_label and value in skip_train_label[dialogue_idx]:
            return -1, -1, 0

    return -1, -1, 2


def multiwoz_evaluate(all_results, examples, features, dataset_version, n_best_size=20, max_answer_length=10,
                      max_dialogue_len=9, dialogue_stride=1, use_vn=False, use_n_best=True,
                      usage_rate=1, theta=-1):
    if use_vn:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vn_config_version = "1"
        usage_rate = usage_rate
        theta = theta
        config = BertConfig.from_pretrained('model/vn_{}-config.json'.format(vn_config_version))
        tokenizer = BertTokenizer.from_pretrained('model/bert-base-uncased-vocab.txt', do_lower_case=False)
        model = ValueNormalize.from_pretrained(
            'output/vn_{}_{}_{}/pytorch_model.bin'.format(vn_config_version, dataset_version, usage_rate),
            config=config)
        model.to(device)
        model.eval()

        ontology = json.load(open("data/{}/ontology_{}.json".format(dataset_version, usage_rate), 'r'))
        ontology_embeds = torch.load(
            'data/{}/ontology_embed_{}_{}'.format(dataset_version, vn_config_version, usage_rate))

    slots = get_slots(dataset_version)
    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    special_tokens = ["[ANSWER_SEP]", "[SEP]", "[USER]", "[SYS]"]
    all_predict_results = {}
    for feature in tqdm(features):
        if feature.dialogue_idx not in all_predict_results:
            all_predict_results[feature.dialogue_idx] = {}
        result = unique_id_to_result[feature.unique_id]
        belief_states = []
        for slot_index, slot in enumerate(slots):
            if result.types[slot_index] == 0:
                belief_states.append('none')
            elif result.types[slot_index] == 1:
                belief_states.append('dontcare')
            else:
                if use_n_best:
                    start_indexes = _get_best_indexes(result.start_logits[slot_index], n_best_size)
                    end_indexes = _get_best_indexes(result.end_logits[slot_index], n_best_size)
                    n_best_spans = []
                    for start_index in start_indexes:
                        for end_index in end_indexes:
                            if start_index >= len(feature.all_doc_tokens):
                                continue
                            if end_index >= len(feature.all_doc_tokens):
                                continue
                            if end_index < start_index:
                                continue
                            length = end_index - start_index + 1
                            if length > max_answer_length:
                                continue
                            if any(special_token in feature.all_doc_tokens[start_index:end_index + 1] for special_token
                                   in
                                   special_tokens):
                                continue
                            n_best_spans.append({
                                "start_index": start_index,
                                "end_index": end_index,
                                "score": result.start_logits[slot_index][start_index] + result.end_logits[slot_index][
                                    end_index]}
                            )

                    if not n_best_spans:
                        n_best_spans.append({"start_index": 1, "end_index": 1, "score": 0.0})

                    assert len(n_best_spans) >= 1

                    n_best_spans.sort(key=lambda x: x["score"], reverse=True)
                    best_start_index = n_best_spans[0]["start_index"]
                    best_end_index = n_best_spans[0]["end_index"]
                else:
                    best_start_index = _get_best_indexes(result.start_logits[slot_index], 1)[0]
                    best_end_index = _get_best_indexes(result.end_logits[slot_index], 1)[0]

                best_value = " ".join(feature.raw_tokens[
                                      feature.tok_to_orig_index[best_start_index]:feature.tok_to_orig_index[
                                                                                      best_end_index] + 1])
                if use_vn:
                    answer_from_an, gate = get_answer_from_vn(
                        feature.all_doc_tokens[best_start_index:best_end_index + 1],
                        ontology_embeds[slot], ontology[slot], model,
                        tokenizer, device)
                    if gate > 1:
                        gate = 1
                    if gate > theta:
                        belief_states.append(answer_from_an)
                    else:
                        belief_states.append(best_value)
                else:
                    belief_states.append(best_value)

        assert len(belief_states) == len(slots)
        all_predict_results[feature.dialogue_idx][feature.turn_idx] = belief_states

    for dialogue_idx, dialogue in all_predict_results.items():
        for turn_idx, belief_state in dialogue.items():
            while turn_idx > max_dialogue_len - 1:
                turn_idx -= dialogue_stride
                for i in range(len(slots)):
                    if belief_state[i] == 'none' and dialogue[turn_idx][i] != 'none':
                        belief_state[i] = dialogue[turn_idx][i]

    total_count, correct_count, slot_correct_count, slot_total_count, slot_cls_correct_count = 0, 0, 0, 0, 0
    max_correct_count = 0
    error_cls_slot_count = {slots[i]: 0 for i in range(30)}

    for example in examples:

        belief_state = all_predict_results[example.dialogue_idx][example.turn_idx]
        if belief_state == example.orig_value_text:
            correct_count += 1

        max_flag = 1
        for index, value in enumerate(example.orig_value_text):
            if belief_state[index] == value:
                slot_correct_count += 1
                slot_cls_correct_count += 1

            elif belief_state[index] != "none" and value != "none":
                slot_cls_correct_count += 1
            else:
                max_flag = 0

            if belief_state[index] != value:
                error_cls_slot_count[slots[index]] += 1

            slot_total_count += 1

        max_correct_count += max_flag
        total_count += 1
    joint_acc, slot_acc, cls_acc, max_acc = correct_count / total_count, slot_correct_count / slot_total_count, slot_cls_correct_count / slot_total_count, max_correct_count / total_count

    print("joint acc: {}, correct_count: {}, total_count: {}".format(joint_acc, correct_count,
                                                                     total_count))
    print("slot acc: {}, correct_count: {}, total_count: {}".format(slot_acc,
                                                                    slot_correct_count, slot_total_count), )
    print("cls acc: {}, correct_count: {}, total_count: {}".format(cls_acc,
                                                                   slot_cls_correct_count, slot_total_count), )
    print(sorted(error_cls_slot_count.items(), key=lambda x: x[1], reverse=True))

    print("max_acc:{}".format(max_acc, max_correct_count, total_count))
    return joint_acc, slot_acc, cls_acc, max_acc


def get_slot_input_ids(tokenizer, dataset_version):
    slots = get_slots(dataset_version)
    slots_input_ids = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(slot)) for slot in slots]
    slots_attention_masks = [[1] * len(slot) for slot in slots_input_ids]
    max_len = max([len(slot) for slot in slots_input_ids])
    for index, slot in enumerate(slots_input_ids):
        while len(slot) < max_len:
            slots_input_ids[index].append(0)
            slots_attention_masks[index].append(0)
    return [slots_input_ids, slots_attention_masks]


def get_slots(dataset_version):
    if os.path.exists("data/{}/slots.pkl".format(dataset_version)):
        slots = pickle.load(open("data/{}/slots.pkl".format(dataset_version), "rb"))
    else:
        ontology = json.load(open("data/{}/ontology.json".format(dataset_version), 'r'))
        if dataset_version == "2.1":
            ontology_domains = dict([(k, v) for k, v in ontology.items() if k.split("-")[0] in EXPERIMENT_DOMAINS])
            slots = [k.replace("-semi", "").lower() if ("book" not in k) else k.replace("book-", "book ").lower() for k
                     in
                     ontology_domains.keys()]
        else:
            ontology_domains = dict([(k, v) for k, v in ontology.items() if k.split("-")[0] in EXPERIMENT_DOMAINS])
            slots = [k.replace(" ", "").lower() if ("book" not in k) else k.lower() for k in ontology_domains.keys()]

        pickle.dump(slots, open("data/{}/slots.pkl".format(dataset_version), "wb"))
    return slots


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes
