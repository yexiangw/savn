import json
from transformers import BertTokenizer, BertConfig
from utils.sp_label import *
from tqdm import tqdm
import torch
from vn_model import ValueNormalize
import os
import random

EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]


class ValueNormalizeFeatures(object):
    def __init__(self,
                 unique_id,
                 dialogue_idx,
                 turn_idx,
                 supporting_span_tokens,
                 input_ids,
                 attention_masks,
                 ontology_input_ids,
                 ontology_attention_masks,
                 answer_index=None,
                 answer_type=None):
        self.unique_id = unique_id
        self.dialogue_idx = dialogue_idx
        self.turn_idx = turn_idx
        self.supporting_span_tokens = supporting_span_tokens
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.ontology_input_ids = ontology_input_ids
        self.ontology_attention_masks = ontology_attention_masks
        self.answer_index = answer_index
        self.answer_type = answer_type


def generate_ontology(dataset_version, usage_rate):
    train_data = json.load(open('data/{}/train_dials.json'.format(dataset_version), 'r'))
    dev_data = json.load(open('data/{}/dev_dials.json'.format(dataset_version), 'r'))
    test_data = json.load(open('data/{}/test_dials.json'.format(dataset_version), 'r'))
    all_ontology = {}
    for dataset in [train_data, dev_data, test_data]:
        for data in dataset:
            for turn in data['dialogue']:
                for slot in turn['turn_label']:
                    if slot[0] in all_ontology and slot[1] not in all_ontology[slot[0]]:
                        all_ontology[slot[0]].append(slot[1])
                    elif slot[0] not in all_ontology:
                        all_ontology[slot[0]] = [slot[1]]
    if usage_rate ==1:
        json.dump(all_ontology, open('data/{}/ontology_{}.json'.format(dataset_version, usage_rate), 'w'))
    else:
        _ontology = {}
        for slot in all_ontology:
            values = all_ontology[slot]
            if len(values) > 20:
                _ontology[slot] = random.sample(values, int(len(values) * usage_rate))
            else:
                _ontology[slot] = values

        json.dump(_ontology, open('data/{}/ontology_{}.json'.format(dataset_version, usage_rate), 'w'))


def generate_VN_dataset(tokenizer, dataset_version):
    an_dataset = []
    train_data = json.load(open('data/{}/train_dials.json'.format(dataset_version), 'r'))
    value_count = 0
    for data in train_data:
        content = ['[CLS]', 'yes', '[ANSWER_SEP]', 'no', '[ANSWER_SEP]', 'not', 'care', '[ANSWER_SEP]', 'none',
                   '[ANSWER_SEP]', 'cambridge', '[ANSWER_SEP]', '1', '[SEP]']
        for turn in data['dialogue']:
            content += tokenizer.tokenize(turn['system_transcript']) + tokenizer.tokenize(turn['transcript'])
            for slot in turn['turn_label']:
                if slot[1] != 'none':
                    value_count += 1

                if slot[0].split('-')[0] not in EXPERIMENT_DOMAINS:
                    continue
                fixed_value, fixed_flag = search_value_for_index(content, slot[1], tokenizer, data['dialogue_idx'])

                if fixed_flag:
                    an_dataset.append({
                        'dialogue_idx': data['dialogue_idx'],
                        'turn_idx': turn['turn_idx'],
                        'slot_name': slot[0],
                        'supporting_span': fixed_value,
                        'correct_value': slot[1]
                    })
    print(value_count)
    print(len(an_dataset))
    json.dump(an_dataset, open('data/{}/VN_dataset.json'.format(dataset_version), 'w'), indent=4)


def get_VN_features(tokenizer, dataset_version, usage_rate):
    dataset_path = 'data/{}/VN_dataset.json'.format(dataset_version)
    if not os.path.exists(dataset_path):
        generate_VN_dataset(tokenizer, dataset_version)

    examples = json.load(open(dataset_path.format(dataset_version), "r"))

    unique_id = 1000000000

    max_same_count = 5
    examples_count = {}

    features = []
    ontology_input_ids_dict = {}
    ontology_attention_masks_dict = {}
    ontology_path = "data/{}/ontology_{}.json".format(dataset_version, usage_rate)
    if not os.path.exists(ontology_path):
        generate_ontology(dataset_version, usage_rate)
    ontology = json.load(
        open(ontology_path, 'r'))
    type1_count, type0_count, fixed_count, unfixed_count = 0, 0, 0, 0

    for key in ontology:
        ontology_input_ids_dict[key] = [
            tokenizer.convert_tokens_to_ids(['[CLS]'] + tokenizer.tokenize(raw_answer) + ['[SEP]']) for raw_answer in
            ontology[key]]
        ontology_attention_masks_dict[key] = [[1] * len(raw_answer_ids) for raw_answer_ids in
                                              ontology_input_ids_dict[key]]
        max_len = max([len(raw_answer_ids) for raw_answer_ids in ontology_input_ids_dict[key]])
        for index, raw_answer_ids in enumerate(ontology_input_ids_dict[key]):
            while len(raw_answer_ids) < max_len:
                ontology_input_ids_dict[key][index].append(0)
                ontology_attention_masks_dict[key][index].append(0)

    for (example_index, example) in enumerate(tqdm(examples)):
        supporting_span_tokens = ['[CLS]'] + tokenizer.tokenize(example['supporting_span']) + ['[SEP]']
        input_ids = tokenizer.convert_tokens_to_ids(supporting_span_tokens)
        attention_masks = [1] * len(input_ids)

        if example['correct_value'] in ontology[example['slot_name']]:
            type1_count += 1
            answer_type = 1
            answer_index = ontology[example['slot_name']].index(example['correct_value'])
        else:
            type0_count += 1
            answer_type = 0
            answer_index = -1

        if example['slot_name'] in examples_count:
            if example['supporting_span'] in examples_count[example['slot_name']]:
                examples_count[example['slot_name']][example['supporting_span']] += 1
            else:
                examples_count[example['slot_name']][example['supporting_span']] = 1
        else:
            examples_count[example['slot_name']] = {}
            examples_count[example['slot_name']][example['supporting_span']] = 1

        if example['correct_value'] != example['supporting_span'] or examples_count[example['slot_name']][
            example['supporting_span']] <= max_same_count:
            features.append(
                ValueNormalizeFeatures(
                    unique_id=unique_id,
                    dialogue_idx=example['dialogue_idx'],
                    turn_idx=example['turn_idx'],
                    supporting_span_tokens=supporting_span_tokens,
                    input_ids=input_ids,
                    attention_masks=attention_masks,
                    ontology_input_ids=ontology_input_ids_dict[example['slot_name']],
                    ontology_attention_masks=ontology_attention_masks_dict[example['slot_name']],
                    answer_index=answer_index,
                    answer_type=answer_type,
                ))
            unique_id += 1
            if example['correct_value'] != example['supporting_span']:
                fixed_count += 1
            else:
                unfixed_count += 1
    print("in ontology", type1_count)
    print("not in ontology", type0_count)
    print("not same:", fixed_count)
    print("same:", unfixed_count)
    print("total:", unique_id - 1000000000)
    return features


def search_value_for_index(tokens, value, tokenizer, dialogue_idx):
    value_tokens = tokenizer.tokenize(value)
    for i in range(len(tokens) - len(value_tokens), -1, -1):
        if tokens[i:i + len(value_tokens)] == value_tokens:
            return value, 1

    if value in sp_common_dict_1:
        fixed_value = sp_common_dict_1[value]
        value_tokens = tokenizer.tokenize(fixed_value)
        for i in range(len(tokens) - len(value_tokens), -1, -1):
            if tokens[i:i + len(value_tokens)] == value_tokens:
                return fixed_value, 1

    if value in sp_common_multi_dict:
        for fixed_value in sp_common_multi_dict[value]:
            value_tokens = tokenizer.tokenize(fixed_value)
            for i in range(len(tokens) - len(value_tokens), -1, -1):
                if tokens[i:i + len(value_tokens)] == value_tokens:
                    return fixed_value, 1

    if dialogue_idx in sp_dict_2 and value in sp_dict_2[dialogue_idx]:
        fixed_value = sp_dict_2[dialogue_idx][value]
        value_tokens = tokenizer.tokenize(fixed_value)
        for i in range(len(tokens) - len(value_tokens), -1, -1):
            if tokens[i:i + len(value_tokens)] == value_tokens:
                return fixed_value, 1

    if dialogue_idx in sp_dict_3 and value in sp_dict_3[dialogue_idx]:
        fixed_value = sp_dict_3[dialogue_idx][value]
        value_tokens = tokenizer.tokenize(fixed_value)
        for i in range(len(tokens) - len(value_tokens), -1, -1):
            if tokens[i:i + len(value_tokens)] == value_tokens:
                return fixed_value, 1

    return value, 0


def generate_ontology_tensor_file(config_version, dataset_version, usage_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = BertConfig.from_pretrained('model/vn_{}-config.json'.format(config_version))
    tokenizer = BertTokenizer.from_pretrained('model/bert-base-uncased-vocab.txt', do_lower_case=False)
    model = ValueNormalize.from_pretrained(
        'output/vn_{}_{}_{}/pytorch_model.bin'.format(config_version, dataset_version, usage_rate),
        config=config)
    model.to(device)
    model.eval()
    ontology_input_ids_dict = {}
    ontology_attention_masks_dict = {}
    ontology = json.load(open("data/{}/ontology_{}.json".format(dataset_version,usage_rate), 'r'))
    ontology_tensors_dict = {}

    for key in ontology:
        ontology_input_ids_dict[key] = [
            tokenizer.convert_tokens_to_ids(['[CLS]'] + tokenizer.tokenize(raw_answer) + ['[SEP]']) for raw_answer in
            ontology[key]]
        ontology_attention_masks_dict[key] = [[1] * len(raw_answer_ids) for raw_answer_ids in
                                              ontology_input_ids_dict[key]]
        max_len = max([len(raw_answer_ids) for raw_answer_ids in ontology_input_ids_dict[key]])
        for index, raw_answer_ids in enumerate(ontology_input_ids_dict[key]):
            while len(raw_answer_ids) < max_len:
                ontology_input_ids_dict[key][index].append(0)
                ontology_attention_masks_dict[key][index].append(0)
        with torch.no_grad():
            ontology_tensors_dict[key] = model.generate_ontology_tensor(
                torch.tensor(ontology_input_ids_dict[key], device=device),
                torch.tensor(ontology_attention_masks_dict[key], device=device))
    torch.save(ontology_tensors_dict, 'data/{}/ontology_embed_{}_{}'.format(dataset_version, config_version,usage_rate))


def get_answer_from_vn(tokens, ontology_embeds, ontology, model, tokenizer, device):
    with torch.no_grad():
        input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(['[CLS]'] + tokens + ['[SEP]']),
                                 device=device).unsqueeze(0)
        answer_index, gate = model.evaluate(input_ids, ontology_embeds)

    return ontology[answer_index], gate


def count_fix_label():
    fix_dict = {}
    fix_count = {"spelling_mistakes": 0, "annotation_errors": 0, "varied_expressions": 0}
    for dialogue_idx in sp_dict_3:
        for value, sp in sp_dict_3[dialogue_idx].items():
            if value not in fix_dict:
                fix_dict[value] = []
            if sp not in fix_dict[value]:
                fix_dict[value].append(sp)
                fix_count["spelling_mistakes"] += 1

    for dialogue_idx in sp_dict_2:
        for value, sp in sp_dict_2[dialogue_idx].items():
            if value not in fix_dict:
                fix_dict[value] = []
            if sp not in fix_dict[value]:
                fix_dict[value].append(sp)
                fix_count["varied_expressions"] += 1

    for dialogue_idx in sp_dict:
        for value, sp in sp_dict[dialogue_idx].items():
            if value not in fix_dict:
                fix_dict[value] = []
            if sp not in fix_dict[value]:
                fix_dict[value].append(sp)
                fix_count["annotation_errors"] += 1

    for value in sp_common_multi_dict:
        if value not in fix_dict:
            fix_dict[value] = []
        for sp in sp_common_multi_dict[value]:
            if sp not in fix_dict[value]:
                fix_dict[value].append(sp)
                fix_count["varied_expressions"] += 1

    for value, sp in sp_common_dict_1.items():
        if value not in fix_dict:
            fix_dict[value] = []
        if sp not in fix_dict[value]:
            fix_dict[value].append(sp)
            fix_count["varied_expressions"] += 1

    print(0)


if __name__ == '__main__':
    # create VN dataset and all ontology
    # tokenizer = BertTokenizer.from_pretrained('model/bert-base-uncased-vocab.txt', do_lower_case=False)
    # generate_all_ontology("2.1")
    # generate_VN_dataset(tokenizer, "2.1")
    # get_VN_features(tokenizer, "2.0", True, True)
    count_fix_label()
    # generate_ontology_tensor_file("1", "2.1",'1')
