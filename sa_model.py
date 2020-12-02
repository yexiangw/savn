import torch
from torch import nn
from transformers import BertPreTrainedModel, BertModel
from torch.nn import CrossEntropyLoss
import math


class Att_Layer(nn.Module):
    def __init__(self, config):
        super(Att_Layer, self).__init__()
        self.hidden_size = config.hidden_size
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.q = nn.Linear(config.hidden_size, config.hidden_size)
        self.k = nn.Linear(config.hidden_size, config.hidden_size)
        self.v = nn.Linear(config.hidden_size, config.hidden_size)
        self.attention_dropout = nn.Dropout(config.attention_probs_dropout_prob)  # 0.1
        self.dense2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dense3 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = self.gelu
        self.dense4 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm4 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def forward(self, input_para, slot_hidden, attention_mask):
        q = self.q(slot_hidden)
        k = self.k(input_para)
        v = self.v(input_para)
        attention_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.hidden_size)
        attention_scores = attention_scores + (1.0 - attention_mask[:, None, :]) * -10000.0
        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.attention_dropout(attention_probs)
        att_output = torch.matmul(attention_probs, v)

        att_output = self.dense2(att_output)
        att_output = self.dropout(att_output)
        att_output = att_output + slot_hidden
        att_output = self.LayerNorm2(att_output)

        mlp_output = self.dense3(att_output)
        mlp_output = self.intermediate_act_fn(mlp_output)

        layer_output = self.dense4(mlp_output)
        layer_output = self.dropout(layer_output)
        layer_output = self.LayerNorm4(layer_output + att_output)
        return layer_output


class SlotAttention(BertPreTrainedModel):
    def __init__(self, config, args):
        super(SlotAttention, self).__init__(config)
        self.num_labels = config.num_labels
        self.cls_lambda = args.cls_lambda
        self.ans_lambda = args.ans_lambda
        self.bert = BertModel(config)
        self.start_layer = nn.Linear(config.hidden_size, config.hidden_size)
        self.end_layer = nn.Linear(config.hidden_size, config.hidden_size)

        self.type_attention = Att_Layer(config)
        self.cls_layer = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, value_types=None, slot_input_ids=None,
                start_positions=None, end_positions=None):
        sequence_output, pool_output = self.bert(input_ids,
                                                 attention_mask=attention_mask,
                                                 token_type_ids=token_type_ids,
                                                 return_dict=False)

        slot_hidden = self.bert.embeddings(slot_input_ids[0][0]).mean(-2)

        type_att_output = self.type_attention(sequence_output, slot_hidden, attention_mask)
        type_logits = self.cls_layer(type_att_output)

        start_hidden = self.start_layer(slot_hidden)
        end_hidden = self.end_layer(slot_hidden)

        start_logits = torch.matmul(start_hidden, sequence_output.permute(0, 2, 1))
        end_logits = torch.matmul(end_hidden, sequence_output.permute(0, 2, 1))

        outputs = (type_logits, start_logits, end_logits)
        if value_types is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            cls_loss = loss_fct(type_logits.view(-1, self.num_labels), value_types.view(-1))
            start_loss = loss_fct(start_logits.reshape(-1, start_logits.size(-1)), start_positions.view(-1))
            end_loss = loss_fct(end_logits.reshape(-1, end_logits.size(-1)), end_positions.view(-1))

            total_loss = self.ans_lambda * (start_loss + end_loss) + self.cls_lambda * cls_loss
            outputs = (total_loss, cls_loss, start_loss, end_loss) + outputs
        return outputs
