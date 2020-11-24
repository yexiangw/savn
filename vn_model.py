import torch
from torch import nn
from transformers import BertPreTrainedModel, BertModel
from torch.nn import CrossEntropyLoss

class ValueNormalize(BertPreTrainedModel):
    def __init__(self, config):
        super(ValueNormalize, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, answer_index=None, ontology_input_ids=None, ontology_attention_masks=None,
                answer_type=None):
        batch_size = input_ids.size(0)
        ss_output = self.bert(input_ids, attention_mask=attention_mask, )[0][:, 0, :]
        on_outputs = [self.bert(ontology_input_ids[i], attention_mask=ontology_attention_masks[i])[0][:, 0, :] for i in
                      range(batch_size)]

        logits = [torch.matmul(ss_output[i], on_outputs[i].transpose(-1, -2)).unsqueeze(0) for i in
                  range(batch_size)]

        cos_logits = [torch.cosine_similarity(ss_output[i], on_outputs[i], dim=-1) for i in range(batch_size)]

        outputs = (logits,)
        if answer_index is not None:
            loss = 0
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            for i in range(input_ids.size(0)):
                if answer_type[i]:
                    loss += loss_fct(logits[i], answer_index[i].unsqueeze(0))
                else:
                    loss += torch.max(cos_logits[i]) + 1
            loss = loss / input_ids.size(0)
            outputs = (loss,) + outputs
        return outputs

    def evaluate(self, input_ids=None, ontology_embeds=None):
        ss_outputs = self.bert(input_ids)[0][:, 0, :]
        logits = torch.matmul(ss_outputs, ontology_embeds.transpose(-1, -2)).unsqueeze(0)
        answer_index = logits.argmax(-1)
        gate = torch.cosine_similarity(ss_outputs, ontology_embeds[answer_index], dim=-1)
        return answer_index, gate

    def generate_ontology_tensor(self, ontology_input_ids, ontology_attention_masks):
        ontology_output = self.bert(ontology_input_ids, attention_mask=ontology_attention_masks)[0][:, 0, :]
        return ontology_output
