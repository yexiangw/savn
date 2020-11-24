## Slot Attention with Value Normalization for Multi-Domain Dialogue State Tracking

This is the Pytorch implementation for the paper: **Slot Attention with Value Normalization for Multi-Domain Dialogue State Tracking**. Yexiang Wang, Yi Guo and Siqi Zhu. **EMNLP 2020**. [[PDF]](https://www.aclweb.org/anthology/2020.emnlp-main.243/)

The code was written by PyTorch >= 1.1.0. If you use any source codes or SP labels in your work, please cite the following paper. The bibtex is listed below:

```html
@inproceedings{wang-etal-2020-slot,
    title = "Slot Attention with Value Normalization for Multi-Domain Dialogue State Tracking",
    author = "Wang, Yexiang  and
      Guo, Yi  and
      Zhu, Siqi",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.243",
    pages = "3019--3028",
}
```

### Abstract

Incompleteness of domain ontology and unavailability of some values are two inevitable problems of dialogue state tracking (DST). Existing approaches generally fall into two extremes: choosing models without ontology or embedding ontology in models leading to over-dependence. In this paper, we propose a new architecture to cleverly exploit ontology, which consists of Slot Attention (SA) and Value Normalization (VN), referred to as SAVN. Moreover, we supplement the annotation of supporting span for MultiWOZ 2.1, which is the shortest span in utterances to support the labeled value. SA shares knowledge between slots and utterances and only needs a simple structure to predict the supporting span. VN is designed specifically for the use of ontology, which can convert supporting spans to the values. Empirical results demonstrate that SAVN achieves the state-of-the-art joint accuracy of 54.52% on MultiWOZ 2.0 and 54.86% on MultiWOZ 2.1. Besides, we evaluate VN with incomplete ontology. The results show that even if only 30% ontology is used, VN can also contribute to our model.

### Model

![The overview of SAVN](https://github.com/wyxlzsq/savn/blob/master/.metas/SAVN.png)

![The model architecture of Slot Attention.](https://github.com/wyxlzsq/savn/blob/master/.metas/SA.png)

![The model architecture of Value Normalization.](https://github.com/wyxlzsq/savn/blob/master/.metas/VN.png)

### Run

You can follow these steps to get the results on the development set.

```console
❱❱❱ python3 create_data_2.1.py
❱❱❱ python3 vn_run.py --use_default_args
❱❱❱ python3 sa_run.py --use_default_args
```

### Incomplete Ontology Evaluation Details

![Incomplete Ontology Evaluation Details](https://github.com/wyxlzsq/savn/blob/master/.metas/ioed.png)

