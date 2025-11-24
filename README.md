# Rethinking Label Smoothing on Multi-hop Question Answering (Smoothing-R3-with-MindSpore)

![License](https://img.shields.io/badge/License-Apache%202.0-green)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)

This repository contains the official implementation of the paper:

**“Rethinking Label Smoothing on Multi-hop Question Answering”**
(🏆 *CCL 2023 Best Paper*)

The work introduces **Smoothing-R³**, an advanced Multi-hop QA framework integrating multiple novel label smoothing strategies, including:

* **F1 Smoothing** — inspired by F1 metrics in MRC tasks
* **LDLA (Linear Decay Label Smoothing Algorithm)** — curriculum-learning-inspired smoothing schedule

The repository includes model implementations, training scripts, and datasets used in the paper.

---

## 📌 Quick Links

* [Requirements](#requirements-)
* [Data](#data-)
* [Reproducing Baselines](#reproducing-baselines-)
* [Training Retriever](#training-the-retriever-model)
* [Training Reader](#training-the-reader-model)
* [Evaluation](#evaluation-)
* [Citation](#citation-)

---

## Requirements 📚

Please install the following dependencies:

```
transformers >= 4.20.0
fastNLP == 1.0.1
jsonlines
ipdb
pandas
torch
ujson
```

---

## Data 💾

We use the **HotpotQA** dataset.

### Steps to prepare:

1. Visit the official dataset page: [https://hotpotqa.github.io/](https://hotpotqa.github.io/)
2. Create a folder named **HotpotQAData** in the project root (same level as `code` folder).
3. Place the downloaded dataset files inside `HotpotQAData`.

---

## Reproducing Baselines 🚀

Starter scripts are provided in [`main.py`](code/main.py).
Below are commands for training both Retriever and Reader models.

---

## Training the Retriever Model

```bash
python train.py \
    --task RE \
    --lr 5e-6 \
    --batch-size 16 \
    --accumulation-steps 1 \
    --epoch 8 \
    --seed 41 \
    --re-model Electra
```

---

## Training the Reader Model

```bash
python train.py \
    --task QA \
    --lr 2e-6 \
    --batch-size 8 \
    --accumulation-steps 2 \
    --epoch 8 \
    --seed 41 \
    --qa-model Deberta
```

---

## Script Parameters Explained

| Argument                | Description                                                                          |
| ----------------------- | ------------------------------------------------------------------------------------ |
| `task`                  | Which model to train: Retriever (`RE`) or Reader (`QA`)                              |
| `lr`                    | Learning rate                                                                        |
| `batch-size`            | Batch size per GPU step                                                              |
| `accumulation-steps`    | Gradient accumulation steps (Effective batch size = batch-size × accumulation-steps) |
| `epoch`                 | Total number of epochs                                                               |
| `seed`                  | Random seed                                                                          |
| `re-model` / `qa-model` | Electra / Roberta for RE, Deberta / Roberta for QA                                   |
| `LDLA-decay-rate`       | Decay rate of LDLA smoothing                                                         |
| `data-path`             | Dataset directory                                                                    |
| `warmupsteps`           | Warmup ratio for LR schedule                                                         |

---

## Hardware Recommendations & Notes

* Experiment settings are optimized for **NVIDIA A100 GPUs**.
* **RoBERTa** is supported as a backbone for both Retriever & Reader.
* Processed data is cached in `cache/*.pkl`.
  If you modify `preprocess.py`, remember to **clear the cache**.
* For the Reader model, metric `cl_acc` is included to evaluate answer type classification.

### Checkpoints

* Retriever: Top 3 checkpoints saved based on **F1 score**
* Reader: Top 3 checkpoints saved based on **Joint F1 score**

---

## Evaluation 💻

Use the provided script to evaluate predictions:

```bash
python code/hotpot_official_evaluate.py \
    --prediction-file model_pred.json \
    --gold-file HotpotQAData/hotpot_dev_distractor_v1.json
```

### Metrics Reported

| Metric                                               | Meaning                            |
| ---------------------------------------------------- | ---------------------------------- |
| `sp_em`, `sp_f1`, `sp_prec`, `sp_recall`             | Supporting fact prediction metrics |
| `em`, `f1`, `prec`, `recall`                         | Answer span extraction             |
| `joint_em`, `joint_f1`, `joint_prec`, `joint_recall` | Combined reasoning + span accuracy |

---

## Bug Reports or Questions 🤔

For any questions, please email:

📧 **[yinzhangyue@126.com](mailto:yinzhangyue@126.com)**

Or open an issue on GitHub — feedback is welcome!

---

## Citation 📖

If you find this work helpful, please cite:

```bibtex
@InProceedings{yin-etal-2023-rethinking,
  author    = "Yin, Zhangyue and Wang, Yuxin and Hu, Xiannian and Wu, Yiguang and Yan, Hang and Zhang, Xinyu and Cao, Zhao and Huang, Xuanjing and Qiu, Xipeng",
  title     = "Rethinking Label Smoothing on Multi-Hop Question Answering",
  booktitle = "Chinese Computational Linguistics",
  year      = "2023",
  publisher = "Springer Nature Singapore",
  address   = "Singapore",
  pages     = "72--87",
  isbn      = "978-981-99-6207-5"
}
```
