import argparse
import numpy as np
import mindspore as ms
from mindspore import nn, ops, Tensor
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig

# ===== 引入前面我们迁移好的模块 =====
from dataloader_ms import HotpotREPipeMS, HotpotQAPipeMS, HotpotLoaderMS
from metrics_ms import DocselectionMetricMS, SpanSentenceMetricMS
from model_ms.electra_retriever_ms import ElectraRetrieverMS
from model_ms.roberta_retriever_ms import RobertaRetrieverMS
from model_ms.deberta_reader_ms import DebertaReaderMS
from model_ms.roberta_reader_ms import RobertaReaderMS

from transformers import ElectraTokenizerFast, RobertaTokenizerFast, DebertaV2TokenizerFast


parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, choices=["RE", "QA"], default="RE")
parser.add_argument("--data-path", type=str, default="../HotpotQAData")
parser.add_argument("--re-model", default="Roberta", choices=["Roberta", "Electra"])
parser.add_argument("--qa-model", default="Roberta", choices=["Roberta", "Deberta"])
parser.add_argument("--lr", default=5e-6, type=float)
parser.add_argument("--warmupsteps", default=0.1, type=float)
parser.add_argument("--batch-size", default=1, type=int)
parser.add_argument("--accumulation-steps", default=16, type=int)
parser.add_argument("--epoch", default=8, type=int)
parser.add_argument("--seed", default=41, type=int)
parser.add_argument("--LDLA-decay-rate", default=0.01, type=float)
args = parser.parse_args()

ms.set_seed(args.seed)
np.random.seed(args.seed)

Sentence_token = "</e>"
DOC_token = "</d>"

# ===============================
#       Warmup scheduler
# ===============================
class WarmupDecayLR(nn.learning_rate_schedule.LearningRateSchedule):
    def __init__(self, base_lr, warmup_ratio, total_steps):
        super().__init__()
        self.base_lr = base_lr
        self.warmup_steps = int(total_steps * warmup_ratio)
        self.total_steps = total_steps

    def construct(self, step):
        step = ops.cast(step, ms.float32)
        if step < self.warmup_steps:
            return self.base_lr * (step / self.warmup_steps)
        return self.base_lr


# ===============================
#           辅助函数
# ===============================
def build_dataset_RE(tokenizer):
    loader = HotpotLoaderMS()
    raw_train = loader.load(f"{args.data_path}/hotpot_train_v1.json")
    raw_dev = loader.load(f"{args.data_path}/hotpot_dev_v1.json")

    pipe = HotpotREPipeMS(tokenizer)
    train = pipe.process(raw_train)
    dev = pipe.process(raw_dev)

    def generator(data):
        for x in data:
            yield (
                np.array(x["question_ids"], dtype=np.int32),
                np.array(x["document_ids"], dtype=object),
                np.array(x["question_length"]),
                np.array(x["doc_length"]),
                np.array(x["gold_doc_pair"]),
                np.array(x["gold_answer_doc"]),
                np.array(x["doc_num"])
            )

    train_dataset = ms.dataset.GeneratorDataset(generator(train),
                    ["question_ids","document_ids","question_length","doc_length",
                     "gold_doc_pair","gold_answer_doc","doc_num"])
    dev_dataset = ms.dataset.GeneratorDataset(generator(dev),
                    ["question_ids","document_ids","question_length","doc_length",
                     "gold_doc_pair","gold_answer_doc","doc_num"])
    train_dataset = train_dataset.batch(args.batch_size)
    dev_dataset = dev_dataset.batch(args.batch_size)
    return train_dataset, dev_dataset


def build_dataset_QA(tokenizer):
    loader = HotpotLoaderMS()
    raw_train = loader.load(f"{args.data_path}/hotpot_train_v1.json")
    raw_dev = loader.load(f"{args.data_path}/hotpot_dev_v1.json")

    pipe = HotpotQAPipeMS(tokenizer)
    train = pipe.process(raw_train)
    dev = pipe.process(raw_dev)

    def generator(data):
        for x in data:
            yield (
                np.array(x["input_ids"], dtype=np.int32),
                np.array(x["sentence_index"]),
                np.array(x["sentence_labels"]),
                np.array(x["sentence_num"]),
                np.array(x["start_positions"]),
                np.array(x["end_positions"]),
                np.array(x["answer_type"]),
                np.array(x["F1_smoothing_start_label"]),
                np.array(x["F1_smoothing_end_label"])
            )

    train_dataset = ms.dataset.GeneratorDataset(
        generator(train),
        ["input_ids","sentence_index","sentence_labels","sentence_num",
         "start_positions","end_positions","answer_type",
         "F1_start","F1_end"]
    )
    dev_dataset = ms.dataset.GeneratorDataset(
        generator(dev),
        ["input_ids","sentence_index","sentence_labels","sentence_num",
         "start_positions","end_positions","answer_type",
         "F1_start","F1_end"]
    )
    train_dataset = train_dataset.batch(args.batch_size)
    dev_dataset = dev_dataset.batch(args.batch_size)
    return train_dataset, dev_dataset


# ===============================
#           训练循环（通用）
# ===============================
def train_loop(model, optimizer, train_ds, dev_ds, metric, name):
    total_steps = args.epoch * train_ds.get_dataset_size()
    lr = WarmupDecayLR(args.lr, args.warmupsteps, total_steps)

    loss_fn = None  # loss 内部由模型 forward 返回 "loss"
    net_with_loss = model

    # 梯度累积
    grad_accum = args.accumulation_steps
    grad_fn = ops.value_and_grad(lambda *inputs: net_with_loss(*inputs)["loss"],
                                 None, optimizer.parameters)

    ckpt_cfg = CheckpointConfig(save_checkpoint_steps=2000, keep_checkpoint_max=3)
    ckpt_cb = ModelCheckpoint(prefix=name, directory=f"../checkpoints/{name}", config=ckpt_cfg)

    step = 0
    for epoch in range(args.epoch):
        print(f"=== Epoch {epoch+1}/{args.epoch} ===")

        # LDLA update
        if epoch == 0:
            model.smoothing_weight = 0.1
        else:
            model.smoothing_weight = max(0, model.smoothing_weight - args.LDLA_decay_rate)

        for batch in train_ds:
            lr_now = lr(step)
            optimizer.learning_rate = lr_now

            # forward + grad
            loss, grads = grad_fn(*batch)
            loss = loss / grad_accum
            grads = [g / grad_accum for g in grads]

            optimizer(grads)

            if (step+1) % 50 == 0:
                print("Step", step, "Loss:", loss.asnumpy().item())

            step += 1


        # ===== 验证 =====
        metric.reset()
        model.set_train(False)

        for batch in dev_ds:
            output = model(*batch)
            metric.update(output, batch)

        print("Eval:", metric.get_metric())
        model.set_train(True)

        ckpt_cb.step_end(run_context=None)


# ===============================
#               Main
# ===============================
def main():
    print("LDLA:", args.LDLA_decay_rate)

    if args.task == "RE":
        # ---------- Tokenizer ----------
        if args.re_model == "Electra":
            tokenizer = ElectraTokenizerFast.from_pretrained("google/electra-large-discriminator")
            model = ElectraRetrieverMS("google/electra-large-discriminator")
        else:
            tokenizer = RobertaTokenizerFast.from_pretrained("roberta-large")
            model = RobertaRetrieverMS("roberta-large")

        train_ds, dev_ds = build_dataset_RE(tokenizer)
        metric = DocselectionMetricMS()

        optimizer = nn.AdamWeightDecay(model.trainable_params(), learning_rate=args.lr)
        name = f"RE-{args.re_model}-seed{args.seed}"

        train_loop(model, optimizer, train_ds, dev_ds, metric, name)

    # ======================================
    #                 QA
    # ======================================
    if args.task == "QA":
        if args.qa_model == "Deberta":
            tokenizer = DebertaV2TokenizerFast.from_pretrained("microsoft/deberta-v2-xxlarge")
            tokenizer.add_tokens([Sentence_token, DOC_token])
            model = DebertaReaderMS("microsoft/deberta-v2-xxlarge", tokenizer)

        else:
            tokenizer = RobertaTokenizerFast.from_pretrained("roberta-large")
            tokenizer.add_tokens([Sentence_token, DOC_token])
            model = RobertaReaderMS("roberta-large", tokenizer)

        train_ds, dev_ds = build_dataset_QA(tokenizer)
        metric = SpanSentenceMetricMS(tokenizer)

        optimizer = nn.AdamWeightDecay(model.trainable_params(), learning_rate=args.lr)
        name = f"QA-{args.qa_model}-seed{args.seed}"

        train_loop(model, optimizer, train_ds, dev_ds, metric, name)


if __name__ == "__main__":
    main()
