import mindspore as ms
from mindspore import nn, ops, Tensor
from mindnlp.transformers import RobertaModel, RobertaConfig


##############  RobertaRetriever in MindSpore  ##############
class RobertaRetriever(nn.Cell):
    def __init__(self, config: RobertaConfig):
        super().__init__()

        self.num_labels = config.num_labels
        self.smoothing_weight = 0
        self.epoch = 0

        self.roberta = RobertaModel(config)

        hidden = config.hidden_size
        self.single_document_classifier_layer = nn.Dense(hidden, 2)
        self.double_document_classifier_layer = nn.Dense(hidden, 1)

        # ops
        self.softmax = ops.Softmax(-1)
        self.log = ops.Log()
        self.concat = ops.Concat(axis=-1)

    def construct(
        self,
        question_ids=None,
        document_ids=None,
        question_length=None,
        doc_length=None,
        gold_doc_pair=None,
        gold_answer_doc=None,
        doc_num=None,
    ):
        batch = question_ids.shape[0]

        total_rank_loss = 0.0
        total_pair_loss = 0.0

        selected_pair = ops.Zeros()((batch, 2), ms.int32)

        for b in range(batch):

            # If only 2 documents → skip stage 1
            if doc_num[b] < 3:
                selected_pair[b] = Tensor([0, 1], ms.int32)
                continue

            # -------------------- Stage 1: Single Document Ranking --------------------
            q_len = int(question_length[b])
            N = int(doc_num[b])

            q_doc_len = q_len + doc_length[b][:N]
            max_len = min(int(q_doc_len.max()), 512)

            q_doc_ids = ops.Zeros()((N, max_len), ms.int32)
            q_doc_mask = ops.Zeros()((N, max_len), ms.int32)

            for i in range(N):
                seq = self.concat([
                    question_ids[b][:q_len],
                    document_ids[b][i][:doc_length[b][i]]
                ])[:512]

                L = seq.shape[0]
                q_doc_ids[i, :L] = seq
                q_doc_mask[i, :L] = Tensor(1, ms.int32)

            # Roberta encode
            stage1_out = self.roberta(q_doc_ids, attention_mask=q_doc_mask, return_dict=False)
            cls_vec = stage1_out[0][:, 0, :]   # [N, H]

            doc_logits = self.single_document_classifier_layer(cls_vec)
            doc_prob = self.softmax(doc_logits)

            # top-3 according to prob[:,1]
            scores = doc_prob[:, 1]
            _, top3_idx = ops.TopK(sorted=True)(scores, 3)

            # -------------------- Rank Loss --------------------
            if gold_doc_pair is not None:
                labels = ops.Zeros()((N,), ms.float32)

                for idx in gold_doc_pair[b]:
                    labels[int(idx)] = 1

                smoothing = ops.Ones()((N,), ms.float32) / N
                smooth_labels = (1 - self.smoothing_weight) * labels + self.smoothing_weight * smoothing

                pos_loss = -self.log(doc_prob[:, 1]) * smooth_labels
                neg_loss = -self.log(doc_prob[:, 0]) * (1 - labels)

                rank_loss = (pos_loss + neg_loss).sum() / N
                total_rank_loss += rank_loss

            # -------------------- Stage 2: Double Document Selection --------------------
            pairs = [
                (top3_idx[0], top3_idx[1]),
                (top3_idx[0], top3_idx[2]),
                (top3_idx[1], top3_idx[2]),
            ]

            seq_lens = [
                q_len + doc_length[b][i] + doc_length[b][j]
                for (i, j) in pairs
            ]

            max_len2 = min(int(max(seq_lens)), 512)

            pair_ids = ops.Zeros()((3, max_len2), ms.int32)
            pair_mask = ops.Zeros()((3, max_len2), ms.int32)
            pair_doc_ids = ops.Zeros()((3, 2), ms.int32)

            for k, (i, j) in enumerate(pairs):
                seq = self.concat([
                    question_ids[b][:q_len],
                    document_ids[b][i][:doc_length[b][i]],
                    document_ids[b][j][:doc_length[b][j]]
                ])[:512]

                L = seq.shape[0]
                pair_ids[k, :L] = seq
                pair_mask[k, :L] = Tensor(1, ms.int32)
                pair_doc_ids[k] = Tensor([int(i), int(j)], ms.int32)

            stage2_out = self.roberta(pair_ids, attention_mask=pair_mask, return_dict=False)
            cls2 = stage2_out[0][:, 0, :]

            pair_logits = self.double_document_classifier_layer(cls2).squeeze(-1)
            pair_prob = self.softmax(pair_logits)

            best = int(ops.Argmax()(pair_prob))
            selected_pair[b] = pair_doc_ids[best]

            # -------------------- Double Document Pair Loss --------------------
            if gold_doc_pair is not None:
                goldset = set([int(gold_doc_pair[b][0]), int(gold_doc_pair[b][1])])

                labels = ops.Zeros()((3,), ms.float32)
                for idx, (i, j) in enumerate(pairs):
                    if set([int(i), int(j)]) == goldset:
                        labels[idx] = 1

                pair_loss = -(self.log(pair_prob) * labels).sum()
                total_pair_loss += pair_loss

        total_loss = (total_rank_loss + total_pair_loss) / batch

        return {
            "loss": total_loss,
            "selected_pair": selected_pair
        }
