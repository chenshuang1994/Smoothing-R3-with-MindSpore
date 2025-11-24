import mindspore as ms
from mindspore import nn, ops, Tensor
from mindnlp.transformers import ElectraModel, ElectraConfig


##############  ElectraRetriever in MindSpore  ##############
class ElectraRetriever(nn.Cell):
    def __init__(self, config: ElectraConfig):
        super().__init__()

        self.num_labels = config.num_labels
        self.smoothing_weight = 0
        self.epoch = 0

        self.electra = ElectraModel(config)

        hidden = config.hidden_size
        self.single_document_classifier_layer = nn.Dense(hidden, 2)
        self.double_document_classifier_layer = nn.Dense(hidden, 1)

        # operators
        self.softmax = ops.Softmax(-1)
        self.log = ops.Log()
        self.concat = ops.Concat(axis=-1)

        self.ce_loss = nn.CrossEntropyLoss()

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
        """
        question_ids: (B, Q)
        document_ids: (B, N, D)
        question_length: (B,)
        doc_length: (B, N)
        gold_doc_pair: (B,2)
        gold_answer_doc: (B,)
        doc_num: (B,)
        """
        batch = question_ids.shape[0]
        device = question_ids.device_target

        total_rank_loss = 0.0
        total_pair_loss = 0.0

        selected_document_pair = ops.Zeros()((batch, 2), ms.int32)

        for b in range(batch):

            # ------------------ If only 2 documents, skip stage-1 ranking ------------------
            if doc_num[b] < 3:
                selected_document_pair[b] = Tensor([0, 1], ms.int32)
                continue

            # ------------------ Stage One: Single Document Ranking ------------------
            # Construct (Q+D_i) sequence
            q_len = question_length[b]
            N = doc_num[b]

            # lengths of concat (Q + D_i)
            q_doc_len = q_len + doc_length[b][:N]
            max_len = min(int(q_doc_len.max().asnumpy()), 512)

            q_doc_ids = ops.Zeros()((N, max_len), ms.int32)
            q_doc_mask = ops.Zeros()((N, max_len), ms.int32)

            for i in range(N):
                seq = self.concat([
                    question_ids[b][:q_len],
                    document_ids[b][i][:doc_length[b][i]]
                ])[:512]

                seq_len = seq.shape[0]
                q_doc_ids[i, :seq_len] = seq
                q_doc_mask[i, :seq_len] = Tensor(1, ms.int32)

            stage1_outputs = self.electra(q_doc_ids, attention_mask=q_doc_mask, return_dict=False)
            stage1_hidden = stage1_outputs[0]        # [N, L, H]
            cls_hidden = stage1_hidden[:, 0, :]      # [N, H]

            doc_logits = self.single_document_classifier_layer(cls_hidden)  # [N,2]
            doc_prob = self.softmax(doc_logits)

            # select top 3 documents based on prob[:,1]
            select_scores = doc_prob[:, 1]
            top3 = ops.TopK(sorted=True)(select_scores, 3)[1]

            # ------------------ Rank Loss ------------------
            if gold_doc_pair is not None:
                N_valid = doc_num[b]

                labels = ops.Zeros()((N_valid,), ms.float32)
                for idx in gold_doc_pair[b]:
                    labels[int(idx)] = 1

                smoothing = ops.Ones()((N_valid,), ms.float32) / N_valid
                smooth_labels = (1 - self.smoothing_weight) * labels + self.smoothing_weight * smoothing

                pos_loss = -self.log(doc_prob[:, 1]) * smooth_labels
                neg_loss = -self.log(doc_prob[:, 0]) * (1 - labels)

                rank_loss = (pos_loss + neg_loss).sum() / N_valid
                total_rank_loss += rank_loss

            # ------------------ Stage Two: Double Document Selection ------------------
            triple_pairs = []
            triple_ids = []
            triple_masks = []

            pair_list = []

            pair_index = 0
            triple_pairs = []
            triple_logits_input = []

            # 3 docs → 3 choose 2 = 3 pairs
            pairs = [
                (top3[0], top3[1]),
                (top3[0], top3[2]),
                (top3[1], top3[2]),
            ]

            seq_lens = []
            for (i, j) in pairs:
                len_ij = q_len + doc_length[b][i] + doc_length[b][j]
                seq_lens.append(len_ij)
            max_len2 = min(int(max(seq_lens)), 512)

            pair_input_ids = ops.Zeros()((3, max_len2), ms.int32)
            pair_att_mask = ops.Zeros()((3, max_len2), ms.int32)
            pair_doc_ids = ops.Zeros()((3, 2), ms.int32)

            for k, (i, j) in enumerate(pairs):
                seq = self.concat([
                    question_ids[b][:q_len],
                    document_ids[b][i][:doc_length[b][i]],
                    document_ids[b][j][:doc_length[b][j]]
                ])[:512]

                L = seq.shape[0]
                pair_input_ids[k, :L] = seq
                pair_att_mask[k, :L] = Tensor(1, ms.int32)
                pair_doc_ids[k] = Tensor([int(i), int(j)], ms.int32)

            stage2_out = self.electra(pair_input_ids, attention_mask=pair_att_mask, return_dict=False)
            cls2 = stage2_out[0][:, 0, :]       # [3,H]

            pair_logits = self.double_document_classifier_layer(cls2).squeeze(-1)
            pair_prob = self.softmax(pair_logits)

            # select best pair
            best = int(ops.Argmax()(pair_prob))

            selected_document_pair[b] = pair_doc_ids[best]

            # ------------------ Document Pair Loss ------------------
            if gold_doc_pair is not None:
                labels = ops.Zeros()((3,), ms.float32)
                gold = set([int(gold_doc_pair[b][0]), int(gold_doc_pair[b][1])])

                for idx, (i, j) in enumerate(pairs):
                    if set([int(i), int(j)]) == gold:
                        labels[idx] = 1

                pair_loss = -(self.log(pair_prob) * labels).sum()
                total_pair_loss += pair_loss

        total_loss = (total_rank_loss + total_pair_loss) / batch

        return {
            "loss": total_loss,
            "selected_pair": selected_document_pair
        }
