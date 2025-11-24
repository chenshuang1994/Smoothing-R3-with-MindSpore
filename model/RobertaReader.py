import mindspore as ms
from mindspore import nn, ops, Tensor
from mindnlp.transformers import RobertaModel, RobertaConfig


##############  RobertaReader in MindSpore  ##############
class RobertaReader(nn.Cell):
    def __init__(self, config: RobertaConfig):
        super().__init__()

        self.num_labels = config.num_labels
        self.smoothing_weight = 0.1
        self.epoch = 0

        self.roberta = RobertaModel(config)

        hidden = config.hidden_size
        self.sentence_outputs = nn.Dense(hidden, 2)
        self.answer_typeout = nn.Dense(hidden, 3)
        self.qa_outputs = nn.Dense(hidden, config.num_labels)

        # ops
        self.softmax = ops.Softmax(-1)
        self.log = ops.Log()

        # loss
        self.ce_loss = nn.CrossEntropyLoss()
        self.ce_loss_smooth = nn.CrossEntropyLoss(label_smoothing=self.smoothing_weight)

    def construct(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        sentence_index=None,
        sentence_labels=None,
        answer_type=None,
        sentence_num=None,
        F1_smoothing_start_label=None,
        F1_smoothing_end_label=None,
    ):
        """
        MindSpore version of RobertaReader forward
        """
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            return_dict=False
        )

        sequence_output = outputs[0]     # [B, L, H]
        B, L, E = sequence_output.shape
        S = sentence_index.shape[1]

        # ============ Supporting Facts ============
        sentence_output = []
        for b in range(B):
            gathered = ops.Gather()(sequence_output[b], sentence_index[b], 0)
            sentence_output.append(gathered.expand_dims(0))

        sentence_output = ops.Concat(axis=0)(sentence_output)   # [B, S, H]

        sentence_select = self.sentence_outputs(sentence_output)  # [B, S, 2]

        # Loss: Supporting Facts
        Lsentence = None
        if sentence_labels is not None:
            # mask padded sentences
            mask_val = Tensor([5.0, -5.0], ms.float32)
            for b in range(B):
                idx = sentence_num[b]
                sentence_select[b, idx:, :] = mask_val

            # reshape to [B,2,S]
            sentence_select_p = ops.Transpose()(sentence_select, (0, 2, 1))

            # ensure aligned
            slabels = sentence_labels[:, :sentence_select_p.shape[-1]]

            Lsentence = self.ce_loss_smooth(sentence_select_p, slabels)

        # predicted supporting sentences
        sentence_pred = ops.Argmax(-1)(sentence_select)

        # ============ Answer Type ============
        cls_vec = sequence_output[:, 0, :]
        output_answer_type = self.answer_typeout(cls_vec)

        Ltype = None
        if answer_type is not None:
            Ltype = self.ce_loss(output_answer_type, answer_type)
            ans_mask = (answer_type > 1).astype(ms.float32)

        # ============ Answer Span ============
        logits = self.qa_outputs(sequence_output)    # [B, L, 2]
        start_logits, end_logits = ops.Split(-1, 2)(logits)

        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        Lspan = None
        if start_positions is not None and end_positions is not None:

            start_prob = self.softmax(start_logits)
            end_prob = self.softmax(end_logits)

            start_loss = -self.log(start_prob) * F1_smoothing_start_label
            end_loss = -self.log(end_prob) * F1_smoothing_end_label

            start_loss = start_loss.sum(-1)
            end_loss = end_loss.sum(-1)

            start_loss = (start_loss * ans_mask).sum() / ans_mask.sum()
            end_loss = (end_loss * ans_mask).sum() / ans_mask.sum()

            Lspan = start_loss + end_loss

        # ============ Total Loss ============
        total_loss = None
        if Lsentence is not None and Ltype is not None and Lspan is not None:
            total_loss = Lsentence + Ltype + Lspan

        return {
            "loss": total_loss,
            "type_logits": output_answer_type,
            "start_logits": start_logits,
            "end_logits": end_logits,
            "sentence_predictions": sentence_pred
        }
