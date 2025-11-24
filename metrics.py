import json
import re
import string
import argparse
from collections import Counter


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    IncompleteError, SuperfluityError, MultiHopError = 0, 0, 0

    ZERO = (0, 0, 0)

    # handle yes/no/noanswer first
    if normalized_prediction != normalized_ground_truth:
        if normalized_prediction in normalized_ground_truth:
            IncompleteError += 1
        elif normalized_ground_truth in normalized_prediction:
            SuperfluityError += 1
        else:
            MultiHopError += 1

        if normalized_prediction in ["yes", "no", "noanswer"]:
            return ZERO + (IncompleteError, SuperfluityError, MultiHopError)
        if normalized_ground_truth in ["yes", "no", "noanswer"]:
            return ZERO + (IncompleteError, SuperfluityError, MultiHopError)

    pred_tokens = normalized_prediction.split()
    gold_tokens = normalized_ground_truth.split()

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO + (IncompleteError, SuperfluityError, MultiHopError)

    prec = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gold_tokens)
    f1 = 2 * prec * recall / (prec + recall)

    return f1, prec, recall, IncompleteError, SuperfluityError, MultiHopError


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def update_answer(metrics, prediction, gold):
    em = exact_match_score(prediction, gold)
    f1, prec, recall, inc, sup, mul = f1_score(prediction, gold)

    metrics["em"] += float(em)
    metrics["f1"] += f1
    metrics["prec"] += prec
    metrics["recall"] += recall

    metrics["IncompletenessError"] += inc
    metrics["SuperfluityError"] += sup
    metrics["MultiHopRCError"] += mul

    return em, prec, recall


def update_sp(metrics, prediction, gold):
    pred = set(map(tuple, prediction))
    gold = set(map(tuple, gold))

    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)

    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    em = 1.0 if (fp + fn) == 0 else 0.0

    metrics["sp_em"] += em
    metrics["sp_f1"] += f1
    metrics["sp_prec"] += prec
    metrics["sp_recall"] += recall

    return em, prec, recall


def eval(prediction_file, gold_file):
    with open(prediction_file, "r", encoding="utf-8") as f:
        prediction = json.load(f)
    with open(gold_file, "r", encoding="utf-8") as f:
        gold = json.load(f)

    metrics = {
        "IncompletenessError": 0,
        "SuperfluityError": 0,
        "MultiHopRCError": 0,
        "em": 0,
        "f1": 0,
        "prec": 0,
        "recall": 0,
        "sp_em": 0,
        "sp_f1": 0,
        "sp_prec": 0,
        "sp_recall": 0,
        "joint_em": 0,
        "joint_f1": 0,
        "joint_prec": 0,
        "joint_recall": 0,
    }

    for dp in gold:
        _id = dp["_id"]
        can_eval_joint = True

        # -------- Answer --------
        if _id not in prediction["answer"]:
            print(f"Missing answer for {_id}")
            can_eval_joint = False
        else:
            em, prec, recall = update_answer(metrics, prediction["answer"][_id], dp["answer"])

        # -------- Supporting Facts --------
        if _id not in prediction["sp"]:
            print(f"Missing sp for {_id}")
            can_eval_joint = False
        else:
            sp_em, sp_prec, sp_recall = update_sp(metrics, prediction["sp"][_id], dp["supporting_facts"])

        # -------- Joint Metrics --------
        if can_eval_joint:
            joint_prec = prec * sp_prec
            joint_recall = recall * sp_recall

            joint_f1 = (
                2 * joint_prec * joint_recall / (joint_prec + joint_recall)
                if joint_prec + joint_recall > 0
                else 0.0
            )
            joint_em = em * sp_em

            metrics["joint_em"] += joint_em
            metrics["joint_f1"] += joint_f1
            metrics["joint_prec"] += joint_prec
            metrics["joint_recall"] += joint_recall

    N = len(gold)
    for k in metrics:
        metrics[k] /= N

    print("\n===== HotpotQA Evaluation Results =====")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    print("\nError counts based on N:")
    print(f"IncompletenessError Count: {metrics['IncompletenessError'] * N}")
    print(f"SuperfluityError Count: {metrics['SuperfluityError'] * N}")
    print(f"MultiHopRCError Count: {metrics['MultiHopRCError'] * N}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-file", type=str, default="pred.json")
    parser.add_argument("--gold-file", type=str, default="hotpot_dev.json")
    args = parser.parse_args()

    eval(args.prediction_file, args.gold_file)
