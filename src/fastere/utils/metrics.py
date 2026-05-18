def f1_score(outputs, pred_name, gold_name, slice=None):
    pred = 0
    gold = 0
    correct = 0

    for val_out in outputs:
        if slice is not None:
            pred_triples = [t[:slice] for t in val_out[pred_name]]
            gold_triples = [t[:slice] for t in val_out[gold_name]]
        else:
            pred_triples = val_out[pred_name]
            gold_triples = val_out[gold_name]

        pred += len(pred_triples)
        gold += len(gold_triples)
        for _pred in pred_triples:
            if _pred in gold_triples:
                correct += 1

    precision = correct / (pred + 1e-8)
    recall = correct / (gold + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return f1, precision, recall


def f1_score_simple(labels, pred, ignore_index=0):
    gold_count = sum(labels != 0)
    pred_count = sum(pred != 0)

    zero = sum((labels | pred) == 0)
    correct = sum(labels == pred) - zero

    precision = correct / (pred_count + 1e-8)
    recall = correct / (gold_count + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return f1, precision, recall
