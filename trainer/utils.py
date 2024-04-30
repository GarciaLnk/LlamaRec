import torch
import torch.nn.functional as F
from torch import optim as optim

from config import *


def ndcg(scores, labels, k):
    scores = scores.cpu()
    labels = labels.cpu()
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hits = labels.gather(1, cut)
    position = torch.arange(2, 2 + k)
    weights = 1 / torch.log2(position.float())
    dcg = (hits.float() * weights).sum(1)
    idcg = torch.Tensor([weights[: min(int(n), k)].sum() for n in labels.sum(1)])
    ndcg = dcg / idcg
    return ndcg.mean()


def absolute_metrics_batch_wrapper(
    scores, labels, ks, num_classes=None, preprocessed=False, batch_size=10000
):
    """
    Wrapper for metrics calculation, calculate metrics of smaller batches of labels and average them
    """
    metrics = {}
    total_batches = (labels.size(0) + batch_size - 1) // batch_size
    total_samples = labels.size(0)

    for i in range(total_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, scores.size(0))
        curr_batch_size = end_idx - start_idx

        scores_batch = scores[start_idx:end_idx]
        labels_batch = labels[start_idx:end_idx]

        metrics_batch = absolute_recall_mrr_ndcg_for_ks(
            scores_batch,
            labels_batch,
            ks,
            num_classes=num_classes,
            preprocessed=preprocessed,
        )

        for key, value in metrics_batch.items():
            if key not in metrics:
                metrics[key] = 0
            metrics[key] += value * curr_batch_size

    for key in metrics:
        metrics[key] /= total_samples

    return metrics


def absolute_recall_mrr_ndcg_for_ks(
    scores, labels, ks, num_classes=None, preprocessed=False
):
    metrics = {}
    if num_classes is None:
        num_classes = scores.size(1)
    labels = F.one_hot(labels, num_classes=num_classes)
    answer_count = labels.sum(1)

    labels_float = labels.float()

    if not preprocessed:
        rank = (-scores).argsort(dim=1)
    else:
        rank = scores

    cut = rank
    for k in sorted(ks, reverse=True):
        cut = cut[:, :k]
        hits = labels_float.gather(1, cut)
        metrics["Recall@%d" % k] = (
            (
                hits.sum(1)
                / torch.min(torch.Tensor([k]).to(labels.device), labels.sum(1).float())
            )
            .mean()
            .cpu()
            .item()
        )

        metrics["MRR@%d" % k] = (
            (hits / torch.arange(1, k + 1).unsqueeze(0).to(labels.device))
            .sum(1)
            .mean()
            .cpu()
            .item()
        )

        position = torch.arange(2, 2 + k)
        weights = 1 / torch.log2(position.float())
        dcg = (hits * weights.to(hits.device)).sum(1)
        idcg = torch.Tensor([weights[: min(int(n), k)].sum() for n in answer_count]).to(
            dcg.device
        )
        ndcg = (dcg / idcg).mean()
        metrics["NDCG@%d" % k] = ndcg.cpu().item()

    return metrics


class AverageMeterSet(object):
    def __init__(self, meters=None):
        self.meters = meters if meters else {}

    def __getitem__(self, key):
        if key not in self.meters:
            meter = AverageMeter()
            meter.update(0)
            return meter
        return self.meters[key]

    def update(self, name, value, n=1):
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, format_string="{}"):
        return {
            format_string.format(name): meter.val for name, meter in self.meters.items()
        }

    def averages(self, format_string="{}"):
        return {
            format_string.format(name): meter.avg for name, meter in self.meters.items()
        }

    def sums(self, format_string="{}"):
        return {
            format_string.format(name): meter.sum for name, meter in self.meters.items()
        }

    def counts(self, format_string="{}"):
        return {
            format_string.format(name): meter.count
            for name, meter in self.meters.items()
        }


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(
            self=self, format=format
        )
