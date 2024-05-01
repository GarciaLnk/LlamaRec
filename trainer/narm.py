import pickle
from abc import *

import torch
import torch.nn as nn

from .base import *
from .loggers import *
from .utils import *


class RNNTrainer(BaseTrainer):
    def __init__(
        self, args, model, train_loader, val_loader, test_loader, export_root, use_wandb
    ):
        super().__init__(
            args, model, train_loader, val_loader, test_loader, export_root, use_wandb
        )
        self.ce = nn.CrossEntropyLoss(ignore_index=0)

    def calculate_loss(self, batch):
        seqs, lengths, labels = batch
        lengths = lengths.flatten().cpu()
        seqs, labels = seqs.to(self.device), labels.to(self.device)

        logits = self.model(seqs, lengths)
        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)
        loss = self.ce(logits, labels)
        return loss

    def calculate_metrics(self, batch, exclude_history=True):
        seqs, lengths, labels = batch
        lengths = lengths.flatten().cpu()
        seqs, labels = seqs.to(self.device), labels.to(self.device)

        scores = self.model(seqs, lengths)  # B x V
        B, L = seqs.shape
        if exclude_history:
            for i in range(L):
                scores[torch.arange(scores.size(0)), seqs[:, i]] = -1e9
            scores[:, 0] = -1e9  # padding
        metrics = absolute_recall_mrr_ndcg_for_ks(
            scores, labels.view(-1), self.metric_ks
        )
        return metrics

    def generate_candidates(self, retrieved_data_path):
        self.model.eval()
        val_users, val_candidates = [], []
        test_probs, test_labels, test_users, test_candidates = [], [], [], []
        non_test_users = []
        val_metrics, test_metrics = {}, {}
        for k in sorted(self.metric_ks, reverse=True):
            val_metrics[f"Recall@{k}"] = 0
            val_metrics[f"MRR@{k}"] = 0
            val_metrics[f"NDCG@{k}"] = 0
            test_metrics[f"Recall@{k}"] = 0
            test_metrics[f"MRR@{k}"] = 0
            test_metrics[f"NDCG@{k}"] = 0
        with torch.no_grad():
            print(
                "*************** Generating Candidates for Validation Set ***************"
            )
            tqdm_dataloader = tqdm(self.val_loader)
            total_items_processed = 0
            for _, batch in enumerate(tqdm_dataloader):
                batch = self.to_device(batch)
                seqs, lengths, all_labels = batch
                lengths = lengths.flatten().cpu()
                seqs, all_labels = seqs.to(self.device), all_labels.to(self.device)

                all_scores = self.model(seqs, lengths)
                B, L = seqs.shape
                for j in range(B):
                    scores = all_scores[j, :].unsqueeze(0)
                    labels = all_labels[j, :].unsqueeze(0)
                    for i in range(L):
                        scores[torch.arange(scores.size(0)), seqs[j, i]] = -1e9
                    scores[:, 0] = -1e9  # padding
                    metrics_batch = absolute_recall_mrr_ndcg_for_ks(
                        scores, labels.view(-1), self.metric_ks
                    )
                    for k in sorted(self.metric_ks, reverse=True):
                        val_metrics[f"Recall@{k}"] += metrics_batch[f"Recall@{k}"]
                        val_metrics[f"MRR@{k}"] += metrics_batch[f"MRR@{k}"]
                        val_metrics[f"NDCG@{k}"] += metrics_batch[f"NDCG@{k}"]
                    _, top_indices = torch.topk(
                        scores, self.args.llm_negative_sample_size + 1
                    )
                    user_id = total_items_processed + j + 1
                    if labels[0].item() in top_indices[0].tolist():
                        val_users.append(user_id)
                        val_candidates.append(top_indices[0].tolist())
                total_items_processed += B
            for k in sorted(self.metric_ks, reverse=True):
                val_metrics[f"Recall@{k}"] /= self.args.num_users
                val_metrics[f"MRR@{k}"] /= self.args.num_users
                val_metrics[f"NDCG@{k}"] /= self.args.num_users
            print(val_metrics)

            print(
                "****************** Generating Candidates for Test Set ******************"
            )
            tqdm_dataloader = tqdm(self.test_loader)
            total_items_processed = 0
            for _, batch in enumerate(tqdm_dataloader):
                batch = self.to_device(batch)
                seqs, lengths, all_labels = batch
                lengths = lengths.flatten().cpu()
                seqs, all_labels = seqs.to(self.device), all_labels.to(self.device)

                all_scores = self.model(seqs, lengths)
                B, L = seqs.shape
                for j in range(B):
                    scores = all_scores[j, :].unsqueeze(0)
                    labels = all_labels[j, :].unsqueeze(0)
                    for i in range(L):
                        scores[torch.arange(scores.size(0)), seqs[j, i]] = -1e9
                    scores[:, 0] = -1e9  # padding
                    test_probs.extend(
                        (-scores).argsort(dim=1)[:, : max(self.metric_ks)].tolist()
                    )
                    test_labels.extend(labels.view(-1).tolist())
                    metrics_batch = absolute_recall_mrr_ndcg_for_ks(
                        scores, labels.view(-1), self.metric_ks
                    )
                    for k in sorted(self.metric_ks, reverse=True):
                        test_metrics[f"Recall@{k}"] += metrics_batch[f"Recall@{k}"]
                        test_metrics[f"MRR@{k}"] += metrics_batch[f"MRR@{k}"]
                        test_metrics[f"NDCG@{k}"] += metrics_batch[f"NDCG@{k}"]
                    _, top_indices = torch.topk(
                        scores, self.args.llm_negative_sample_size + 1
                    )
                    user_id = total_items_processed + j + 1
                    if labels[0].item() in top_indices[0].tolist():
                        test_users.append(user_id)
                        test_candidates.append(top_indices[0].tolist())
                    else:
                        non_test_users.append(user_id)
                total_items_processed += B
            for k in sorted(self.metric_ks, reverse=True):
                test_metrics[f"Recall@{k}"] /= self.args.num_users
                test_metrics[f"MRR@{k}"] /= self.args.num_users
                test_metrics[f"NDCG@{k}"] /= self.args.num_users
            print(test_metrics)

        with open(retrieved_data_path, "wb") as f:
            pickle.dump(
                {
                    "val_metrics": val_metrics,
                    "val_users": val_users,
                    "val_candidates": val_candidates,
                    "test_probs": test_probs,
                    "test_labels": test_labels,
                    "test_metrics": test_metrics,
                    "test_users": test_users,
                    "test_candidates": test_candidates,
                    "non_test_users": non_test_users,
                },
                f,
            )
