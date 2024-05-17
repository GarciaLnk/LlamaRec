import pickle

import torch
import torch.nn as nn
from tqdm import tqdm

from config import args

from .base import BaseTrainer
from .utils import absolute_metrics_batch_wrapper, absolute_recall_mrr_ndcg_for_ks


class BERTTrainer(BaseTrainer):
    def __init__(
        self, args, model, train_loader, val_loader, test_loader, export_root, use_wandb
    ):
        super().__init__(
            args, model, train_loader, val_loader, test_loader, export_root, use_wandb
        )
        self.ce = nn.CrossEntropyLoss(ignore_index=0)

    def calculate_loss(self, batch):
        seqs, labels = batch
        logits = self.model(seqs)
        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)
        loss = self.ce(logits, labels)
        return loss

    def calculate_metrics(self, batch, exclude_history=True):
        seqs, labels = batch

        scores = self.model(seqs)[:, -1, :]
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
                seqs, all_labels = batch

                all_scores = self.model(seqs)[:, -1, :]
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
                seqs, all_labels = batch

                all_scores = self.model(seqs)[:, -1, :]
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

            test_retrieval = {
                "original_size": len(test_probs),
                "retrieval_size": len(test_candidates),
                "original_metrics": test_metrics,
                "retrieval_metrics": absolute_metrics_batch_wrapper(
                    torch.tensor(test_probs)[torch.tensor(test_users) - 1],
                    torch.tensor(test_labels)[torch.tensor(test_users) - 1],
                    args.metric_ks,
                    num_classes=args.num_items + 1,
                    preprocessed=True,
                ),
                "non_retrieval_metrics": absolute_metrics_batch_wrapper(
                    torch.tensor(test_probs)[torch.tensor(non_test_users) - 1],
                    torch.tensor(test_labels)[torch.tensor(non_test_users) - 1],
                    args.metric_ks,
                    num_classes=args.num_items + 1,
                    preprocessed=True,
                ),
            }

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
                    "test_retrieval": test_retrieval,
                },
                f,
            )
