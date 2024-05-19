import json
import os
from typing import Optional

import torch
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments
from transformers.trainer import DataLoader, Dataset

from config import args

from .utils import absolute_recall_mrr_ndcg_for_ks
from .verb import ManualVerbalizer


def llama_collate_fn_w_truncation(llm_max_length, eval=False, tokenizer=None):
    def llama_collate_fn(batch):
        all_input_ids = []
        all_attention_mask = []
        all_labels = []
        example_max_length = max(
            [len(batch[idx]["input_ids"]) for idx in range(len(batch))]
        )
        max_length = min(llm_max_length, example_max_length)

        for i in range(len(batch)):
            input_ids = batch[i]["input_ids"]
            attention_mask = batch[i]["attention_mask"]
            labels = batch[i]["labels"]
            if len(input_ids) > max_length:
                input_ids = input_ids[-max_length:]
                attention_mask = attention_mask[-max_length:]
                if not eval:
                    labels = labels[-max_length:]
            elif len(input_ids) < max_length:
                padding_length = max_length - len(input_ids)
                input_ids = [0] * padding_length + input_ids
                attention_mask = [0] * padding_length + attention_mask
                if not eval:
                    labels = [-100] * padding_length + labels

            if tokenizer is None:
                eos_token_id = 2
            else:
                eos_token_id = tokenizer.eos_token_id

            if not eval:
                assert input_ids[-1] == eos_token_id
                assert labels[-3] == -100 and labels[-2] != -100

            all_input_ids.append(torch.tensor(input_ids).long())
            all_attention_mask.append(torch.tensor(attention_mask).long())
            all_labels.append(torch.tensor(labels).long())

        return {
            "input_ids": torch.vstack(all_input_ids),
            "attention_mask": torch.vstack(all_attention_mask),
            "labels": torch.vstack(all_labels),
        }

    return llama_collate_fn


def compute_metrics_for_ks(ks, verbalizer):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        logits = torch.tensor(logits)
        labels = torch.tensor(labels).view(-1)
        scores = verbalizer.process_logits(logits)
        metrics = absolute_recall_mrr_ndcg_for_ks(scores, labels, ks)
        return metrics

    return compute_metrics


class LLMTrainer(Trainer):
    def __init__(
        self,
        args,
        model,
        train_loader,
        val_loader,
        test_loader,
        tokenizer,
        export_root,
        use_wandb,
        **kwargs
    ):
        self.original_args = args
        self.export_root = export_root
        self.use_wandb = use_wandb
        self.llm_max_text_len = args.llm_max_text_len
        self.rerank_metric_ks = args.rerank_metric_ks
        self.verbalizer = ManualVerbalizer(
            tokenizer=tokenizer,
            prefix="",
            post_log_softmax=False,
            classes=list(range(args.llm_negative_sample_size + 1)),
            label_words={
                i: chr(ord("A") + i) for i in range(args.llm_negative_sample_size + 1)
            },
        )

        hf_args = TrainingArguments(
            per_device_train_batch_size=args.lora_micro_batch_size,
            gradient_accumulation_steps=args.train_batch_size
            // args.lora_micro_batch_size,
            warmup_steps=args.warmup_steps,
            num_train_epochs=args.lora_num_epochs,
            learning_rate=args.lora_lr,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            tf32=torch.cuda.is_bf16_supported(),
            bf16_full_eval=torch.cuda.is_bf16_supported(),
            fp16_full_eval=not torch.cuda.is_bf16_supported(),
            logging_steps=10,
            optim="adamw_bnb_8bit",
            eval_strategy="steps",
            save_strategy="steps",
            eval_steps=args.lora_val_iterations,
            save_steps=args.lora_val_iterations,
            eval_delay=args.lora_val_delay,
            eval_accumulation_steps=args.lora_val_accumulation_steps,
            max_steps=args.lora_max_steps,
            output_dir=export_root,
            save_total_limit=3,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=False,
            group_by_length=False,
            report_to="wandb" if use_wandb else None,
            run_name=args.model_code + "_" + args.dataset_code if use_wandb else None,
            metric_for_best_model=args.rerank_best_metric,
            greater_is_better=True,
            torch_compile=not args.llm_enable_unsloth,
            dataloader_pin_memory=True,
            dataloader_num_workers=args.num_workers,
        )
        super().__init__(
            model=model,
            args=hf_args,
            callbacks=[EarlyStoppingCallback(args.lora_early_stopping_patience)],
            **kwargs
        )  # hf_args is now args

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.tokenizer = tokenizer

        self.train_loader.collate_fn = llama_collate_fn_w_truncation(
            self.llm_max_text_len, eval=False, tokenizer=tokenizer
        )
        self.val_loader.collate_fn = llama_collate_fn_w_truncation(
            self.llm_max_text_len, eval=True, tokenizer=tokenizer
        )
        self.test_loader.collate_fn = llama_collate_fn_w_truncation(
            self.llm_max_text_len, eval=True, tokenizer=tokenizer
        )
        self.compute_metrics = compute_metrics_for_ks(
            self.rerank_metric_ks, self.verbalizer
        )

        if len(self.label_names) == 0:
            self.label_names = ["labels"]  # for some reason label name is not set

    def test(self, test_retrieval):
        average_metrics = self.predict(test_dataset=None).metrics
        print("Ranking Performance on Subset:", average_metrics)
        print("************************************************************")
        with open(os.path.join(self.export_root, "subset_metrics.json"), "w") as f:
            json.dump(average_metrics, f, indent=4)

        print("Original Performance:", test_retrieval["original_metrics"])
        print("************************************************************")
        original_size = test_retrieval["original_size"]
        retrieval_size = test_retrieval["retrieval_size"]

        overall_metrics = {}
        for key in test_retrieval["non_retrieval_metrics"].keys():
            if "test_" + key in average_metrics:
                overall_metrics["test_" + key] = (
                    average_metrics["test_" + key] * retrieval_size
                    + test_retrieval["non_retrieval_metrics"][key]
                    * (original_size - retrieval_size)
                ) / original_size
        print("Overall Performance of Our Framework:", overall_metrics)
        with open(os.path.join(self.export_root, "overall_metrics.json"), "w") as f:
            json.dump(overall_metrics, f, indent=4)

        return average_metrics

    def get_train_dataloader(self):
        return self.train_loader

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        return self.val_loader

    def get_test_dataloader(self, test_dataset: Optional[Dataset] = None) -> DataLoader:
        return self.test_loader
