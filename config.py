import argparse

import torch

RAW_DATASET_ROOT_FOLDER = "data"
EXPERIMENT_ROOT = "experiments"
STATE_DICT_KEY = "model_state_dict"
OPTIMIZER_STATE_DICT_KEY = "optimizer_state_dict"
PROJECT_NAME = "llmrec"


def set_template(args):
    if args.dataset_code is None:
        print("******************** Dataset Selection ********************")
        dataset_code = {"1": "ml-100k", "b": "beauty", "g": "games", "m": "music"}
        args.dataset_code = dataset_code[
            input("Input 1 for ml-100k, b for beauty, g for games and m for music: ")
        ]

    match args.llm:
        case "phi3":
            if not args.llm_enable_unsloth:
                args.llm_base_model = "microsoft/Phi-3-mini-4k-instruct"
                args.llm_base_tokenizer = "microsoft/Phi-3-mini-4k-instruct"
                args.lora_target_modules = ["qkv_proj"]
            else:
                args.llm_base_model = "unsloth/Phi-3-mini-4k-instruct-bnb-4bit"
                args.llm_base_tokenizer = "unsloth/Phi-3-mini-4k-instruct-bnb-4bit"
        case "llama2":
            args.llm_base_model = "meta-llama/Llama-2-7b-hf"
            args.llm_base_tokenizer = "meta-llama/Llama-2-7b-hf"
        case "llama3":
            args.llm_base_model = "meta-llama/Meta-Llama-3-8B"
            args.llm_base_tokenizer = "meta-llama/Meta-Llama-3-8B"

    if args.bert_max_len is None:
        if args.dataset_code == "ml-100k":
            args.bert_max_len = 200
        else:
            args.bert_max_len = 50

    if args.bert_max_predictions is None:
        if args.dataset_code == "ml-100k":
            args.bert_max_predictions = 40
        else:
            args.bert_max_predictions = 20

    # fmt: off
    if args.dataset_code == "music":
        args.min_uc = 20 if args.min_uc is None else args.min_uc
        args.min_sc = 20 if args.min_sc is None else args.min_sc 
        args.val_strategy = "epoch" if args.val_strategy is None else args.val_strategy
        args.early_stopping_patience = 5 if args.early_stopping_patience is None else args.early_stopping_patience
        args.lora_val_iterations = 500 if args.lora_val_iterations is None else args.lora_val_iterations
        args.lora_val_delay = 5000 if args.lora_val_delay is None else args.lora_val_delay
        args.lora_early_stopping_patience = 4 if args.lora_early_stopping_patience is None else args.lora_early_stopping_patience
        args.lora_max_steps = 20000 if args.lora_max_steps is None else args.lora_max_steps
    # fmt: on

    if args.min_uc is None:
        args.min_uc = 5
    if args.min_sc is None:
        args.min_sc = 5

    if args.val_strategy is None:
        args.val_strategy = "iteration"
    if args.early_stopping_patience is None:
        args.early_stopping_patience = 20

    if args.lora_val_iterations is None:
        args.lora_val_iterations = 100
    if args.lora_val_delay is None:
        args.lora_val_delay = 0
    if args.lora_early_stopping_patience is None:
        args.lora_early_stopping_patience = 20
    if args.lora_max_steps is None:
        args.lora_max_steps = -1

    if "llm" in args.model_code:
        batch = 16 if args.dataset_code != "beauty" else 8
        if args.lora_micro_batch_size is None:
            args.lora_micro_batch_size = batch
        batch = 16 if args.dataset_code != "ml-100k" else 32
    else:
        batch = 16 if args.dataset_code == "ml-100k" else 64

    if args.train_batch_size is None:
        args.train_batch_size = batch
    if args.val_batch_size is None:
        args.val_batch_size = batch
    if args.test_batch_size is None:
        args.test_batch_size = batch

    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
        else:
            args.device = "cpu"

    if args.optimizer is None:
        args.optimizer = "AdamW"
    if args.lr is None:
        args.lr = 0.001
    if args.weight_decay is None:
        args.weight_decay = 0.01
    if args.enable_lr_schedule is None:
        args.enable_lr_schedule = False
    if args.decay_step:
        args.decay_step = 10000
    if args.gamma is None:
        args.gamma = 1.0
    if args.enable_lr_warmup is None:
        args.enable_lr_warmup = False
    if args.warmup_steps is None:
        args.warmup_steps = 100

    if args.metric_ks is None:
        args.metric_ks = [1, 5, 10, 20, 50]
    if args.rerank_metric_ks is None:
        args.rerank_metric_ks = [1, 5, 10]
    if args.best_metric is None:
        args.best_metric = "Recall@10"
    if args.rerank_best_metric is None:
        args.rerank_best_metric = "NDCG@10"

    if args.bert_num_blocks is None:
        args.bert_num_blocks = 2
    if args.bert_num_heads is None:
        args.bert_num_heads = 2


parser = argparse.ArgumentParser()

################
# Dataset
################
parser.add_argument("--dataset_code", type=str, default=None)
parser.add_argument("--min_rating", type=int, default=0)
parser.add_argument("--min_uc", type=int, default=None)
parser.add_argument("--min_sc", type=int, default=None)
parser.add_argument("--seed", type=int, default=42)

################
# Dataloader
################
parser.add_argument("--train_batch_size", type=int, default=None)
parser.add_argument("--val_batch_size", type=int, default=None)
parser.add_argument("--test_batch_size", type=int, default=None)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--sliding_window_size", type=float, default=1.0)
parser.add_argument("--negative_sample_size", type=int, default=10)

################
# Trainer
################
# optimization #
parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
parser.add_argument("--num_epochs", type=int, default=500)
parser.add_argument("--optimizer", type=str, default=None, choices=["AdamW", "Adam"])
parser.add_argument("--weight_decay", type=float, default=None)
parser.add_argument("--adam_epsilon", type=float, default=1e-9)
parser.add_argument("--momentum", type=float, default=None)
parser.add_argument("--lr", type=float, default=None)
parser.add_argument("--max_grad_norm", type=float, default=5.0)
parser.add_argument("--enable_lr_schedule", type=bool, default=None)
parser.add_argument("--decay_step", type=int, default=None)
parser.add_argument("--gamma", type=float, default=None)
parser.add_argument("--enable_lr_warmup", type=bool, default=None)
parser.add_argument("--warmup_steps", type=int, default=None)

# evaluation #
parser.add_argument(
    "--val_strategy", type=str, default=None, choices=["epoch", "iteration"]
)
parser.add_argument(
    "--val_iterations", type=int, default=500
)  # only for iteration val_strategy
parser.add_argument("--early_stopping", type=bool, default=True)
parser.add_argument("--early_stopping_patience", type=int, default=None)
parser.add_argument("--metric_ks", nargs="+", type=int, default=None)
parser.add_argument("--rerank_metric_ks", nargs="+", type=int, default=None)
parser.add_argument("--best_metric", type=str, default=None)
parser.add_argument("--rerank_best_metric", type=str, default=None)
parser.add_argument("--use_wandb", type=bool, default=False)

################
# Retriever Model
################
parser.add_argument("--hyperparam_search", type=bool, default=False)
parser.add_argument("--model_code", type=str, default=None)
parser.add_argument("--bert_max_len", type=int, default=None)
parser.add_argument("--bert_hidden_units", type=int, default=64)
parser.add_argument("--bert_num_blocks", type=int, default=None)
parser.add_argument("--bert_num_heads", type=int, default=None)
parser.add_argument("--bert_head_size", type=int, default=None)
parser.add_argument("--bert_dropout", type=float, default=0.2)
parser.add_argument("--bert_attn_dropout", type=float, default=0.2)
parser.add_argument("--bert_mask_prob", type=float, default=0.25)
parser.add_argument("--bert_max_predictions", type=float, default=20)

################
# LLM Model
################
parser.add_argument(
    "--llm", type=str, default=None, choices=["llama2", "llama3", "phi3"]
)
parser.add_argument("--llm_enable_unsloth", action="store_true")
parser.add_argument("--llm_base_model", type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument(
    "--llm_base_tokenizer", type=str, default="meta-llama/Llama-2-7b-hf"
)
parser.add_argument("--llm_max_title_len", type=int, default=32)
parser.add_argument("--llm_max_text_len", type=int, default=1536)
parser.add_argument("--llm_max_history", type=int, default=20)
parser.add_argument("--llm_train_on_inputs", type=bool, default=False)
parser.add_argument(
    "--llm_negative_sample_size", type=int, default=19
)  # 19 negative & 1 positive
parser.add_argument(
    "--llm_system_template",
    type=str,  # instruction
    default="Given user history in chronological order, recommend an item from the candidate pool with its index letter.",
)
parser.add_argument(
    "--llm_input_template", type=str, default="User history: {}; \n Candidate pool: {}"
)
parser.add_argument("--llm_load_in_4bit", type=bool, default=True)
parser.add_argument("--llm_retrieved_path", type=str, default=None)
parser.add_argument("--llm_cache_dir", type=str, default=None)

################
# Lora
################
parser.add_argument("--lora_r", type=int, default=8)
parser.add_argument("--lora_alpha", type=int, default=32)
parser.add_argument("--lora_dropout", type=float, default=0.05)
parser.add_argument("--lora_target_modules", type=list, default=["q_proj", "v_proj"])
parser.add_argument("--lora_num_epochs", type=int, default=1)
parser.add_argument("--lora_val_iterations", type=int, default=None)
parser.add_argument("--lora_val_delay", type=int, default=None)
parser.add_argument("--lora_val_accumulation_steps", type=int, default=500)
parser.add_argument("--lora_early_stopping_patience", type=int, default=None)
parser.add_argument("--lora_max_steps", type=int, default=None)
parser.add_argument("--lora_lr", type=float, default=2e-4)
parser.add_argument("--lora_micro_batch_size", type=int, default=None)
parser.add_argument("--lora_gradient_checkpointing", type=bool, default=True)

################


args = parser.parse_args()
