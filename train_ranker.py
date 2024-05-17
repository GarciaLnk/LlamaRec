import os

import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from accelerate import PartialState
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from pytorch_lightning import seed_everything
from transformers import BitsAndBytesConfig

from config import EXPERIMENT_ROOT, PROJECT_NAME, args, set_template
from dataloader import dataloader_factory
from model import LlamaForCausalLM
from trainer import LLMTrainer

try:
    os.environ["WANDB_PROJECT"] = PROJECT_NAME
except:
    print("WANDB_PROJECT not available, please set it in config.py")


def main(args, export_root=None):
    seed_everything(args.seed)
    if export_root == None:
        export_root = (
            EXPERIMENT_ROOT
            + "/"
            + args.llm_base_model.split("/")[-1]
            + "/"
            + args.dataset_code
        )

    (
        train_loader,
        val_loader,
        test_loader,
        tokenizer,
        test_retrieval,
    ) = dataloader_factory(args)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    is_distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    device_map = {"": PartialState().process_index} if is_distributed else "auto"
    model = LlamaForCausalLM.from_pretrained(
        args.llm_base_model,
        quantization_config=bnb_config,
        device_map=device_map,
        attn_implementation="flash_attention_2",
    )
    if args.lora_gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    model.config.use_cache = False
    trainer = LLMTrainer(
        args,
        model,
        train_loader,
        val_loader,
        test_loader,
        tokenizer,
        export_root,
        args.use_wandb,
    )

    trainer.train()
    trainer.test(test_retrieval)


if __name__ == "__main__":
    args.model_code = "llm"
    set_template(args)
    main(args, export_root=None)
