import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from accelerate import PartialState

from config import EXPERIMENT_ROOT, PROJECT_NAME, args, set_template
from dataloader import dataloader_factory
from model import FastLlamaModelPatched
from trainer import LLMTrainer

from pytorch_lightning import seed_everything  # isort: skip

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
    is_distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    device_map = {"": PartialState().process_index} if is_distributed else "sequential"
    model, _ = FastLlamaModelPatched.from_pretrained(
        model_name=args.llm_base_model,
        load_in_4bit=True,
        device_map=device_map,
    )
    model = FastLlamaModelPatched.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing=args.lora_gradient_checkpointing,
    )
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
