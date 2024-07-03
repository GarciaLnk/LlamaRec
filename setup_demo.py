import glob
import os
import pickle

import torch
from peft.auto import AutoPeftModelForCausalLM
from pytorch_lightning import seed_everything
from transformers import AutoTokenizer, BitsAndBytesConfig

from config import EXPERIMENT_ROOT, STATE_DICT_KEY, args, set_template
from dataloader import LRUDataloader
from demo.index import create_index
from llamarec_datasets import dataset_factory
from model import LRURec
from model.llm import AutoModelForCausalLMPatched


def main(args):
    seed_everything(args.seed)

    dataset = dataset_factory(args)
    dataset_dict = dataset.load_dataset()
    if not os.path.exists("demo/dataset_meta.pkl"):
        combined_meta = {
            "meta": dataset_dict["meta"],
            "spotify_meta": dataset_dict["spotify_meta"],
        }
        with open("demo/dataset_meta.pkl", "wb") as f:
            pickle.dump(combined_meta, f)

    create_index("demo/dataset_meta.pkl", "demo/indexdir")

    LRUDataloader(args, dataset)
    model = LRURec(args)

    export_root = EXPERIMENT_ROOT + "/" + args.model_code + "/" + args.dataset_code
    print("Loading retriever model at ", export_root)
    retriever_path = os.path.join(export_root, "models", "best_acc_model.pth")
    if not retriever_path:
        print("Retriever model not found.")
        return

    best_model_dict = torch.load(retriever_path).get(STATE_DICT_KEY)
    model.load_state_dict(best_model_dict)
    model.eval()
    model_scripted = torch.jit.script(model)
    model_scripted.save("demo/retriever.pth")
    print("Retriever model saved to demo directory.")

    export_root = (
        EXPERIMENT_ROOT
        + "/"
        + args.llm_base_model.split("/")[-1]
        + "/"
        + args.dataset_code
    )
    checkpoint_dirs = sorted(glob.glob(f"{export_root}/checkpoint-*"))
    if checkpoint_dirs:
        export_root = checkpoint_dirs[0]
    else:
        print("LLM model not found.")
        return

    print("Loading LLM model at ", export_root)
    llm_path = "demo/llm"
    device_map = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=(
            torch.bfloat16 if torch.cuda.is_bf16_supported() else None
        ),
    )
    base_model = AutoModelForCausalLMPatched.from_pretrained(
        args.llm_base_model,
        quantization_config=bnb_config,
        device_map=device_map,
        attn_implementation="flash_attention_2",
    )
    base_model.eval()
    base_model.save_pretrained(llm_path)
    print("Base LLM saved to demo directory.")

    tokenizer = AutoTokenizer.from_pretrained(export_root)
    tokenizer.save_pretrained(llm_path)
    print("Tokenizer saved to demo directory.")

    peft_model = AutoPeftModelForCausalLM.from_pretrained(
        export_root,
        quantization_config=bnb_config,
        device_map=device_map,
        attn_implementation="flash_attention_2",
    )
    peft_model.peft_config["default"].base_model_name_or_path = llm_path.split("/")[-1]
    peft_model.eval()
    peft_model.save_pretrained(llm_path)
    print("LoRA adapter saved to demo directory.")

    print("Demo setup complete.")


if __name__ == "__main__":
    args.model_code = "lru"
    args.dataset_code = "music"
    set_template(args)
    main(args)
