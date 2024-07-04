import json
import pickle

import torch
from peft.auto import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
compute_capability = float(".".join(map(str, torch.cuda.get_device_capability())))


def load_dataset_map(
    dataset_path: str = "dataset_meta.pkl",
) -> tuple[dict[int, str], dict[int, str]]:
    with open(dataset_path, "rb") as f:
        dataset_map = pickle.load(f)
    return dataset_map["meta"], dataset_map["spotify_meta"]


def load_retriever(model_path: str = "retriever.pth") -> torch.jit.ScriptModule:
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model


def load_llm(
    llm_path: str = "llm",
) -> tuple[AutoPeftModelForCausalLM, PreTrainedTokenizer | PreTrainedTokenizerFast]:
    tokenizer_name_or_path = llm_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

    if compute_capability >= 8.0:
        attn_implementation = "flash_attention_2"
    else:
        attn_implementation = "sdpa"
    model = AutoPeftModelForCausalLM.from_pretrained(
        llm_path,
        device_map=device,
        attn_implementation=attn_implementation,
    )
    model.eval()
    print("Loaded LLM model.")
    return model, tokenizer


def retrieve_candidates(model, query: list[int], top_k: int = 20) -> list[int]:
    seqs = torch.tensor(query).unsqueeze(0).to(device)
    candidates = model(seqs)[:, -1, :]
    candidates = torch.topk(candidates, top_k).indices[0].tolist()

    if device.type == "cuda":
        torch.cuda.empty_cache()
    return candidates


def rank_candidates(
    model,
    tokenizer,
    prompt: str,
    candidates: list[int],
    verbalizer,
    top_k: int = 10,
) -> list[int]:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model(input_ids=inputs["input_ids"])

    logits = outputs.logits[:, -1].to("cpu")
    scores = verbalizer.process_logits(logits)

    top_candidates = [
        candidates[i] for i in torch.topk(scores, top_k).indices.squeeze().tolist()
    ]

    if device.type == "cuda":
        torch.cuda.empty_cache()
    return top_candidates


def generate_prompt(
    query: list[int],
    candidates: list[int],
    dataset_map: dict[int, str],
    instruction: str = "Given user history in chronological order, recommend an item from the candidate pool with its index letter.",
) -> str:
    template_file = "template.json"
    input_template = "User history: {}; \n Candidate pool: {}"

    q_t = " \n ".join(
        [
            "(" + str(idx + 1) + ") " + dataset_map[item]
            for idx, item in enumerate(query)
        ]
    )

    c_t = " \n ".join(
        [
            "(" + chr(ord("A") + idx) + ") " + dataset_map[item]
            for idx, item in enumerate(candidates)
        ]
    )

    prompt_input = input_template.format(q_t, c_t)

    with open(template_file, encoding="utf-8") as fp:
        template = json.load(fp)
    prompt = template["prompt_input"].format(
        instruction=instruction, input=prompt_input
    )
    return prompt
