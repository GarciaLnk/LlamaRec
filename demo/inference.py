import json
import pickle
import re

import torch
from peft.auto import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
compute_capability = float(".".join(map(str, torch.cuda.get_device_capability())))


def load_dataset_map(dataset_path: str = "dataset_meta.pkl") -> dict[int, str]:
    with open(dataset_path, "rb") as f:
        dataset_map = pickle.load(f)
    return dataset_map


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
    dataset_map: dict[int, str],
    top_k: int = 10,
) -> list[str]:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model(input_ids=inputs["input_ids"])
    logits = outputs.logits.view(1, -1, outputs.logits.size(-1))

    curr_top_k = min(top_k, len(candidates))
    ranked_candidates = []
    seen_candidates = set()
    pattern = re.compile(r"^[A-Z]$")
    while len(ranked_candidates) < top_k:
        top_tokens = torch.topk(logits, curr_top_k, dim=-1)
        decoded_tokens = tokenizer.batch_decode(
            top_tokens.indices[0, -1, :], skip_special_tokens=True
        )
        for token in decoded_tokens:
            token = token.strip()
            if token and pattern.match(token) and token not in seen_candidates:
                seen_candidates.add(token)
                ranked_candidates.append(token)
                if len(ranked_candidates) == top_k:
                    break
        if len(ranked_candidates) < top_k:
            curr_top_k += top_k

    candidates = [candidates[ord(token) - ord("A")] for token in ranked_candidates]
    result = [dataset_map[candidate] for candidate in candidates]

    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def generate_prompt(
    query: list[int], candidates: list[int], dataset_map: dict[int, str]
) -> str:
    template_file = "template.json"
    instruction = "Given user history in chronological order, recommend an item from the candidate pool with its index letter."
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
