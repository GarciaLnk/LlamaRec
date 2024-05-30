import pickle

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dataset_map(dataset_path: str = "dataset_meta.pkl") -> dict[int, str]:
    with open(dataset_path, "rb") as f:
        dataset_map = pickle.load(f)
    return dataset_map


def load_retriever(model_path: str = "retriever.pth") -> torch.jit.ScriptModule:
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model


def retrieve_candidates(model, query: list[int], top_k: int = 20) -> list[int]:
    seqs = torch.tensor(query).unsqueeze(0).to(device)
    candidates = model(seqs)[:, -1, :]
    candidates = torch.topk(candidates, top_k).indices[0].tolist()

    if device.type == "cuda":
        torch.cuda.empty_cache()
    return candidates
