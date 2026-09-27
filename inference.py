# 模型推理层：保留原来的特征计算、模型结构和预测公式。
# Web 路由放在 app.py；这个文件只负责加载模型和执行预测。
from pathlib import Path
import gc
import os

import torch
import esm

from model import Predictor, AA_FEATURE_DIM
from mutation_utils import (
    create_multi_mutant,
    mutation_set_label,
    normalize_sequence,
    parse_mutation_set,
)

MODEL_PATH = Path(__file__).resolve().parent / "model.pt"
MAX_MUTATIONS = 10

SCALAR_KEYS = ["hydrophobicity", "volume", "polarity", "charge"]
FLAG_KEYS = ["aromatic", "polar", "aliphatic", "glycine", "proline", "cysteine"]

AA_PROPERTIES = {
    "A": {"hydrophobicity": 1.8, "volume": 88.6, "polarity": 8.1, "charge": 0.0},
    "R": {"hydrophobicity": -4.5, "volume": 173.4, "polarity": 10.5, "charge": 1.0},
    "N": {"hydrophobicity": -3.5, "volume": 114.1, "polarity": 11.6, "charge": 0.0},
    "D": {"hydrophobicity": -3.5, "volume": 111.1, "polarity": 13.0, "charge": -1.0},
    "C": {"hydrophobicity": 2.5, "volume": 108.5, "polarity": 5.5, "charge": 0.0},
    "Q": {"hydrophobicity": -3.5, "volume": 143.8, "polarity": 10.5, "charge": 0.0},
    "E": {"hydrophobicity": -3.5, "volume": 138.4, "polarity": 12.3, "charge": -1.0},
    "G": {"hydrophobicity": -0.4, "volume": 60.1, "polarity": 9.0, "charge": 0.0},
    "H": {"hydrophobicity": -3.2, "volume": 153.2, "polarity": 10.4, "charge": 0.5},
    "I": {"hydrophobicity": 4.5, "volume": 166.7, "polarity": 5.2, "charge": 0.0},
    "L": {"hydrophobicity": 3.8, "volume": 166.7, "polarity": 4.9, "charge": 0.0},
    "K": {"hydrophobicity": -3.9, "volume": 168.6, "polarity": 11.3, "charge": 1.0},
    "M": {"hydrophobicity": 1.9, "volume": 162.9, "polarity": 5.7, "charge": 0.0},
    "F": {"hydrophobicity": 2.8, "volume": 189.9, "polarity": 5.2, "charge": 0.0},
    "P": {"hydrophobicity": -1.6, "volume": 112.7, "polarity": 8.0, "charge": 0.0},
    "S": {"hydrophobicity": -0.8, "volume": 89.0, "polarity": 9.2, "charge": 0.0},
    "T": {"hydrophobicity": -0.7, "volume": 116.1, "polarity": 8.6, "charge": 0.0},
    "W": {"hydrophobicity": -0.9, "volume": 227.8, "polarity": 5.4, "charge": 0.0},
    "Y": {"hydrophobicity": -1.3, "volume": 193.6, "polarity": 6.2, "charge": 0.0},
    "V": {"hydrophobicity": 4.2, "volume": 140.0, "polarity": 5.9, "charge": 0.0},
}

AA_FLAGS = {
    aa: {
        "aromatic": float(aa in {"F", "W", "Y", "H"}),
        "polar": float(aa in {"S", "T", "N", "Q", "C", "Y", "H", "D", "E", "K", "R"}),
        "aliphatic": float(aa in {"A", "V", "I", "L", "M"}),
        "glycine": float(aa == "G"),
        "proline": float(aa == "P"),
        "cysteine": float(aa == "C"),
    }
    for aa in AA_PROPERTIES
}

_NORM_MEANS = {
    k: sum(p[k] for p in AA_PROPERTIES.values()) / len(AA_PROPERTIES)
    for k in SCALAR_KEYS
}
_NORM_STDS = {
    k: (sum((p[k] - _NORM_MEANS[k]) ** 2 for p in AA_PROPERTIES.values()) / len(AA_PROPERTIES)) ** 0.5
    for k in SCALAR_KEYS
}


def pair_physchem_features(wt_aa, mt_aa):
    wt, mt = wt_aa.upper(), mt_aa.upper()
    features = []
    for k in SCALAR_KEYS:
        wt_n = (AA_PROPERTIES[wt][k] - _NORM_MEANS[k]) / _NORM_STDS[k]
        mt_n = (AA_PROPERTIES[mt][k] - _NORM_MEANS[k]) / _NORM_STDS[k]
        features.append(mt_n - wt_n)
    for k in FLAG_KEYS:
        features.append(AA_FLAGS[mt][k] - AA_FLAGS[wt][k])
    return torch.tensor(features, dtype=torch.float32)


def aggregate_physchem_features(mutations):
    return torch.stack(
        [pair_physchem_features(mutation.wildtype, mutation.mutant) for mutation in mutations],
        dim=0,
    ).sum(dim=0)


class ModelService:
    """一个应用进程使用一个模型对象，由 FastAPI lifespan 创建和关闭。"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cpu":
            torch.set_num_threads(int(os.environ.get("DDG_CPU_THREADS", "4")))
        # 复用机器上已存在的 ESM 缓存；缺少缓存时默认下载到项目 D 盘目录。
        cached = Path(torch.hub.get_dir()) / "checkpoints" / "esm2_t33_650M_UR50D.pt"
        if not os.environ.get("TORCH_HOME") and not cached.exists():
            torch.hub.set_dir(str(MODEL_PATH.parent / ".local" / "torch" / "hub"))
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.esm_model = model.to(self.device).eval()
        self.batch_converter = alphabet.get_batch_converter()
        self.predictor = Predictor().to(self.device).eval()
        state_dict = torch.load(MODEL_PATH, map_location=self.device, weights_only=True)
        self.predictor.load_state_dict(state_dict)

    def extract_embedding(self, sequence):
        data = [(sequence, sequence)]
        _, _, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)
        with torch.no_grad():
            results = self.esm_model(batch_tokens, repr_layers=[33], return_contacts=False)
            token_repr = results["representations"][33].squeeze(0)
            embedding = token_repr[1: len(sequence) + 1, :].cpu()
        return embedding

    def predict(self, sequence, mutation_set):
        mutations = parse_mutation_set(mutation_set, max_mutations=MAX_MUTATIONS)
        mutant_seq = create_multi_mutant(sequence, mutations)
        wt_emb = self.extract_embedding(sequence)
        vt_emb = self.extract_embedding(mutant_seq)
        wt_tensor = wt_emb.unsqueeze(0).float().to(self.device)
        vt_tensor = vt_emb.unsqueeze(0).float().to(self.device)
        length = torch.tensor([wt_emb.shape[0]], dtype=torch.long, device=self.device)
        aa_feat = aggregate_physchem_features(mutations).unsqueeze(0).to(self.device)
        with torch.no_grad():
            ddg = self.predictor(wt_tensor, vt_tensor, lengths=length, aa_features=aa_feat).item()
        return ddg, mutation_set_label(mutations)

    def close(self):
        self.esm_model = self.predictor = self.batch_converter = None
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
