from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
import torch
import clip

from torch_geometric.data import download_url


def download_numberbatch(path: Path):
    return download_url(f"https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/{path.name}", path.parent)


def load_numberbatch(path: Path | str) -> pd.DataFrame:

    if not isinstance(path, Path):
        path = Path(path)

    if path.suffix != ".txt.gz":
        numberbatch_path = path / "numberbatch-en-19.08.txt.gz"
    else:
        numberbatch_path = path

    if not path.exists():
        path.mkdir(parents=True)

    if not numberbatch_path.is_file():
        download_path = download_numberbatch(numberbatch_path)
        print(f"Downloaded numberbatch at {download_path}")

    numberbatch = pd.read_csv(numberbatch_path, sep=" ", header=None, skiprows=1, compression="gzip")
    numberbatch.rename(columns={0: "word"}, inplace=True)
    numberbatch.dropna(inplace=True)  # currently 2 NaNs
    numberbatch.iloc[:, 1:] = numberbatch.iloc[:, 1:].astype(float)

    return numberbatch


def get_numberbatch_embeddings(classes: Iterable[str], numberbatch: pd.DataFrame, drop_words=True) -> Dict[str, np.ndarray]:

    if isinstance(numberbatch, (str, Path)):
        numberbatch_path = Path(numberbatch)
        numberbatch = load_numberbatch(numberbatch_path)

    class_embeddings = numberbatch[numberbatch["word"].isin(classes)]

    class_embeddings = class_embeddings.assign(word=pd.Categorical(class_embeddings["word"], categories=classes, ordered=True))
    class_embeddings = class_embeddings.sort_values("word")

    class_embeddings = class_embeddings.reset_index(drop=drop_words).iloc[:, 1:]

    return class_embeddings


def save_class_embeddings(class_embeddings: pd.DataFrame, path: Path | str):

    if not isinstance(path, Path):
        path = Path(path)

    if path.suffix == ".csv":
        class_embeddings.to_csv(path, index=False)
    elif path.suffix == ".pkl":
        class_embeddings.to_pickle(path)
    elif path.suffix == ".npy":
        np.save(path, class_embeddings.to_numpy(), allow_pickle=False)
    else:
        raise ValueError("Unsupported file format. Supported formats are .csv, .pkl, .npy")


def get_clip_text_features(classes: Iterable[str], clip_model="ViT-L/14", dtype=np.float32, download_root: str | None = None) -> np.ndarray:

    enhanced_labels = ["a " + t for t in classes]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(clip_model, device=device, download_root=download_root)

    tokens = clip.tokenize(enhanced_labels).to(device)

    with torch.no_grad():
        text_features = model.encode_text(tokens)

    return text_features.cpu().numpy().astype(dtype)
