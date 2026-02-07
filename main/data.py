import glob
import json
from pathlib2 import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Union,
)

import datasets
import numpy as np
import torch
from PIL import Image
from datasets.utils import logging

logger = logging.get_logger(__name__)


def resolve_model_key(model_key_or_path: str) -> str:
    """Resolve a clean key name from either 'clip_vit_h14' or '/path/to/clip_vit_h14'."""
    return Path(model_key_or_path).stem


def ensure_image_id(ds: datasets.Dataset) -> datasets.Dataset:
    if "image_id" in ds.column_names:
        return ds

    def _fn(ex: Dict[str, Any]) -> Dict[str, Any]:
        img: Image.Image = ex["image"]
        stem = Path(getattr(img, "filename", "")).stem
        if not stem:
            raise ValueError(
                "Cannot derive image_id because PIL image has no filename."
            )
        ex["image_id"] = stem
        return ex

    return ds.map(_fn, desc="ensure_image_id")


# =========================================================
# 1) Image datasets
# =========================================================
def load_image_dataset(
    *,
    dataset_name: str,
    image_directory: str,
    split: Literal["train", "test", "val", "validation"],
    cache_dir: str = ".cache",
) -> datasets.Dataset:
    """Return a HF Dataset with at least 'image' (PIL.Image) and 'image_id'."""
    name = dataset_name.lower()

    # ImageNet embeddings pipeline often uses parquet for images (per your build_embeddings.py)
    if name in ("imagenet", "imagenet1k", "in1k", "imagenet-1k"):
        hf_split = "train" if split == "train" else "validation"
        ds = datasets.load_dataset(
            "parquet",
            data_dir=image_directory,
            split=hf_split,
            cache_dir=cache_dir,
        )
        return ensure_image_id(ds)

    # THINGS image folder layout
    if name in ("things",):
        root = Path(image_directory)
        if split == "train":
            pattern = str(root / "training_images" / "**" / "*.jpg")
            split_key = "train"
        else:
            pattern = str(root / "test_images" / "**" / "*.jpg")
            split_key = "test"
        ds = datasets.load_dataset(
            "imagefolder",
            data_files={split_key: pattern},
            split=split_key,
            cache_dir=cache_dir,
        )
        return ensure_image_id(ds)

    # generic imagefolder
    pattern = str(Path(image_directory) / "**" / "*.jpg")
    ds = datasets.load_dataset(
        "imagefolder",
        data_files={split: pattern},
        split=split,
        cache_dir=cache_dir,
    )
    return ensure_image_id(ds)


# =========================================================
# 2) Embedding parquet datasets (MULTI-model ONLY)
# =========================================================
def find_embedding_parquets(
    *,
    embedding_directory: str,
    dataset_key: str,
    split: Literal["train", "test", "validation", "val"],
    model_key: str,
) -> List[str]:
    """Find embedding parquet shards for one model key.

    Prefer canonical shards:
      {dataset_key}_{split}_{model_key}-part-*-of-*.parquet
    Otherwise accept rank-style:
      {dataset_key}_{split}_{model_key}.rank-*-of-*.part-*.parquet
    """
    mk = resolve_model_key(model_key)
    base = f"{dataset_key}_{split}_{mk}"
    d = Path(embedding_directory)

    canon = sorted(glob.glob(str(d / f"{base}-part-*-of-*.parquet")))
    if canon:
        return canon
    rankish = sorted(glob.glob(str(d / f"{base}.rank-*-of-*.part-*.parquet")))
    return rankish


def load_embedding_datasets(
    *,
    embedding_directory: str,
    dataset_key: str,
    split: Literal["train", "test", "validation", "val"],
    model_keys: Sequence[str],
    cache_dir: str = ".cache",
    emb_column: str = "emb",
) -> Dict[str, datasets.Dataset]:
    """Load multiple embedding datasets.

    Returns:
      dict {resolved_model_key: Dataset(image_id, emb)}
    """
    if not model_keys:
        raise ValueError("model_keys must be non-empty.")

    out: Dict[str, datasets.Dataset] = {}
    for mk in model_keys:
        k = resolve_model_key(mk)
        files = find_embedding_parquets(
            embedding_directory=embedding_directory,
            dataset_key=dataset_key,
            split=split,
            model_key=k,
        )
        if not files:
            raise FileNotFoundError(
                f"No embedding parquet found: {dataset_key}_{split}_{k} in {embedding_directory}"
            )

        ds = datasets.load_dataset(
            "parquet",
            data_files={"data": files},
            split="data",
            cache_dir=cache_dir,
        )
        if "image_id" not in ds.column_names:
            raise KeyError(
                f"Embedding parquet missing 'image_id'. Example file: {files[0]}"
            )
        if emb_column not in ds.column_names:
            raise KeyError(
                f"Embedding parquet missing '{emb_column}'. Example file: {files[0]}"
            )

        # drop unused cols to reduce memory
        keep = ["image_id", emb_column]
        drop = [c for c in ds.column_names if c not in keep]
        if drop:
            ds = ds.remove_columns(drop)

        out[k] = ds

    return out


# =========================================================
# 3) Brain dataset (THINGS .pt -> HF Dataset)
# =========================================================
def _read_channel_name_to_index_jsonl(jsonl_path: str) -> Dict[str, int]:
    name2idx: Dict[str, int] = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            name2idx[obj["name"]] = int(obj["index"])
    return name2idx


def _parse_selected_channels(
    selected_channels: Optional[Union[str, Sequence[str]]],
) -> Optional[List[str]]:
    if selected_channels is None:
        return None
    if isinstance(selected_channels, str):
        s = selected_channels.strip()
        if not s:
            return None
        return [x.strip() for x in s.split(",") if x.strip()]
    return [str(x).strip() for x in selected_channels if str(x).strip()]


def _selected_channel_indices_from_jsonl(
    selected_channels: Optional[Union[str, Sequence[str]]],
    jsonl_path: str,
) -> Optional[List[int]]:
    names = _parse_selected_channels(selected_channels)
    if not names:
        return None
    name2idx = _read_channel_name_to_index_jsonl(jsonl_path)
    missing = [n for n in names if n not in name2idx]
    if missing:
        raise KeyError(f"Unknown channel names in selected_channels: {missing}")
    return [name2idx[n] for n in names]


def load_things_brain_dataset(
    *,
    data_directory: str,
    split: Literal["train", "test"],
    subject_ids: Union[int, Sequence[int]] = (1,),
    brain_key: Literal["eeg", "meg"] = "eeg",
    avg_trials: bool = True,
    selected_channels: Optional[Union[str, Sequence[str]]] = None,  # EEG names
    eeg_channel_jsonl: str = "configs/THINGS_EEG_CHANNELS.jsonl",
) -> datasets.Dataset:
    """Build HF Dataset with columns:
    - {brain_key}: Array2D float32 [C, T]
    - image_id: string
    - subject_id: int32
    """
    if isinstance(subject_ids, int):
        subject_ids = (subject_ids,)
    subject_ids = [int(x) for x in subject_ids]

    sel_idx: Optional[List[int]] = None
    if brain_key == "eeg" and selected_channels is not None:
        sel_idx = _selected_channel_indices_from_jsonl(
            selected_channels, eeg_channel_jsonl
        )

    all_ds: List[datasets.Dataset] = []
    for sid in subject_ids:
        pt_path = Path(data_directory).joinpath(f"sub-{sid:02d}", f"{split}.pt")
        loaded = torch.load(str(pt_path), weights_only=False)

        x = torch.as_tensor(loaded[brain_key])  # [N,TRIAL,C,T] or [N,C,T]
        if x.ndim == 4:
            if avg_trials:
                x = x.mean(dim=1)  # [N,C,T]
            else:
                x = x.reshape(-1, *x.shape[2:])  # [N*TRIAL,C,T]
        elif x.ndim != 3:
            raise ValueError(
                f"Unexpected {brain_key} shape: {tuple(x.shape)} in {pt_path}"
            )

        if sel_idx is not None:
            x = x[:, sel_idx, :]

        imgs = np.array(loaded["img"])
        if avg_trials:
            if imgs.ndim == 2:
                imgs = imgs[:, 0]
            imgs = imgs.reshape(-1)[: x.shape[0]]
        else:
            imgs = imgs.reshape(-1)

        image_ids = [Path(p).stem for p in imgs.tolist()]
        if len(image_ids) != x.shape[0]:
            raise ValueError(
                f"Brain/image mismatch: {x.shape[0]} vs {len(image_ids)} for {pt_path}"
            )

        x_np = x.float().cpu().numpy()  # [N,C,T]
        C, T = x_np.shape[1], x_np.shape[2]

        features = datasets.Features(
            {
                brain_key: datasets.Array2D(shape=(C, T), dtype="float32"),
                "image_id": datasets.Value("string"),
                "subject_id": datasets.Value("int32"),
            }
        )

        ds = datasets.Dataset.from_dict(
            {
                brain_key: list(x_np),
                "image_id": image_ids,
                "subject_id": [sid] * len(image_ids),
            },
            features=features,
        )
        all_ds.append(ds)

    return datasets.concatenate_datasets(all_ds) if len(all_ds) > 1 else all_ds[0]


# =========================================================
# 4) Pairing / task builders (MULTI-emb)
# =========================================================
PairMode = Literal["by_image_id", "by_index"]


def build_brain_with_emb_dataset(
    *,
    brain_ds: datasets.Dataset,
    emb_dss: Mapping[str, datasets.Dataset],
    emb_prefix: str = "emb_",
    emb_column: str = "emb",
) -> datasets.Dataset:
    """Task 2: brain + multi-emb (by image_id)."""
    id2i_map: Dict[str, Dict[str, int]] = {}
    for k, ds in emb_dss.items():
        ids = ds["image_id"]
        id2i_map[k] = {iid: i for i, iid in enumerate(ids)}

    def _fn(ex: Dict[str, Any]) -> Dict[str, Any]:
        iid = ex["image_id"]
        for k, ds in emb_dss.items():
            j = id2i_map[k].get(iid, None)
            if j is None:
                raise KeyError(f"Missing image_id in emb_ds[{k}]: {iid}")
            ex[f"{emb_prefix}{k}"] = ds[j][emb_column]
        return ex

    return brain_ds.map(_fn, desc=f"pair_brain_with_{len(emb_dss)}_embs")


def _with_transform(
    ds: datasets.Dataset, fn: Callable[[Dict[str, Any]], Dict[str, Any]]
) -> datasets.Dataset:
    """Compatibility helper across datasets versions."""
    if hasattr(ds, "with_transform"):
        return ds.with_transform(fn)
    ds.set_transform(fn)
    return ds


def build_emb_with_image_dataset(
    *,
    image_ds: datasets.Dataset,
    emb_dss: Mapping[str, datasets.Dataset],
    preprocess: Any,  # HF ImageProcessor or torchvision transform/callable
    pair_mode: PairMode = "by_index",
    verify_n: int = 64,
    emb_prefix: str = "emb_",
    emb_column: str = "emb",
) -> datasets.Dataset:
    try:
        from transformers import CLIPImageProcessor, BitImageProcessor  # type: ignore

        _HF_PROC_TYPES = (CLIPImageProcessor, BitImageProcessor)
    except Exception:
        _HF_PROC_TYPES = tuple()

    def _is_batched(x: Any) -> bool:
        return isinstance(x, (list, tuple, np.ndarray))

    def _preprocess_images(images: List[Image.Image]) -> torch.Tensor:
        # match your preferred behavior
        if _HF_PROC_TYPES and isinstance(preprocess, _HF_PROC_TYPES):
            out = preprocess(images=images, return_tensors="pt")
            return out["pixel_values"]  # [B,C,H,W]
        else:
            return torch.stack([preprocess(img) for img in images], dim=0)  # [B,C,H,W]

    def _get_emb_batch(ds: datasets.Dataset, indices: List[int]) -> torch.Tensor:
        # ds[i][emb_column] might be list/np; torch.tensor handles nested lists
        return torch.tensor([ds[i][emb_column] for i in indices], dtype=torch.float32)

    if pair_mode == "by_index":
        if "image_id" in image_ds.column_names:
            for k, ds in emb_dss.items():
                n = min(len(image_ds), len(ds), verify_n)
                for i in range(n):
                    if image_ds[i]["image_id"] != ds[i]["image_id"]:
                        raise ValueError(
                            f"by_index mismatch for model={k} at i={i}: "
                            f"{image_ds[i]['image_id']} vs {ds[i]['image_id']}. "
                            f"Use pair_mode='by_image_id'."
                        )

        n = len(image_ds)
        for _, ds in emb_dss.items():
            n = min(n, len(ds))

        idx = np.arange(n, dtype=np.int32)
        base = datasets.Dataset.from_dict(
            {"__idx__": idx},
            features=datasets.Features({"__idx__": datasets.Value("int32")}),
        )

        def _tf(ex: Dict[str, Any]) -> Dict[str, Any]:
            ids = ex["__idx__"]

            # ---- batched path ----
            if _is_batched(ids):
                indices = [int(i) for i in ids]
                images = [image_ds[i]["image"].convert("RGB") for i in indices]
                pixel_values = _preprocess_images(images)  # [B,C,H,W]
                image_ids = [image_ds[i]["image_id"] for i in indices]

                widths = [im.size[0] for im in images]
                heights = [im.size[1] for im in images]

                out: Dict[str, Any] = {
                    "pixel_values": pixel_values,
                    "image_id": image_ids,
                    "height": heights,
                    "width": widths,
                }
                for k, ds in emb_dss.items():
                    out[f"{emb_prefix}{k}"] = _get_emb_batch(ds, indices)  # [B,D]
                return out

            raise RuntimeError("Only support for batched preprocess")

        return _with_transform(base, _tf)

    image_ids = image_ds["image_id"]
    id2i_map = {
        k: {iid: i for i, iid in enumerate(ds["image_id"])} for k, ds in emb_dss.items()
    }

    valid: List[int] = []
    for i, iid in enumerate(image_ids):
        if all(iid in id2i_map[k] for k in emb_dss.keys()):
            valid.append(i)

    idx = np.asarray(valid, dtype=np.int32)
    base = datasets.Dataset.from_dict(
        {"__idx__": idx},
        features=datasets.Features({"__idx__": datasets.Value("int32")}),
    )

    def _tf(ex: Dict[str, Any]) -> Dict[str, Any]:
        ids = ex["__idx__"]

        if _is_batched(ids):
            indices = [int(i) for i in ids]
            iids = [image_ds[i]["image_id"] for i in indices]
            images = [image_ds[i]["image"].convert("RGB") for i in indices]
            pixel_values = _preprocess_images(images)  # [B,C,H,W]
            widths = [im.size[0] for im in images]
            heights = [im.size[1] for im in images]

            out: Dict[str, Any] = {
                "pixel_values": pixel_values,
                "image_id": iids,
                "height": heights,
                "width": widths,
            }
            for k, ds in emb_dss.items():
                js = [id2i_map[k][iid] for iid in iids]
                out[f"{emb_prefix}{k}"] = _get_emb_batch(ds, js)  # [B,D]
            return out

        raise RuntimeError("Only support for batched preprocess")

    return _with_transform(base, _tf)


def build_brain_with_image_dataset(
    *,
    brain_ds: datasets.Dataset,
    image_ds: datasets.Dataset,
    preprocess: Any,  # HF ImageProcessor or torchvision transform/callable
) -> datasets.Dataset:
    """Task 3: brain + image (by image_id) with batched preprocess -> pixel_values (torch.Tensor)."""
    try:
        from transformers import CLIPImageProcessor, BitImageProcessor  # type: ignore

        _HF_PROC_TYPES = (CLIPImageProcessor, BitImageProcessor)
    except Exception:
        _HF_PROC_TYPES = tuple()

    def _is_batched(x: Any) -> bool:
        return isinstance(x, (list, tuple, np.ndarray))

    def _preprocess_images(images: List[Image.Image]) -> torch.Tensor:
        if _HF_PROC_TYPES and isinstance(preprocess, _HF_PROC_TYPES):
            out = preprocess(images=images, return_tensors="pt")
            return out["pixel_values"]  # [B,C,H,W]
        else:
            return torch.stack([preprocess(img) for img in images], dim=0)  # [B,C,H,W]

    # --- build image_id -> index map ---
    image_ids = image_ds["image_id"]
    id2i = {iid: i for i, iid in enumerate(image_ids)}

    # --- keep only brain examples that have a matching image_id (same as "by_image_id" spirit) ---
    valid: List[int] = []
    for i in range(len(brain_ds)):
        iid = brain_ds[i]["image_id"]
        if iid in id2i:
            valid.append(i)

    idx = np.asarray(valid, dtype=np.int32)
    base = datasets.Dataset.from_dict(
        {"__idx__": idx},
        features=datasets.Features({"__idx__": datasets.Value("int32")}),
    )

    brain_cols = list(brain_ds.column_names)

    def _tf(ex: Dict[str, Any]) -> Dict[str, Any]:
        ids = ex["__idx__"]

        if _is_batched(ids):
            indices = [int(i) for i in ids]

            # brain fields (leave as python lists; your collate can torch.tensor them later)
            out: Dict[str, Any] = {}
            for c in brain_cols:
                out[c] = [brain_ds[i][c] for i in indices]

            # pair images by image_id and preprocess as a batch -> torch.Tensor
            iids = out["image_id"]  # list[str]
            images = [image_ds[id2i[iid]]["image"].convert("RGB") for iid in iids]
            out["pixel_values"] = _preprocess_images(images)  # [B,C,H,W] torch.Tensor

            out["height"] = [im.size[1] for im in images]
            out["width"] = [im.size[0] for im in images]

            return out

        raise RuntimeError("Only support for batched preprocess")

    return _with_transform(base, _tf)
