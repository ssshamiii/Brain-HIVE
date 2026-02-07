# cache.py
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import pyarrow as pa
import pyarrow.parquet as pq


# -----------------------------
# Shard naming helpers (align with build_embeddings.py)
# -----------------------------
_CANON_RE = re.compile(r"^(?P<base>.+)-part-(?P<i>\d{5})-of-(?P<of>\d{5})\.parquet$")
_RANK_RE = re.compile(
    r"^(?P<base>.+)\.rank-(?P<rank>\d{5})-of-(?P<world>\d{5})\.part-(?P<p>\d{6})\.parquet$"
)


def list_embedding_parquets(out_dir: str | Path, base_name: str) -> Tuple[List[Path], List[Path]]:
    """
    Return (canonical_files, rank_style_files) under out_dir for base_name.
    """
    d = Path(out_dir)
    if not d.exists():
        return [], []
    files = [p for p in d.iterdir() if p.is_file() and p.suffix == ".parquet" and p.name.startswith(base_name)]
    canon, rankish = [], []
    for p in files:
        if _CANON_RE.match(p.name):
            canon.append(p)
        elif _RANK_RE.match(p.name):
            rankish.append(p)
    canon.sort()
    rankish.sort()
    return canon, rankish


def canonical_is_complete(canon_files: Sequence[Path]) -> bool:
    """
    Canonical complete iff:
      - all share the same 'of'
      - count == (of + 1)
      - indices cover [0..of] without gaps
    """
    if not canon_files:
        return False
    idx, ofs = [], set()
    for p in canon_files:
        m = _CANON_RE.match(p.name)
        if not m:
            return False
        idx.append(int(m.group("i")))
        ofs.add(int(m.group("of")))
    if len(ofs) != 1:
        return False
    of = next(iter(ofs))
    return len(canon_files) == (of + 1) and set(idx) == set(range(of + 1))


def delete_files(paths: Iterable[str | Path]) -> None:
    for p in paths:
        try:
            Path(p).unlink()
        except FileNotFoundError:
            pass


def canonicalize_rank_shards(out_dir: str | Path, base_name: str, *, delete_existing_canon: bool = True) -> List[Path]:
    """
    Convert rank-style shards into canonical shards:
      base.rank-00000-of-00008.part-000000.parquet
    -> base-part-00000-of-000NN.parquet

    Returns the final canonical shard paths.
    """
    out_dir = Path(out_dir)
    canon, rankish = list_embedding_parquets(out_dir, base_name)

    if delete_existing_canon and canon:
        delete_files(canon)

    if not rankish:
        # maybe already canonical
        canon, _ = list_embedding_parquets(out_dir, base_name)
        return canon

    # rename rankish -> tmp -> final canonical
    tmp_paths: List[Path] = []
    for i, p in enumerate(rankish):
        tmp = out_dir / f".tmp.{base_name}.{i:05d}.parquet"
        if tmp.exists():
            tmp.unlink()
        p.rename(tmp)
        tmp_paths.append(tmp)

    n = len(tmp_paths)
    of = n - 1
    finals: List[Path] = []
    for i, tmp in enumerate(tmp_paths):
        final = out_dir / f"{base_name}-part-{i:05d}-of-{of:05d}.parquet"
        if final.exists():
            final.unlink()
        tmp.rename(final)
        finals.append(final)

    return finals


# -----------------------------
# Optional: streaming parquet writer (kept for build_embeddings.py)
# -----------------------------
def _fixed_list_type(value_type, list_size: int):
    try:
        return pa.fixed_size_list(value_type, list_size)
    except AttributeError:
        return pa.list_(value_type, list_size=list_size)


class ParquetEmbeddingWriter:
    """
    Streaming writer:
      image_id: string
      emb / emb_xxx: FixedSizeList[float16/float32]
    """
    def __init__(
        self,
        out_path: str | Path,
        *,
        dim_map: Dict[str, int],
        dtype: str = "float16",
        compression: str = "zstd",
    ):
        self.out_path = str(out_path)
        self.dim_map = dict(dim_map)
        self.dtype = dtype
        self.compression = compression
        self._writer: Optional[pq.ParquetWriter] = None

        fields = [pa.field("image_id", pa.string())]
        for key, d in self.dim_map.items():
            val_type = pa.float16() if dtype == "float16" else pa.float32()
            fields.append(pa.field(key, _fixed_list_type(val_type, d)))
        self.schema = pa.schema(fields)

    def _ensure_writer(self):
        if self._writer is None:
            os.makedirs(Path(self.out_path).parent, exist_ok=True)
            self._writer = pq.ParquetWriter(self.out_path, self.schema, compression=self.compression)

    def write(self, rows: Dict[str, object]) -> None:
        """
        rows: {"image_id": List[str], "emb": torch.Tensor[B,D], ...}
        """
        self._ensure_writer()
        data = {"image_id": pa.array(rows["image_id"], type=pa.string())}

        for key, d in self.dim_map.items():
            x: torch.Tensor = rows[key]  # type: ignore[assignment]
            if not (isinstance(x, torch.Tensor) and x.ndim == 2 and x.shape[1] == d):
                raise ValueError(f"{key}: expected Tensor[B,{d}], got {type(x)} {getattr(x,'shape',None)}")

            x = x.detach().cpu()
            if self.dtype == "float16":
                x = x.to(torch.float16)
                val_type = pa.float16()
            else:
                x = x.to(torch.float32)
                val_type = pa.float32()

            flat = x.numpy().reshape(-1)
            values = pa.array(flat, type=val_type)
            arr = pa.FixedSizeListArray.from_arrays(values, list_size=d)
            data[key] = arr

        table = pa.Table.from_pydict(data, schema=self.schema)
        assert self._writer is not None
        self._writer.write_table(table)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None


class RollingParquetEmbeddingWriter:
    """
    Rotate files at max_rows_per_file (rank-style naming).
    """
    def __init__(
        self,
        out_dir: str | Path,
        *,
        base_name: str,
        rank: int,
        world: int,
        dim_map: Dict[str, int],
        dtype: str = "float16",
        compression: str = "zstd",
        max_rows_per_file: int = 200_000,
    ):
        self.out_dir = str(out_dir)
        self.base_name = base_name
        self.rank = int(rank)
        self.world = int(world)
        self.dim_map = dict(dim_map)
        self.dtype = dtype
        self.compression = compression
        self.max_rows_per_file = int(max_rows_per_file)

        self._part = 0
        self._rows_in_file = 0
        self._writer: Optional[ParquetEmbeddingWriter] = None

    def _new_file(self) -> None:
        if self._writer is not None:
            self._writer.close()
        os.makedirs(self.out_dir, exist_ok=True)
        out_path = os.path.join(
            self.out_dir,
            f"{self.base_name}.rank-{self.rank:05d}-of-{self.world:05d}.part-{self._part:06d}.parquet",
        )
        self._writer = ParquetEmbeddingWriter(
            out_path,
            dim_map=self.dim_map,
            dtype=self.dtype,
            compression=self.compression,
        )
        self._rows_in_file = 0
        self._part += 1

    def write(self, rows: Dict[str, object]) -> None:
        n = len(rows["image_id"])  # type: ignore[arg-type]
        if self._writer is None:
            self._new_file()
        if self._rows_in_file + n > self.max_rows_per_file:
            self._new_file()
        assert self._writer is not None
        self._writer.write(rows)
        self._rows_in_file += n

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
