#!/usr/bin/env python3
r"""
DHA PaiNN / SchNetPack decomposition probe v7
==========================================

Purpose
-------
Reproduce the MLFF side of Kabylda et al. "How Atoms Interact Within Molecules"
more faithfully than the transparent toy MLFF harness:

  DHA via Hugging Face/ColabFit -> local .npz cache -> SchNetPack ASE DB
  -> actual SchNetPack PaiNN -> per-atom energy decomposition
  -> F_ij = -dE_j/dR_i -> interaction-depth / anisotropy diagnostics
  -> autograd when available, finite-difference fallback when SchNetPack detaches the attribution graph
  -> reconstruction diagnostics for the finite-difference attribution
  -> zero-sum attribution-gauge tests -> optional total-force response witness.

Why this file exists
--------------------
The paper's MLFF decomposition defines atom j's contribution to the force on atom i as
F_ij = -dE_j/dR_i, where E_j is the learned per-atom energy contribution. SchNetPack's
Atomwise block can expose those per-atom contributions via per_atom_output_key, which
lets us probe exactly that attribution layer instead of using a hand-rolled stand-in.

Install
-------
Recommended fresh venv on Windows/PowerShell:

  py -3.12 -m venv .venv
  .\.venv\Scripts\Activate.ps1
  python -m pip install --upgrade pip
  pip install torch numpy scipy pandas pyarrow datasets huggingface_hub matplotlib tqdm ase torchmetrics pytorch-lightning schnetpack

Smoke test:

  python dha_painn_schnetpack_probe_v7_adil_response.py --nframes 500 --max-epochs 2 --analysis-frames 2 --response-frames 0 --device cuda --force-train

Paper-ish first pass:

  python dha_painn_schnetpack_probe_v7_adil_response.py --nframes 2500 --max-epochs 50 --analysis-frames 24 --response-frames 2 --device cuda --force-train

Notes
-----
* Defaults match the paper's reported DHA PaiNN settings as closely as practical:
  n_atom_basis=128, n_rbf=30, n_interactions=3, cutoff=10 A, lr=5e-4, weight_decay=1e-3,
  num_train=950, num_val=50.
* This still replicates the public-data MLFF side, not the private/expensive SQ-MBD
  pairwise finite-difference decomposition.
* The R^-7 baseline is handled in log space to avoid overflow warnings.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import random
import shutil
import sys
import time
import platform
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Keep Windows multiprocessing from becoming an unreadable wall of harmless CUDA monitor warnings.
warnings.filterwarnings("ignore", message=".*pynvml package is deprecated.*", category=FutureWarning)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np

try:
    import torch
except Exception as exc:  # pragma: no cover
    raise SystemExit("PyTorch is required. Install with: pip install torch") from exc

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise SystemExit("matplotlib is required. Install with: pip install matplotlib") from exc

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x


ENERGY_KEY = "energy"
FORCES_KEY = "forces"
PER_ATOM_KEY = "per_atom_energy"
EPS = 1e-12


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_dump(obj: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)


# -----------------------------------------------------------------------------
# Public data -> local cache
# -----------------------------------------------------------------------------

def _is_num_array(x: Any, min_ndim: int = 1) -> bool:
    try:
        arr = np.asarray(x)
        return arr.ndim >= min_ndim and np.issubdtype(arr.dtype, np.number)
    except Exception:
        return False


def _candidate_keys(row: Dict[str, Any]) -> Dict[str, str]:
    """Heuristic key detector for Hugging Face / ColabFit rows."""
    keys = list(row.keys())
    low = {k.lower(): k for k in keys}

    def find_by_names(names: Sequence[str]) -> Optional[str]:
        for name in names:
            if name.lower() in low:
                return low[name.lower()]
        for k in keys:
            kl = k.lower()
            if any(name.lower() in kl for name in names):
                return k
        return None

    energy_key = find_by_names(["energy", "potential_energy", "total_energy"])
    forces_key = find_by_names(["atomic_forces", "forces", "force"])
    positions_key = find_by_names(["positions", "coordinates", "coords", "cartesian_positions", "cartesian_coordinates"])
    z_key = find_by_names(["atomic_numbers", "numbers", "atomic_number", "z"])

    # Fallback scan by shape.
    if positions_key is None or forces_key is None or z_key is None:
        arrays = []
        for k, v in row.items():
            if _is_num_array(v):
                arr = np.asarray(v)
                arrays.append((k, arr.shape, arr.dtype, arr))
        nx3 = [(k, arr) for k, shape, dtype, arr in arrays if arr.ndim == 2 and arr.shape[-1] == 3]
        if positions_key is None and len(nx3) >= 1:
            positions_key = nx3[0][0]
        if forces_key is None and len(nx3) >= 2:
            forces_key = nx3[1][0]
        one_d_int = [(k, arr) for k, shape, dtype, arr in arrays if arr.ndim == 1 and np.issubdtype(arr.dtype, np.integer)]
        if z_key is None and one_d_int:
            z_key = one_d_int[0][0]

    missing = [name for name, val in {"energy": energy_key, "forces": forces_key, "positions": positions_key, "atomic_numbers": z_key}.items() if val is None]
    if missing:
        preview = {k: str(type(v))[:80] for k, v in list(row.items())[:40]}
        raise RuntimeError(f"Could not infer keys for {missing}. Row keys: {list(row.keys())}. Preview: {preview}")
    return {"energy": energy_key, "forces": forces_key, "positions": positions_key, "atomic_numbers": z_key}


def _row_to_arrays(row: Dict[str, Any], keymap: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    R = np.asarray(row[keymap["positions"]], dtype=np.float64)
    F = np.asarray(row[keymap["forces"]], dtype=np.float64)
    E = float(np.asarray(row[keymap["energy"]]).reshape(-1)[0])
    Z = np.asarray(row[keymap["atomic_numbers"]], dtype=np.int64)
    if R.ndim != 2 or R.shape[-1] != 3:
        raise ValueError(f"Bad positions shape: {R.shape}")
    if F.shape != R.shape:
        raise ValueError(f"Forces shape {F.shape} does not match positions {R.shape}")
    if Z.ndim != 1 or Z.shape[0] != R.shape[0]:
        raise ValueError(f"Atomic numbers shape {Z.shape} incompatible with positions {R.shape}")
    return R, F, E, Z


def load_hf_colabfit(repo_id: str, nframes: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load ColabFit MD22_DHA from Hugging Face using datasets first, parquet snapshot second."""
    try:
        from datasets import load_dataset  # type: ignore
        ds = load_dataset(repo_id, split="train")
        n_available = len(ds)
        rng = np.random.default_rng(seed)
        if 0 < nframes < n_available:
            idx = np.sort(rng.choice(n_available, size=nframes, replace=False))
            rows = [ds[int(i)] for i in idx]
        else:
            rows = [ds[int(i)] for i in range(n_available)]
        keymap = _candidate_keys(rows[0])
        Rs, Fs, Es = [], [], []
        Z_ref = None
        for row in tqdm(rows, desc="HF rows -> arrays"):
            R, F, E, Z = _row_to_arrays(row, keymap)
            Rs.append(R); Fs.append(F); Es.append(E)
            if Z_ref is None:
                Z_ref = Z
            elif not np.array_equal(Z_ref, Z):
                raise RuntimeError("Atomic numbers changed across frames; expected fixed DHA composition.")
        meta = {"source": "huggingface_datasets", "repo_id": repo_id, "keymap": keymap, "n_available": n_available}
        return np.stack(Rs), np.stack(Fs), np.asarray(Es), np.asarray(Z_ref), meta
    except Exception as first_exc:
        print(f"[WARN] datasets.load_dataset failed: {first_exc!r}")

    try:
        import pandas as pd  # type: ignore
        from huggingface_hub import snapshot_download  # type: ignore
        snap = Path(snapshot_download(repo_id=repo_id, repo_type="dataset"))
        parquet_files = sorted(snap.rglob("*.parquet"))
        if not parquet_files:
            raise RuntimeError(f"No parquet files found in {snap}")
        df = pd.concat([pd.read_parquet(p) for p in parquet_files], ignore_index=True)
        n_available = len(df)
        rng = np.random.default_rng(seed)
        if 0 < nframes < n_available:
            df = df.iloc[np.sort(rng.choice(n_available, size=nframes, replace=False))]
        rows = df.to_dict("records")
        keymap = _candidate_keys(rows[0])
        Rs, Fs, Es = [], [], []
        Z_ref = None
        for row in tqdm(rows, desc="Parquet rows -> arrays"):
            R, F, E, Z = _row_to_arrays(row, keymap)
            Rs.append(R); Fs.append(F); Es.append(E)
            if Z_ref is None:
                Z_ref = Z
            elif not np.array_equal(Z_ref, Z):
                raise RuntimeError("Atomic numbers changed across frames; expected fixed DHA composition.")
        meta = {"source": "huggingface_snapshot_parquet", "repo_id": repo_id, "keymap": keymap, "n_available": n_available, "snapshot": str(snap)}
        return np.stack(Rs), np.stack(Fs), np.asarray(Es), np.asarray(Z_ref), meta
    except Exception as second_exc:
        raise RuntimeError(f"Could not load {repo_id} via datasets or parquet. Last error: {second_exc!r}") from second_exc


def load_or_build_npz_cache(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    cache_dir = ensure_dir(Path(args.cache_dir))
    npz_path = cache_dir / args.npz_name
    meta_path = cache_dir / (Path(args.npz_name).stem + ".metadata.json")

    if npz_path.exists() and not args.refresh_cache:
        data = np.load(npz_path, allow_pickle=True)
        R = data["R"].astype(np.float64)
        F = data["F"].astype(np.float64)
        E = data["E"].astype(np.float64)
        Z = data["Z"].astype(np.int64)
        meta = json.loads(str(data["meta_json"])) if "meta_json" in data.files else {}
        print(f"[CACHE] Loaded {npz_path}")
        return R, F, E, Z, meta

    print("[DATA] Cache missing or refresh requested. Pulling public DHA data...")
    R, F, E, Z, meta = load_hf_colabfit(args.hf_repo, args.nframes, args.seed)

    # Optional unit conversion hook. Default is as-is because ColabFit metadata can vary.
    if args.energy_scale != 1.0 or args.force_scale != 1.0:
        E = E * float(args.energy_scale)
        F = F * float(args.force_scale)
        meta["applied_energy_scale"] = args.energy_scale
        meta["applied_force_scale"] = args.force_scale

    np.savez_compressed(npz_path, R=R, F=F, E=E, Z=Z, meta_json=json.dumps(meta))
    json_dump(meta, meta_path)
    print(f"[CACHE] Saved {npz_path}")
    return R, F, E, Z, meta


# -----------------------------------------------------------------------------
# SchNetPack DB + model
# -----------------------------------------------------------------------------



def patch_ase_sqlite_metadata_for_new_ase() -> None:
    """
    Compatibility shim for newer ASE builds whose SQLite DB metadata property
    asserts that the DB connection is already open. SchNetPack's ASEAtomsData
    accesses conn.metadata on a persistent ASE connection object, which worked
    on older ASE releases but can trip `AssertionError: self.connection is not None`
    on Windows/newer ASE.

    The shim preserves ASE's original metadata getter/setter and simply opens a
    short context when SchNetPack asks for metadata while the connection is idle.
    """
    try:
        from ase.db.sqlite import SQLite3Database  # type: ignore
    except Exception:
        return

    if getattr(SQLite3Database, "_ghost_oracle_metadata_patch", False):
        return

    prop = getattr(SQLite3Database, "metadata", None)
    if not isinstance(prop, property) or prop.fget is None:
        return

    orig_get = prop.fget
    orig_set = prop.fset
    orig_del = prop.fdel

    def metadata_get(self):  # type: ignore[no-untyped-def]
        if getattr(self, "connection", None) is not None:
            return orig_get(self)
        with self:
            return orig_get(self)

    def metadata_set(self, value):  # type: ignore[no-untyped-def]
        if orig_set is None:
            raise AttributeError("metadata is read-only")
        if getattr(self, "connection", None) is not None:
            return orig_set(self, value)
        with self:
            return orig_set(self, value)

    SQLite3Database.metadata = property(metadata_get, metadata_set if orig_set else None, orig_del, prop.__doc__)
    SQLite3Database._ghost_oracle_metadata_patch = True


def ase_db_looks_complete(db_path: Path, expected_rows: int) -> bool:
    """Return True only when the cached ASE DB has metadata and all expected frames."""
    if not db_path.exists():
        return False
    try:
        patch_ase_sqlite_metadata_for_new_ase()
        from ase.db import connect  # type: ignore
        with connect(str(db_path), use_lock_file=False) as conn:
            md = conn.metadata
            if "_distance_unit" not in md or "_property_unit_dict" not in md:
                return False
            units = md.get("_property_unit_dict", {})
            if ENERGY_KEY not in units or FORCES_KEY not in units:
                return False
            return int(conn.count()) == int(expected_rows)
    except Exception:
        return False

def import_schnetpack_stack():
    patch_ase_sqlite_metadata_for_new_ase()
    try:
        import schnetpack as spk  # type: ignore
        import schnetpack.transform as trn  # type: ignore
        import torchmetrics  # type: ignore
        from ase import Atoms  # type: ignore
        from schnetpack.data import ASEAtomsData, AtomsDataModule  # type: ignore
        try:
            import pytorch_lightning as pl  # type: ignore
        except Exception:
            import lightning.pytorch as pl  # type: ignore
        return spk, trn, torchmetrics, Atoms, ASEAtomsData, AtomsDataModule, pl
    except Exception as exc:
        raise SystemExit(
            "SchNetPack stack is required for this probe. Install with:\n"
            "  pip install schnetpack ase torchmetrics pytorch-lightning\n"
            f"Original import error: {exc!r}"
        ) from exc


def build_ase_db(args: argparse.Namespace, R: np.ndarray, F: np.ndarray, E: np.ndarray, Z: np.ndarray) -> Path:
    spk, trn, torchmetrics, Atoms, ASEAtomsData, AtomsDataModule, pl = import_schnetpack_stack()
    db_dir = ensure_dir(Path(args.cache_dir))
    db_path = (db_dir / args.db_name).resolve()

    if db_path.exists() and not args.rebuild_db:
        if ase_db_looks_complete(db_path, len(E)):
            print(f"[DB] Using existing complete ASE DB: {db_path}")
            return db_path
        print(f"[DB] Existing ASE DB is incomplete/stale; rebuilding: {db_path}")

    if db_path.exists():
        print(f"[DB] Removing old ASE DB: {db_path}")
        db_path.unlink()

    print(f"[DB] Building ASE DB: {db_path}")
    dataset = ASEAtomsData.create(
        str(db_path),
        distance_unit="Ang",
        property_unit_dict={ENERGY_KEY: args.energy_unit, FORCES_KEY: args.force_unit},
    )

    atoms_list = []
    property_list = []
    for pos, energy, forces in tqdm(zip(R, E, F), total=len(E), desc="ASE DB systems"):
        atoms_list.append(Atoms(numbers=Z, positions=pos))
        property_list.append({
            ENERGY_KEY: np.asarray([energy], dtype=np.float64),
            FORCES_KEY: np.asarray(forces, dtype=np.float64),
        })
    dataset.add_systems(property_list, atoms_list)
    print(f"[DB] Wrote {len(E)} frames")
    return db_path


def build_datamodule(args: argparse.Namespace, db_path: Path):
    spk, trn, torchmetrics, Atoms, ASEAtomsData, AtomsDataModule, pl = import_schnetpack_stack()
    split_path = Path(args.cache_dir) / args.split_name
    transforms = [
        trn.ASENeighborList(cutoff=args.cutoff),
        trn.RemoveOffsets(ENERGY_KEY, remove_mean=True, remove_atomrefs=False),
        trn.CastTo32(),
    ]
    dm_kwargs = dict(
        datapath=str(db_path),
        batch_size=args.batch_size,
        num_train=args.num_train,
        num_val=args.num_val,
        num_test=args.num_test,
        split_file=str(split_path),
        load_properties=[ENERGY_KEY, FORCES_KEY],
        transforms=transforms,
        num_workers=args.num_workers,
        persistent_workers=bool(args.persistent_workers and args.num_workers > 0),
        pin_memory=(args.device.startswith("cuda") and torch.cuda.is_available()),
        property_units={ENERGY_KEY: args.energy_unit, FORCES_KEY: args.force_unit},
    )
    try:
        dm = AtomsDataModule(**dm_kwargs)
    except TypeError as exc:
        # SchNetPack minor versions have moved/renamed a few datamodule kwargs.
        # Retry with the most portable core argument set.
        print(f"[WARN] AtomsDataModule rejected one optional kwarg: {exc}")
        for optional in ["property_units", "pin_memory", "num_test", "persistent_workers"]:
            dm_kwargs.pop(optional, None)
        dm = AtomsDataModule(**dm_kwargs)
    return dm


def build_painn_model(args: argparse.Namespace):
    spk, trn, torchmetrics, Atoms, ASEAtomsData, AtomsDataModule, pl = import_schnetpack_stack()

    pairwise_distance = spk.atomistic.PairwiseDistances()
    radial_basis = spk.nn.GaussianRBF(n_rbf=args.n_rbf, cutoff=args.cutoff)
    cutoff_fn = spk.nn.CosineCutoff(args.cutoff)

    # Actual SchNetPack PaiNN, not a stand-in.
    painn = spk.representation.PaiNN(
        n_atom_basis=args.n_atom_basis,
        n_interactions=args.n_interactions,
        radial_basis=radial_basis,
        cutoff_fn=cutoff_fn,
    )

    pred_energy = spk.atomistic.Atomwise(
        n_in=args.n_atom_basis,
        output_key=ENERGY_KEY,
        per_atom_output_key=PER_ATOM_KEY,
        aggregation_mode="sum",
    )
    pred_forces = spk.atomistic.Forces(energy_key=ENERGY_KEY, force_key=FORCES_KEY)

    nnpot = spk.model.NeuralNetworkPotential(
        representation=painn,
        input_modules=[pairwise_distance],
        output_modules=[pred_energy, pred_forces],
        postprocessors=[
            trn.CastTo64(),
            trn.AddOffsets(ENERGY_KEY, add_mean=True, add_atomrefs=False),
        ],
    )
    return nnpot


def find_latest_inference_model(run_root: str) -> Optional[Path]:
    """Find the newest SchNetPack best_inference_model under run_root."""
    root = Path(run_root)
    if not root.exists():
        return None
    candidates = []
    for path in root.glob("dha_painn_probe_*/best_inference_model"):
        if path.exists():
            try:
                candidates.append((path.stat().st_mtime, path))
            except OSError:
                pass
    if not candidates:
        return None
    return sorted(candidates, key=lambda x: x[0])[-1][1]


def train_or_load_model(args: argparse.Namespace, dm, run_dir: Path):
    spk, trn, torchmetrics, Atoms, ASEAtomsData, AtomsDataModule, pl = import_schnetpack_stack()
    from schnetpack.utils.compatibility import load_model  # type: ignore

    model_path = run_dir / "best_inference_model"
    if args.model_path:
        print(f"[MODEL] Loading explicit model: {args.model_path}")
        return load_model(args.model_path, device=args.device), Path(args.model_path)

    # Default behavior: do not retrain if a usable model already exists. This is important
    # because analysis bugs should not force another 50-epoch training run.
    if not args.force_train:
        latest = find_latest_inference_model(args.run_root)
        if latest is not None:
            print(f"[MODEL] Auto-loading latest existing inference model: {latest}")
            print("[MODEL] Use --force-train to train a fresh PaiNN model.")
            return load_model(str(latest), device=args.device), latest

    if args.skip_train:
        raise RuntimeError("--skip-train was set, but no --model-path was provided and no existing inference model was found.")
    if dm is None:
        raise RuntimeError("Training requested but datamodule is None. This is an internal routing error.")

    nnpot = build_painn_model(args)
    output_energy = spk.task.ModelOutput(
        name=ENERGY_KEY,
        loss_fn=torch.nn.MSELoss(),
        loss_weight=args.energy_loss_weight,
        metrics={"MAE": torchmetrics.MeanAbsoluteError()},
    )
    output_forces = spk.task.ModelOutput(
        name=FORCES_KEY,
        loss_fn=torch.nn.MSELoss(),
        loss_weight=args.force_loss_weight,
        metrics={"MAE": torchmetrics.MeanAbsoluteError()},
    )

    task = spk.task.AtomisticTask(
        model=nnpot,
        outputs=[output_energy, output_forces],
        optimizer_cls=torch.optim.AdamW,
        optimizer_args={"lr": args.lr, "weight_decay": args.weight_decay},
    )

    # CSVLogger avoids TensorBoard/TensorFlow startup spam and is plenty for this probe.
    try:
        logger = pl.loggers.CSVLogger(save_dir=str(run_dir), name="lightning_logs")
    except Exception:
        logger = False
    callbacks = [
        spk.train.ModelCheckpoint(
            model_path=str(model_path),
            save_top_k=1,
            monitor="val_loss",
        )
    ]

    accelerator = "gpu" if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    trainer_kwargs = dict(
        callbacks=callbacks,
        logger=logger,
        default_root_dir=str(run_dir),
        max_epochs=args.max_epochs,
        accelerator=accelerator,
        devices=1,
        log_every_n_steps=args.log_every_n_steps,
        num_sanity_val_steps=0,
        gradient_clip_val=args.grad_clip,
    )
    # Lightning version compatibility: some older installs dislike enable_checkpointing here.
    try:
        trainer = pl.Trainer(**trainer_kwargs)
    except TypeError:
        trainer_kwargs.pop("gradient_clip_val", None)
        trainer = pl.Trainer(**trainer_kwargs)

    print("[TRAIN] Starting SchNetPack PaiNN training")
    trainer.fit(task, datamodule=dm)
    print(f"[TRAIN] Best inference model expected at: {model_path}")

    if not model_path.exists():
        print("[WARN] best_inference_model not found. Trying to use task.model directly.")
        model = task.model.to(args.device)
        return model, model_path

    model = load_model(str(model_path), device=args.device)
    return model, model_path




def _find_existing_key(inputs: Dict[str, Any], candidates: Sequence[str]) -> Optional[str]:
    """Return the first candidate key that exists in a SchNetPack input dict."""
    for key in candidates:
        if key in inputs:
            return key
    return None


def _property_candidates(name: str, fallbacks: Sequence[str]) -> List[str]:
    """Build robust SchNetPack property-key candidates across minor versions."""
    keys: List[str] = []
    try:
        spk, *_ = import_schnetpack_stack()
        props = spk.properties
        if hasattr(props, name):
            val = getattr(props, name)
            if isinstance(val, str):
                keys.append(val)
    except Exception:
        pass
    for val in fallbacks:
        if val not in keys:
            keys.append(val)
    return keys


def reconnect_pairwise_geometry(inputs: Dict[str, Any], pos_key: str) -> None:
    """Rebuild pair vectors/distances from the gradient-enabled positions if present.

    Some SchNetPack/ASE converter versions precompute pairwise geometry before we replace
    positions with a leaf tensor. If the model consumes a stale _Rij/_distances tensor,
    per_atom_energy is not connected to the new position tensor and autograd reports
    "tensor was not used in the graph". Recomputing these fields makes the chain
    position -> pair geometry -> PaiNN -> per_atom_energy explicit again.
    """
    if pos_key not in inputs or not torch.is_tensor(inputs[pos_key]):
        return
    pos = inputs[pos_key]
    idx_i_key = _find_existing_key(inputs, _property_candidates("idx_i", ["_idx_i", "idx_i"]))
    idx_j_key = _find_existing_key(inputs, _property_candidates("idx_j", ["_idx_j", "idx_j"]))
    if idx_i_key is None or idx_j_key is None:
        return
    idx_i = inputs[idx_i_key].long()
    idx_j = inputs[idx_j_key].long()

    rij = pos[idx_j] - pos[idx_i]

    # Periodic offsets are normally zero for these molecular frames, but keep the path robust.
    cell_key = _find_existing_key(inputs, _property_candidates("cell", ["_cell", "cell"]))
    offset_key = _find_existing_key(
        inputs,
        _property_candidates("offsets", ["_offsets", "offsets", "_cell_offset", "cell_offset", "_cell_offsets", "cell_offsets"]),
    )
    if cell_key is not None and offset_key is not None:
        cell = inputs[cell_key]
        offsets = inputs[offset_key]
        if torch.is_tensor(cell) and torch.is_tensor(offsets) and offsets.ndim >= 2 and cell.ndim >= 2:
            try:
                if cell.ndim == 3:
                    cell0 = cell[0]
                else:
                    cell0 = cell
                rij = rij + offsets.to(pos.dtype).to(pos.device) @ cell0.to(pos.dtype).to(pos.device)
            except Exception:
                # Non-periodic molecules do not need this; silently keep the non-periodic vector.
                pass

    rij_keys = _property_candidates("Rij", ["_Rij", "Rij", "rij"])
    for key in rij_keys:
        if key in inputs:
            inputs[key] = rij
            break
    else:
        # Harmless even if unused by this version; several SchNetPack builds look for _Rij.
        inputs["_Rij"] = rij

    dist = torch.linalg.norm(rij, dim=-1)
    dist_keys = _property_candidates("distances", ["_distances", "distances", "_distance", "distance"])
    for key in dist_keys:
        if key in inputs:
            inputs[key] = dist
            break


@contextlib.contextmanager
def without_force_output_modules(model):
    """Temporarily remove SchNetPack Forces output modules during custom autograd analysis.

    The trained model includes a Forces output module so SchNetPack can train against force labels.
    For our decomposition, we want to run the energy/per-atom head only and take gradients ourselves.

    Important SchNetPack detail: removing the Forces module alone is not enough in some
    versions because NeuralNetworkPotential caches `model_outputs` during construction and
    will still try to extract `forces` from the input/output dict. We therefore patch both
    output_modules and model_outputs while inside this context.
    """
    if not hasattr(model, "output_modules"):
        yield model
        return

    old_modules = model.output_modules
    old_model_outputs = getattr(model, "model_outputs", None)
    old_required_derivatives = getattr(model, "required_derivatives", None)
    old_required_outputs = getattr(model, "required_outputs", None)

    def _drop_force_key(seq):
        if seq is None:
            return None
        try:
            return type(seq)([x for x in list(seq) if str(x) != FORCES_KEY])
        except Exception:
            return [x for x in list(seq) if str(x) != FORCES_KEY]

    try:
        kept = []
        for module in old_modules:
            cls_name = module.__class__.__name__.lower()
            if cls_name == "forces" or cls_name.endswith("forces"):
                continue
            kept.append(module)
        if len(kept) != len(old_modules):
            model.output_modules = torch.nn.ModuleList(kept)

        # Patch cached output lists so extract_outputs does not ask for `forces`.
        for attr, old_val in [
            ("model_outputs", old_model_outputs),
            ("required_outputs", old_required_outputs),
            ("required_derivatives", old_required_derivatives),
        ]:
            if old_val is not None:
                try:
                    setattr(model, attr, _drop_force_key(old_val))
                except Exception:
                    pass
        yield model
    finally:
        try:
            model.output_modules = old_modules
        except Exception:
            pass
        for attr, old_val in [
            ("model_outputs", old_model_outputs),
            ("required_outputs", old_required_outputs),
            ("required_derivatives", old_required_derivatives),
        ]:
            if old_val is not None:
                try:
                    setattr(model, attr, old_val)
                except Exception:
                    pass


# -----------------------------------------------------------------------------
# SchNetPack inference helpers
# -----------------------------------------------------------------------------

def make_converter(args: argparse.Namespace):
    spk, trn, torchmetrics, Atoms, ASEAtomsData, AtomsDataModule, pl = import_schnetpack_stack()
    return spk.interfaces.AtomsConverter(
        neighbor_list=trn.ASENeighborList(cutoff=args.cutoff),
        dtype=torch.float32,
        device=args.device,
    )


def get_position_key() -> str:
    spk, *_ = import_schnetpack_stack()
    return spk.properties.R


def atoms_from_arrays(R: np.ndarray, Z: np.ndarray):
    spk, trn, torchmetrics, Atoms, ASEAtomsData, AtomsDataModule, pl = import_schnetpack_stack()
    return Atoms(numbers=Z, positions=R)


def forward_single(
    model,
    converter,
    R: np.ndarray,
    Z: np.ndarray,
    device: str,
    require_pos_grad: bool = True,
    energy_only: bool = False,
):
    atoms = atoms_from_arrays(R, Z)
    inputs = converter(atoms)
    pos_key = get_position_key()

    # Move everything first, then replace positions with a gradient-enabled leaf tensor.
    for k, v in list(inputs.items()):
        if torch.is_tensor(v):
            inputs[k] = v.to(device)
    inputs[pos_key] = inputs[pos_key].detach().clone().to(device).requires_grad_(require_pos_grad)

    # Critical compatibility fix: rebuild any precomputed pairwise geometry so it is connected
    # to the new position tensor. Without this, some SchNetPack versions report that positions
    # were not used in the graph when differentiating per_atom_energy.
    if require_pos_grad:
        reconnect_pairwise_geometry(inputs, pos_key)

    model = model.to(device)
    model.train()  # no dropout in default PaiNN; keeps derivative path permissive.
    with torch.enable_grad():
        if energy_only:
            with without_force_output_modules(model):
                out = model(inputs)
        else:
            out = model(inputs)
    return inputs, out



def predict_energy_forces(model, converter, R: np.ndarray, Z: np.ndarray, args: argparse.Namespace) -> Dict[str, Any]:
    """Predict total energy and total forces for one frame using the trained SchNetPack model."""
    inputs, out = forward_single(model, converter, R, Z, args.device, require_pos_grad=True, energy_only=False)
    if ENERGY_KEY not in out:
        raise RuntimeError(f"Model did not return {ENERGY_KEY!r}. Available keys: {list(out.keys())}")
    if FORCES_KEY not in out:
        raise RuntimeError(
            f"Model did not return {FORCES_KEY!r}. Available keys: {list(out.keys())}. "
            "This usually means the loaded inference model was saved without the Forces output module."
        )
    energy = float(out[ENERGY_KEY].reshape(-1)[0].detach().cpu().item())
    forces = out[FORCES_KEY].detach().cpu().numpy()
    if forces.ndim == 3 and forces.shape[0] == 1:
        forces = forces[0]
    return {"energy": energy, "forces": np.asarray(forces, dtype=np.float64)}


def predict_per_atom_energy(model, converter, R: np.ndarray, Z: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """Return per-atom energy contributions for one frame with Forces output disabled."""
    with torch.enable_grad():
        inputs, out = forward_single(model, converter, R, Z, args.device, require_pos_grad=False, energy_only=True)
    if PER_ATOM_KEY not in out:
        raise RuntimeError(
            f"Model output does not contain {PER_ATOM_KEY!r}. "
            f"Available keys: {list(out.keys())}"
        )
    return out[PER_ATOM_KEY].reshape(-1).detach().cpu().numpy().astype(np.float64)


def compute_per_atom_fij_autograd(model, converter, R: np.ndarray, Z: np.ndarray, args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    """Try true autograd F_ij = -dE_j/dR_i. Return None if this SchNetPack build detaches it."""
    inputs, out = forward_single(model, converter, R, Z, args.device, require_pos_grad=True, energy_only=True)
    pos = inputs[get_position_key()]

    if PER_ATOM_KEY not in out:
        raise RuntimeError(
            f"Model output does not contain {PER_ATOM_KEY!r}. "
            "Check that Atomwise(per_atom_output_key=PER_ATOM_KEY) is active and survives model extraction. "
            f"Available keys: {list(out.keys())}"
        )
    per_atom = out[PER_ATOM_KEY].reshape(-1)
    n = int(per_atom.shape[0])
    if n != R.shape[0]:
        raise RuntimeError(f"per_atom_energy has length {n}, expected {R.shape[0]}")

    Fij = torch.empty((n, n, 3), dtype=pos.dtype, device=pos.device)
    unused_count = 0
    for j in range(n):
        grad_j = torch.autograd.grad(
            per_atom[j],
            pos,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )[0]
        if grad_j is None:
            unused_count += 1
            grad_j = torch.zeros_like(pos)
        Fij[:, j, :] = -grad_j

    if unused_count == n:
        return None

    energy = out[ENERGY_KEY].reshape(-1)[0] if ENERGY_KEY in out else per_atom.sum()
    pred_force = Fij.sum(dim=1).detach()
    return {
        "Fij": Fij.detach().cpu().numpy(),
        "per_atom_energy": per_atom.detach().cpu().numpy(),
        "energy": float(energy.detach().cpu().item()),
        "forces": pred_force.detach().cpu().numpy(),
        "unused_per_atom_grads": int(unused_count),
        "fij_method_used": "autograd",
    }


def compute_per_atom_fij_finite_difference(model, converter, R: np.ndarray, Z: np.ndarray, args: argparse.Namespace) -> Dict[str, Any]:
    """Finite-difference F_ij = -dE_j/dR_i for builds where per_atom_energy is detached.

    This is slower than autograd but much more robust across SchNetPack minor versions. It is
    also conceptually aligned with the paper's own finite-difference SQ-MBD force decomposition,
    though here the differentiated object is the PaiNN per-atom energy attribution.
    """
    n = int(R.shape[0])
    delta = float(args.fij_fd_delta)
    per0 = predict_per_atom_energy(model, converter, R, Z, args)
    Fij = np.empty((n, n, 3), dtype=np.float64)

    iterator = range(n * 3)
    if bool(getattr(args, "show_fd_progress", False)):
        iterator = tqdm(iterator, desc="finite-diff Fij coords", leave=False)

    for flat in iterator:
        i = flat // 3
        a = flat % 3
        Rp = np.array(R, copy=True)
        Rm = np.array(R, copy=True)
        Rp[i, a] += delta
        Rm[i, a] -= delta
        ep = predict_per_atom_energy(model, converter, Rp, Z, args)
        em = predict_per_atom_energy(model, converter, Rm, Z, args)
        # dE_j/dR_i,a for all j, then negative gradient gives force contribution.
        Fij[i, :, a] = -(ep - em) / (2.0 * delta)

    total = predict_energy_forces(model, converter, R, Z, args)
    return {
        "Fij": Fij,
        "per_atom_energy": per0,
        "energy": total["energy"],
        "forces": total["forces"],
        "unused_per_atom_grads": n,
        "fij_method_used": "finite_difference",
    }


def compute_per_atom_fij(model, converter, R: np.ndarray, Z: np.ndarray, args: argparse.Namespace) -> Dict[str, Any]:
    """Compute F_ij = -dE_j/dR_i for one frame using autograd or finite differences."""
    method = str(getattr(args, "fij_method", "auto")).lower()
    if method not in {"auto", "autograd", "finite-diff", "finite_difference", "fd"}:
        raise ValueError(f"Unknown --fij-method {method!r}")

    if method in {"auto", "autograd"}:
        pred = compute_per_atom_fij_autograd(model, converter, R, Z, args)
        if pred is not None:
            return pred
        if method == "autograd":
            raise RuntimeError(
                "Autograd Fij failed because per_atom_energy is disconnected from positions in this SchNetPack build. "
                "Rerun with --fij-method finite-diff or --fij-method auto."
            )
        if bool(getattr(args, "verbose_fallback", True)):
            print("[Fij] per_atom_energy is disconnected from positions; using finite-difference fallback for this run.")

    return compute_per_atom_fij_finite_difference(model, converter, R, Z, args)


def compute_force_response_witness_autograd(model, converter, R: np.ndarray, Z: np.ndarray, args: argparse.Namespace) -> Optional[np.ndarray]:
    """Try J_ij = ||dF_i/dR_j||_F from autograd Hessian. Return None if disconnected."""
    inputs, out = forward_single(model, converter, R, Z, args.device, require_pos_grad=True, energy_only=True)
    pos = inputs[get_position_key()]
    if ENERGY_KEY not in out:
        raise RuntimeError(f"No {ENERGY_KEY!r} in model output; cannot compute response witness.")
    energy = out[ENERGY_KEY].reshape(-1)[0]
    force = torch.autograd.grad(energy, pos, create_graph=True, retain_graph=True, allow_unused=True)[0]
    if force is None:
        return None
    n = R.shape[0]
    J = torch.empty((n, n), dtype=pos.dtype, device=pos.device)
    for i in range(n):
        blocks = []
        for a in range(3):
            g = torch.autograd.grad(force[i, a], pos, retain_graph=True, create_graph=False, allow_unused=True)[0]
            if g is None:
                return None
            blocks.append(g)
        block = torch.stack(blocks, dim=0)
        J[i, :] = torch.linalg.norm(block.permute(1, 0, 2).reshape(n, 9), dim=1)
    return J.detach().cpu().numpy()


def compute_force_response_witness_finite_difference(model, converter, R: np.ndarray, Z: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """Finite-difference response witness J_ij = ||dF_i/dR_j||_F from total predicted forces."""
    n = int(R.shape[0])
    delta = float(args.response_fd_delta)
    J2 = np.zeros((n, n), dtype=np.float64)
    iterator = range(n * 3)
    if bool(getattr(args, "show_fd_progress", False)):
        iterator = tqdm(iterator, desc="finite-diff response coords", leave=False)
    for flat in iterator:
        j = flat // 3
        b = flat % 3
        Rp = np.array(R, copy=True); Rm = np.array(R, copy=True)
        Rp[j, b] += delta
        Rm[j, b] -= delta
        Fp = predict_energy_forces(model, converter, Rp, Z, args)["forces"]
        Fm = predict_energy_forces(model, converter, Rm, Z, args)["forces"]
        dF = (Fp - Fm) / (2.0 * delta)  # [i, alpha]
        J2[:, j] += np.sum(dF * dF, axis=1)
    return np.sqrt(J2)


def compute_force_response_witness(model, converter, R: np.ndarray, Z: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    method = str(getattr(args, "response_method", "finite-diff")).lower()
    if method in {"auto", "autograd"}:
        J = compute_force_response_witness_autograd(model, converter, R, Z, args)
        if J is not None:
            return J
        if method == "autograd":
            raise RuntimeError("Autograd response witness failed; rerun with --response-method finite-diff.")
        print("[response] autograd witness disconnected; using finite-difference response witness.")
    return compute_force_response_witness_finite_difference(model, converter, R, Z, args)


def pair_geometry(R: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return flattened off-diagonal i,j, distances and paper-convention displacement d_ij.

    We use d_ij = R_i - R_j so a simple attractive force on i toward j has theta=180 deg,
    matching the paper convention.
    """
    n = R.shape[0]
    ii, jj = np.where(~np.eye(n, dtype=bool))
    d = R[ii] - R[jj]
    dist = np.linalg.norm(d, axis=1)
    return ii, jj, dist, d


def angle_degrees(Fij_flat: np.ndarray, d_flat: np.ndarray) -> np.ndarray:
    num = np.sum(Fij_flat * d_flat, axis=1)
    den = np.linalg.norm(Fij_flat, axis=1) * np.linalg.norm(d_flat, axis=1) + EPS
    cosang = np.clip(num / den, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))


def fit_rminus7_logc(dist: np.ndarray, mag: np.ndarray, fit_min: float = 5.0, fit_max: float = 10.0) -> float:
    mask = np.isfinite(dist) & np.isfinite(mag) & (dist >= fit_min) & (dist <= fit_max) & (dist > 0) & (mag > EPS)
    if mask.sum() < 10:
        return float("nan")
    # log10(mag) = log10(c) - 7 log10(dist) -> logc = mean(logmag + 7 logdist)
    return float(np.mean(np.log10(mag[mask]) + 7.0 * np.log10(dist[mask])))


def deviation_from_rminus7(dist: np.ndarray, mag: np.ndarray, logc: float) -> np.ndarray:
    dev = np.full_like(mag, np.nan, dtype=np.float64)
    mask = np.isfinite(dist) & np.isfinite(mag) & (dist > 0) & (mag > EPS) & np.isfinite(logc)
    dev[mask] = np.log10(mag[mask]) - logc + 7.0 * np.log10(dist[mask])
    return dev


def fit_rminus_power_logc(
    dist: np.ndarray,
    mag: np.ndarray,
    power: float,
    fit_min: float = 5.0,
    fit_max: float = 10.0,
) -> float:
    """Fit log10(c) for mag ~= c * R^-power over a selected distance window."""
    mask = (
        np.isfinite(dist)
        & np.isfinite(mag)
        & (dist >= fit_min)
        & (dist <= fit_max)
        & (dist > 0)
        & (mag > EPS)
    )
    if mask.sum() < 10:
        return float("nan")
    return float(np.mean(np.log10(mag[mask]) + float(power) * np.log10(dist[mask])))


def deviation_from_rminus_power(dist: np.ndarray, mag: np.ndarray, logc: float, power: float) -> np.ndarray:
    """Return log10(mag / (c * R^-power)); positive means stronger than the fitted decay."""
    dev = np.full_like(mag, np.nan, dtype=np.float64)
    mask = np.isfinite(dist) & np.isfinite(mag) & (dist > 0) & (mag > EPS) & np.isfinite(logc)
    dev[mask] = np.log10(mag[mask]) - logc + float(power) * np.log10(dist[mask])
    return dev


def _distance_label(x: float) -> str:
    """Stable label component for distance filters."""
    if math.isinf(float(x)):
        return "inf"
    s = f"{float(x):g}".replace("-", "m").replace(".", "p")
    return s


def distance_filter_masks(dist: np.ndarray, args: argparse.Namespace) -> List[Tuple[str, np.ndarray]]:
    """Build masks for Adil's distance-filtered comparison.

    Defaults include finite all-pairs, contiguous bins, and hard long-range filters
    such as R > 5 Å and R > 10 Å.
    """
    finite = np.isfinite(dist) & (dist > 0)
    masks: List[Tuple[str, np.ndarray]] = [("all", finite)]

    cuts = sorted(set(float(x) for x in getattr(args, "distance_bins", [0.0, 3.0, 5.0, 10.0, 15.0])))
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        masks.append((f"{_distance_label(lo)}to{_distance_label(hi)}A", finite & (dist >= lo) & (dist < hi)))
    if cuts:
        last = cuts[-1]
        masks.append((f"gt{_distance_label(last)}A", finite & (dist >= last)))

    for md in sorted(set(float(x) for x in getattr(args, "min_distances", [5.0, 10.0]))):
        masks.append((f"gt{_distance_label(md)}A", finite & (dist >= md)))

    # De-duplicate while preserving order; bins and hard filters can share labels.
    out: List[Tuple[str, np.ndarray]] = []
    seen = set()
    for label, mask in masks:
        if label not in seen:
            out.append((label, mask))
            seen.add(label)
    return out


def summarize_distance_filters(
    dist: np.ndarray,
    mag: np.ndarray,
    theta: np.ndarray,
    dev: np.ndarray,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    """Distance-binned force-depth/anisotropy summary across analysis frames."""
    rows: List[Dict[str, Any]] = []
    logmag = np.log10(np.maximum(mag, EPS))
    for label, mask in distance_filter_masks(dist, args):
        mask = mask & np.isfinite(logmag)
        n = int(mask.sum())
        if n < int(getattr(args, "min_pairs_per_filter", 20)):
            continue
        row = {
            "distance_filter": label,
            "n_pairs": n,
            "mean_distance_A": float(np.nanmean(dist[mask])),
            "mean_log10_force": float(np.nanmean(logmag[mask])),
            "std_log10_force": float(np.nanstd(logmag[mask])),
            "iqr_log10_force": float(np.nanpercentile(logmag[mask], 75) - np.nanpercentile(logmag[mask], 25)),
            "aligned_fraction_theta_gt150": float(np.nanmean(theta[mask] > 150.0)),
            "anisotropic_fraction_theta_lt150": float(np.nanmean(theta[mask] < 150.0)),
            "mean_log10_deviation_from_rminus7": float(np.nanmean(dev[mask])),
            "p95_log10_deviation_from_rminus7": float(np.nanpercentile(dev[mask], 95)) if np.isfinite(dev[mask]).any() else float("nan"),
            "p99_log10_deviation_from_rminus7": float(np.nanpercentile(dev[mask], 99)) if np.isfinite(dev[mask]).any() else float("nan"),
        }
        rows.append(row)
    return rows


def response_score_variants(
    dist: np.ndarray,
    response: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[Dict[str, np.ndarray], float]:
    """Raw and R^-8-corrected force-response witness scores.

    Ranking on `r8_scaled_log10` and `r8_residual_log10` is identical for one frame,
    but the residual is centered against the fitted c*R^-8 baseline and is easier to
    interpret as "stronger/weaker than expected".
    """
    power = float(getattr(args, "response_decay_power", 8.0))
    fit_min = float(getattr(args, "response_rfit_min", getattr(args, "rfit_min", 5.0)))
    fit_max = float(getattr(args, "response_rfit_max", getattr(args, "rfit_max", 10.0)))

    log_response = np.full_like(response, np.nan, dtype=np.float64)
    good = np.isfinite(response) & (response > EPS)
    log_response[good] = np.log10(response[good])

    logc = fit_rminus_power_logc(dist, response, power=power, fit_min=fit_min, fit_max=fit_max)

    scaled = np.full_like(response, np.nan, dtype=np.float64)
    good_scaled = good & np.isfinite(dist) & (dist > 0)
    scaled[good_scaled] = np.log10(response[good_scaled]) + power * np.log10(dist[good_scaled])

    resid = deviation_from_rminus_power(dist, response, logc=logc, power=power)
    return {
        "raw_log10": log_response,
        f"r{_distance_label(power)}_scaled_log10": scaled,
        f"r{_distance_label(power)}_residual_log10": resid,
    }, logc


def masked_overlap_stats(
    attr_scores: np.ndarray,
    response_scores: np.ndarray,
    mask: np.ndarray,
    topk_frac: float,
) -> Dict[str, Any]:
    """Top-k and Spearman stats inside one distance mask."""
    valid = mask & np.isfinite(attr_scores) & np.isfinite(response_scores)
    idx = np.where(valid)[0]
    if idx.size < 3:
        return {
            "n_pairs": int(idx.size),
            "topk_jaccard_attr_vs_response": float("nan"),
            "spearman_attr_vs_response": float("nan"),
            "topk_size": 0,
        }

    # topk_indices returns local indices; compare within the masked subarray.
    a = attr_scores[idx]
    b = response_scores[idx]
    k = max(1, int(round(float(topk_frac) * idx.size)))
    return {
        "n_pairs": int(idx.size),
        "topk_size": int(k),
        "topk_jaccard_attr_vs_response": jaccard(topk_indices(a, topk_frac), topk_indices(b, topk_frac)),
        "spearman_attr_vs_response": spearman_np(a, b),
    }


def summarize_interactions(dist: np.ndarray, mag: np.ndarray, theta: np.ndarray, dev: np.ndarray, analysis_frames: int, recon_mae: Sequence[float]) -> Dict[str, Any]:
    logmag = np.log10(np.maximum(mag, EPS))
    bins = np.arange(max(0, math.floor(np.nanmin(dist))), math.ceil(np.nanmax(dist)) + 1, 1.0)
    iqr_by_bin = []
    std_by_bin = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (dist >= lo) & (dist < hi) & np.isfinite(logmag)
        if m.sum() >= 50:
            vals = logmag[m]
            iqr_by_bin.append(float(np.nanpercentile(vals, 75) - np.nanpercentile(vals, 25)))
            std_by_bin.append(float(np.nanstd(vals)))

    def frac(mask):
        denom = int(np.sum(mask))
        return float(np.sum(mask) / max(denom, 1))

    ge10 = dist >= 10.0
    ge15 = dist >= 15.0
    summary = {
        "n_pairs": int(len(dist)),
        "analysis_frames": int(analysis_frames),
        "n_bins_used": int(len(iqr_by_bin)),
        "mean_log10_force_iqr_by_1A_bins": float(np.nanmean(iqr_by_bin)) if iqr_by_bin else float("nan"),
        "mean_log10_force_std_by_1A_bins": float(np.nanmean(std_by_bin)) if std_by_bin else float("nan"),
        "overall_log10_force_range": float(np.nanmax(logmag) - np.nanmin(logmag)),
        "aligned_fraction_theta_gt150_all": float(np.nanmean(theta > 150.0)),
        "aligned_fraction_theta_gt150_dist_ge10": float(np.nanmean(theta[ge10] > 150.0)) if ge10.any() else float("nan"),
        "aligned_fraction_theta_gt150_dist_ge15": float(np.nanmean(theta[ge15] > 150.0)) if ge15.any() else float("nan"),
        "anisotropic_fraction_theta_lt150_dist_ge10": float(np.nanmean(theta[ge10] < 150.0)) if ge10.any() else float("nan"),
        "mean_reconstruction_mae_sumj_Fij_equals_Fi": float(np.nanmean(recon_mae)) if recon_mae else float("nan"),
        "mean_log10_deviation_from_rminus7": float(np.nanmean(dev)),
        "p95_log10_deviation_from_rminus7": float(np.nanpercentile(dev[np.isfinite(dev)], 95)) if np.isfinite(dev).any() else float("nan"),
        "p99_log10_deviation_from_rminus7": float(np.nanpercentile(dev[np.isfinite(dev)], 99)) if np.isfinite(dev).any() else float("nan"),
    }
    return summary


def topk_indices(scores: np.ndarray, frac: float) -> set[int]:
    finite = np.where(np.isfinite(scores))[0]
    if finite.size == 0:
        return set()
    k = max(1, int(round(frac * finite.size)))
    order = finite[np.argsort(scores[finite])[-k:]]
    return set(int(x) for x in order)


def jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def spearman_np(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    try:
        from scipy.stats import spearmanr  # type: ignore
        return float(spearmanr(a[mask], b[mask]).correlation)
    except Exception:
        # Tiny fallback rank-corr.
        ar = np.argsort(np.argsort(a[mask]))
        br = np.argsort(np.argsort(b[mask]))
        return float(np.corrcoef(ar, br)[0, 1])


def analytic_zero_sum_gauge(R: np.ndarray, strength: float, phase: float, scale: float) -> np.ndarray:
    """Coordinate-dependent zero-sum per-atom energy gauge g_j(R), sum_j g_j == 0.

    This creates an attribution gauge that changes per-atom gradients while preserving the
    total energy and total force exactly up to floating-point/autograd precision.
    """
    x, y, z = R[:, 0], R[:, 1], R[:, 2]
    # Local coordinate terms plus a geometry/environment term.
    raw = np.sin(0.73 * x + phase) + 0.55 * np.cos(0.41 * y - 0.3 * phase) + 0.25 * np.sin(0.29 * z + 0.7 * phase)
    # Add a smooth pair environment scalar so the gauge gradients are not purely local.
    D = np.linalg.norm(R[:, None, :] - R[None, :, :], axis=-1)
    env = np.exp(-D / 4.0)
    np.fill_diagonal(env, 0.0)
    raw = raw + 0.1 * env.sum(axis=1)
    raw = raw - raw.mean()
    return strength * scale * raw


def compute_gauge_fij_from_baseline(Fij: np.ndarray, R: np.ndarray, strength: float, phase: float, scale: float) -> Tuple[np.ndarray, float]:
    """Analytic gauge derivative added to Fij.

    We use the same zero-sum gauge family as analytic_zero_sum_gauge, but compute the Jacobian
    with finite differences. This avoids coupling the gauge test to SchNetPack internals while
    preserving the total-force invariant sum_j g_j = 0.
    """
    n = R.shape[0]
    delta = 1e-4
    Ggrad = np.zeros((n, n, 3), dtype=np.float64)  # d g_j / d R_i_alpha
    for i in range(n):
        for a in range(3):
            Rp = R.copy(); Rm = R.copy()
            Rp[i, a] += delta
            Rm[i, a] -= delta
            gp = analytic_zero_sum_gauge(Rp, strength, phase, scale)
            gm = analytic_zero_sum_gauge(Rm, strength, phase, scale)
            Ggrad[i, :, a] = (gp - gm) / (2.0 * delta)
    Fij_g = Fij - Ggrad  # F'_ij = -d(E_j + g_j)/dR_i = Fij - dg_j/dR_i
    total_force_delta = np.mean(np.abs(Fij_g.sum(axis=1) - Fij.sum(axis=1)))
    return Fij_g, float(total_force_delta)


def analyze_frame_from_fij(R: np.ndarray, Fij: np.ndarray, fit_min: float, fit_max: float) -> Dict[str, Any]:
    ii, jj, dist, d = pair_geometry(R)
    Fflat = Fij[ii, jj, :]
    mag = np.linalg.norm(Fflat, axis=1)
    theta = angle_degrees(Fflat, d)
    logc = fit_rminus7_logc(dist, mag, fit_min=fit_min, fit_max=fit_max)
    dev = deviation_from_rminus7(dist, mag, logc)
    return {"ii": ii, "jj": jj, "dist": dist, "mag": mag, "theta": theta, "dev": dev, "logc": logc}


def run_interaction_analysis(model, R: np.ndarray, F_ref: np.ndarray, Z: np.ndarray, args: argparse.Namespace, run_dir: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    converter = make_converter(args)
    n_frames = min(args.analysis_frames, len(R))
    rng = np.random.default_rng(args.seed + 101)
    frame_indices = np.sort(rng.choice(len(R), size=n_frames, replace=False)) if n_frames > 0 else np.array([], dtype=int)

    all_dist, all_mag, all_theta, all_dev = [], [], [], []
    recon_mae = []
    recon_mae_neg = []
    recon_mae_scaled = []
    recon_alpha = []
    recon_cosine = []
    force_model_mae = []
    frame_records: List[Dict[str, Any]] = []
    gauge_rows: List[Dict[str, Any]] = []
    response_rows: List[Dict[str, Any]] = []


    for local_idx, frame_idx in enumerate(tqdm(frame_indices, desc="SchNetPack Fij analysis")):
        pred = compute_per_atom_fij(model, converter, R[frame_idx], Z, args)
        Fij = pred["Fij"]
        force_pred = pred["forces"]
        sum_force = Fij.sum(axis=1)
        rec = analyze_frame_from_fij(R[frame_idx], Fij, args.rfit_min, args.rfit_max)

        # Validation diagnostics. For a mathematically perfect PaiNN attribution,
        # sum_j F_ij should match the model force from the same energy. With SchNetPack
        # inference models and finite differences, we also check sign and scalar transforms
        # so we can tell whether a mismatch is just normalization/sign bookkeeping.
        rec_mae = float(np.mean(np.abs(sum_force - force_pred)))
        rec_mae_n = float(np.mean(np.abs(-sum_force - force_pred)))
        denom = float(np.sum(sum_force * sum_force)) + EPS
        alpha = float(np.sum(sum_force * force_pred) / denom)
        rec_mae_s = float(np.mean(np.abs(alpha * sum_force - force_pred)))
        cos = float(np.sum(sum_force * force_pred) / ((np.linalg.norm(sum_force) * np.linalg.norm(force_pred)) + EPS))
        f_mae = float(np.mean(np.abs(force_pred)))
        recon_mae.append(rec_mae)
        recon_mae_neg.append(rec_mae_n)
        recon_mae_scaled.append(rec_mae_s)
        recon_alpha.append(alpha)
        recon_cosine.append(cos)
        force_model_mae.append(f_mae)

        all_dist.append(rec["dist"]); all_mag.append(rec["mag"]); all_theta.append(rec["theta"]); all_dev.append(rec["dev"])
        frame_records.append({
            "frame_idx": int(frame_idx),
            "pred_energy": pred["energy"],
            "fij_method_used": pred.get("fij_method_used", "unknown"),
            "unused_per_atom_grads": pred.get("unused_per_atom_grads", -1),
            "rminus7_log10_c": rec["logc"],
            "reconstruction_mae_sumj_Fij_equals_model_Fi": rec_mae,
            "reconstruction_mae_minus_sumj_Fij_equals_model_Fi": rec_mae_n,
            "reconstruction_best_scalar_alpha": alpha,
            "reconstruction_mae_scaled_sumj_Fij_equals_model_Fi": rec_mae_s,
            "reconstruction_cosine_sumj_vs_model_Fi": cos,
        })

        # Gauge stability on this frame.
        base_scores = rec["dev"] if args.ranking_key == "deviation" else np.log10(np.maximum(rec["mag"], EPS))
        base_top = topk_indices(base_scores, args.topk_frac)
        # Use learned per-atom energy std as a natural scale when possible.
        scale = float(np.std(pred["per_atom_energy"]))
        if not np.isfinite(scale) or scale < EPS:
            scale = float(np.std(pred["energy"])) if np.ndim(pred["energy"]) else max(abs(float(pred["energy"])), 1.0) / max(1, len(Z))
        scale = max(scale, 1e-3)

        for strength in args.gauge_strengths:
            for trial in range(args.gauge_trials):
                phase = 0.37 * trial + 1.91 * local_idx + args.seed * 0.001
                Fij_g, total_delta = compute_gauge_fij_from_baseline(Fij, R[frame_idx], strength, phase, scale)
                rec_g = analyze_frame_from_fij(R[frame_idx], Fij_g, args.rfit_min, args.rfit_max)
                scores_g = rec_g["dev"] if args.ranking_key == "deviation" else np.log10(np.maximum(rec_g["mag"], EPS))
                gauge_rows.append({
                    "frame_idx": int(frame_idx),
                    "strength": float(strength),
                    "trial": int(trial),
                    "topk_jaccard": jaccard(base_top, topk_indices(scores_g, args.topk_frac)),
                    "spearman": spearman_np(base_scores, scores_g),
                    "total_force_delta_mae": total_delta,
                    "base_p99_dev": float(np.nanpercentile(base_scores[np.isfinite(base_scores)], 99)) if np.isfinite(base_scores).any() else float("nan"),
                    "gauge_p99_dev": float(np.nanpercentile(scores_g[np.isfinite(scores_g)], 99)) if np.isfinite(scores_g).any() else float("nan"),
                })

        # Optional response witness. This is expensive.
        # Adil-calibrated layer:
        #   1) test all pairs and long-range-only pairs (>5 A, >10 A by default),
        #   2) compare raw response and R^-8-corrected response rankings.
        if local_idx < args.response_frames:
            J = compute_force_response_witness(model, converter, R[frame_idx], Z, args)
            ii, jj, dist_flat, _ = pair_geometry(R[frame_idx])
            Jflat = J[ii, jj]
            response_scores_by_mode, response_logc = response_score_variants(dist_flat, Jflat, args)

            # Save per-pair arrays for post-hoc plotting/debugging without recomputing Hessian/FD witness.
            np.savez_compressed(
                run_dir / f"response_pair_arrays_frame_{int(frame_idx)}.npz",
                ii=ii,
                jj=jj,
                dist=dist_flat,
                attr_scores=base_scores,
                response_raw=Jflat,
                response_log10_c_rminus_power=response_logc,
                response_decay_power=float(args.response_decay_power),
                **{f"response_score_{k}": v for k, v in response_scores_by_mode.items()},
            )

            for response_mode, response_scores in response_scores_by_mode.items():
                for filter_name, mask in distance_filter_masks(dist_flat, args):
                    stats = masked_overlap_stats(base_scores, response_scores, mask, args.topk_frac)
                    if stats["n_pairs"] < int(args.min_pairs_per_filter):
                        continue
                    valid = mask & np.isfinite(response_scores)
                    response_rows.append({
                        "frame_idx": int(frame_idx),
                        "distance_filter": filter_name,
                        "response_score_mode": response_mode,
                        "response_decay_power": float(args.response_decay_power),
                        "response_log10_c_rminus_power": response_logc,
                        "topk_frac": float(args.topk_frac),
                        "response_p50_score": float(np.nanpercentile(response_scores[valid], 50)) if valid.any() else float("nan"),
                        "response_p95_score": float(np.nanpercentile(response_scores[valid], 95)) if valid.any() else float("nan"),
                        "response_p99_score": float(np.nanpercentile(response_scores[valid], 99)) if valid.any() else float("nan"),
                        **stats,
                    })

    dist = np.concatenate(all_dist) if all_dist else np.array([])
    mag = np.concatenate(all_mag) if all_mag else np.array([])
    theta = np.concatenate(all_theta) if all_theta else np.array([])
    dev = np.concatenate(all_dev) if all_dev else np.array([])
    summary = summarize_interactions(dist, mag, theta, dev, n_frames, recon_mae)
    distance_rows = summarize_distance_filters(dist, mag, theta, dev, args)
    for row in distance_rows:
        label = row["distance_filter"]
        summary[f"distance_{label}_n_pairs"] = int(row["n_pairs"])
        summary[f"distance_{label}_aligned_fraction_theta_gt150"] = float(row["aligned_fraction_theta_gt150"])
        summary[f"distance_{label}_iqr_log10_force"] = float(row["iqr_log10_force"])
        summary[f"distance_{label}_p99_log10_deviation_from_rminus7"] = float(row["p99_log10_deviation_from_rminus7"])
    summary["mean_reconstruction_mae_minus_sumj_Fij_equals_model_Fi"] = float(np.nanmean(recon_mae_neg)) if recon_mae_neg else float("nan")
    summary["mean_reconstruction_best_scalar_alpha_sumj_to_model_Fi"] = float(np.nanmean(recon_alpha)) if recon_alpha else float("nan")
    summary["mean_reconstruction_mae_scaled_sumj_Fij_equals_model_Fi"] = float(np.nanmean(recon_mae_scaled)) if recon_mae_scaled else float("nan")
    summary["mean_reconstruction_cosine_sumj_vs_model_Fi"] = float(np.nanmean(recon_cosine)) if recon_cosine else float("nan")
    summary["mean_abs_model_force_component"] = float(np.nanmean(force_model_mae)) if force_model_mae else float("nan")
    if frame_records:
        methods = {}
        for rr in frame_records:
            m = str(rr.get("fij_method_used", "unknown"))
            methods[m] = methods.get(m, 0) + 1
        for m, c in methods.items():
            summary[f"fij_method_frames_{m}"] = int(c)

    # Aggregate gauge/response stats.
    for strength in args.gauge_strengths:
        rows = [r for r in gauge_rows if abs(r["strength"] - strength) < 1e-15]
        if rows:
            prefix = f"gauge_{strength:g}"
            summary[f"{prefix}_mean_topk_jaccard"] = float(np.nanmean([r["topk_jaccard"] for r in rows]))
            summary[f"{prefix}_mean_spearman"] = float(np.nanmean([r["spearman"] for r in rows]))
            summary[f"{prefix}_mean_total_force_delta_mae"] = float(np.nanmean([r["total_force_delta_mae"] for r in rows]))
    if response_rows:
        # Preserve the old headline keys for raw/all so old comparisons do not break.
        raw_all = [r for r in response_rows if r.get("response_score_mode") == "raw_log10" and r.get("distance_filter") == "all"]
        if raw_all:
            summary["response_mean_topk_jaccard_attr_vs_response"] = float(np.nanmean([r["topk_jaccard_attr_vs_response"] for r in raw_all]))
            summary["response_mean_spearman_attr_vs_response"] = float(np.nanmean([r["spearman_attr_vs_response"] for r in raw_all]))

        for mode in sorted(set(str(r.get("response_score_mode", "unknown")) for r in response_rows)):
            for filt in sorted(set(str(r.get("distance_filter", "unknown")) for r in response_rows)):
                rows = [r for r in response_rows if r.get("response_score_mode") == mode and r.get("distance_filter") == filt]
                if not rows:
                    continue
                key_base = f"response_{mode}_{filt}"
                summary[f"{key_base}_mean_topk_jaccard"] = float(np.nanmean([r["topk_jaccard_attr_vs_response"] for r in rows]))
                summary[f"{key_base}_mean_spearman"] = float(np.nanmean([r["spearman_attr_vs_response"] for r in rows]))
                summary[f"{key_base}_mean_n_pairs"] = float(np.nanmean([r["n_pairs"] for r in rows]))

    # Save raw tables.
    np.savez_compressed(run_dir / "interaction_arrays.npz", dist=dist, mag=mag, theta=theta, dev=dev)
    write_csv(run_dir / "frame_records.csv", frame_records)
    write_csv(run_dir / "distance_filter_summary.csv", distance_rows)
    write_csv(run_dir / "gauge_trials.csv", gauge_rows)
    write_csv(run_dir / "response_trials.csv", response_rows)
    make_plots(run_dir, dist, mag, theta, dev, gauge_rows, response_rows)

    json_dump(summary, run_dir / "analysis_summary.json")
    return summary, frame_records


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted(set().union(*(r.keys() for r in rows)))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def make_plots(run_dir: Path, dist: np.ndarray, mag: np.ndarray, theta: np.ndarray, dev: np.ndarray, gauge_rows: List[Dict[str, Any]], response_rows: List[Dict[str, Any]]) -> None:
    if len(dist) == 0:
        return
    logmag = np.log10(np.maximum(mag, EPS))

    plt.figure(figsize=(9, 6))
    plt.hist2d(dist, logmag, bins=[80, 80])
    plt.xlabel("Distance R_ij [Ang]")
    plt.ylabel("log10 |F_ij|")
    plt.title("DHA SchNetPack PaiNN interaction depth")
    plt.colorbar(label="count")
    plt.tight_layout()
    plt.savefig(run_dir / "interaction_depth_hist2d.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 6))
    plt.hist2d(dist, theta, bins=[80, 72], range=[[float(np.nanmin(dist)), float(np.nanmax(dist))], [0, 180]])
    plt.xlabel("Distance R_ij [Ang]")
    plt.ylabel("theta_ij [deg] (180 = attractive/pairwise-like)")
    plt.title("DHA SchNetPack PaiNN anisotropy")
    plt.colorbar(label="count")
    plt.tight_layout()
    plt.savefig(run_dir / "anisotropy_hist2d.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 6))
    plt.hist2d(dist, dev, bins=[80, 80])
    plt.xlabel("Distance R_ij [Ang]")
    plt.ylabel("log10 deviation from fitted R^-7")
    plt.title("DHA SchNetPack PaiNN R^-7 deviation")
    plt.colorbar(label="count")
    plt.tight_layout()
    plt.savefig(run_dir / "rminus7_deviation_hist2d.png", dpi=180)
    plt.close()

    if gauge_rows:
        strengths = sorted(set(float(r["strength"]) for r in gauge_rows))
        data = [[float(r["topk_jaccard"]) for r in gauge_rows if abs(float(r["strength"]) - s) < 1e-15] for s in strengths]
        plt.figure(figsize=(8, 5))
        kwargs = {"showmeans": True}
        try:
            plt.boxplot(data, tick_labels=[f"{s:g}" for s in strengths], **kwargs)
        except TypeError:
            plt.boxplot(data, labels=[f"{s:g}" for s in strengths], **kwargs)
        plt.xlabel("zero-sum gauge strength")
        plt.ylabel("top-k Jaccard vs baseline")
        plt.title("Attribution-gauge stability")
        plt.tight_layout()
        plt.savefig(run_dir / "gauge_stability_boxplot.png", dpi=180)
        plt.close()

    if response_rows:
        # Compact plot: mean top-k Jaccard by distance filter for each response scoring mode.
        grouped: Dict[Tuple[str, str], List[float]] = {}
        for r in response_rows:
            key = (str(r.get("response_score_mode", "unknown")), str(r.get("distance_filter", "unknown")))
            grouped.setdefault(key, []).append(float(r.get("topk_jaccard_attr_vs_response", np.nan)))
        labels = []
        vals = []
        for mode, filt in sorted(grouped):
            labels.append(f"{mode}\n{filt}")
            vals.append(float(np.nanmean(grouped[(mode, filt)])))
        plt.figure(figsize=(max(8, 0.55 * len(vals)), 5))
        plt.bar(np.arange(len(vals)), vals)
        plt.xticks(np.arange(len(vals)), labels, rotation=75, ha="right", fontsize=8)
        plt.xlabel("response score / distance filter")
        plt.ylabel("mean top-k Jaccard")
        plt.title("Attribution hotspots vs force-response witness")
        plt.tight_layout()
        plt.savefig(run_dir / "response_witness_overlap.png", dpi=180)
        plt.close()



def evaluate_model_subset(model, R: np.ndarray, F: np.ndarray, E: np.ndarray, Z: np.ndarray, args: argparse.Namespace, run_dir: Path) -> Dict[str, Any]:
    """Evaluate total energy/force predictions without doing the expensive Fij decomposition."""
    converter = make_converter(args)
    rng = np.random.default_rng(args.seed + 202)
    n = min(args.eval_frames, len(R))
    idx = np.sort(rng.choice(len(R), size=n, replace=False)) if n > 0 else []
    e_abs, f_abs = [], []
    rows = []
    for frame_idx in tqdm(idx, desc="Eval subset"):
        pred = predict_energy_forces(model, converter, R[frame_idx], Z, args)
        e_mae = abs(pred["energy"] - float(E[frame_idx]))
        f_mae = float(np.mean(np.abs(pred["forces"] - F[frame_idx])))
        e_abs.append(e_mae); f_abs.append(f_mae)
        rows.append({"frame_idx": int(frame_idx), "E_abs": e_mae, "F_mae": f_mae, "E_pred": pred["energy"], "E_ref": float(E[frame_idx])})
    write_csv(run_dir / "eval_subset.csv", rows)
    return {
        "eval_frames": int(n),
        "eval_E_MAE": float(np.mean(e_abs)) if e_abs else float("nan"),
        "eval_F_MAE": float(np.mean(f_abs)) if f_abs else float("nan"),
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DHA SchNetPack PaiNN F_ij decomposition + Adil-calibrated response probe")
    p.add_argument("--hf-repo", default="colabfit/MD22_DHA")
    p.add_argument("--cache-dir", default="dha_probe_cache")
    p.add_argument("--run-root", default="dha_painn_runs")
    p.add_argument("--npz-name", default="md22_dha_colabfit.npz")
    p.add_argument("--db-name", default="md22_dha_schnetpack.db")
    p.add_argument("--split-name", default="md22_dha_split.npz")
    p.add_argument("--refresh-cache", action="store_true")
    p.add_argument("--rebuild-db", action="store_true")
    p.add_argument("--nframes", type=int, default=2500)
    p.add_argument("--seed", type=int, default=7)

    # Units/scales: default as-is. Use --energy-scale/--force-scale if you confirm HF units need conversion.
    p.add_argument("--energy-scale", type=float, default=1.0)
    p.add_argument("--force-scale", type=float, default=1.0)
    p.add_argument("--energy-unit", default="kcal/mol")
    p.add_argument("--force-unit", default="kcal/mol/Ang")

    # Paper-ish PaiNN defaults.
    p.add_argument("--cutoff", type=float, default=10.0)
    p.add_argument("--n-atom-basis", type=int, default=128)
    p.add_argument("--n-rbf", type=int, default=30)
    p.add_argument("--n-interactions", type=int, default=3)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--energy-loss-weight", type=float, default=0.01)
    p.add_argument("--force-loss-weight", type=float, default=0.99)
    p.add_argument("--grad-clip", type=float, default=0.0)

    p.add_argument("--num-train", type=int, default=950)
    p.add_argument("--num-val", type=int, default=50)
    p.add_argument("--num-test", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=8)
    # Windows spawn-based DataLoader workers can be much slower and more memory-hungry than the
    # Lightning warning implies, especially when heavy packages are imported in every child process.
    # Default to safe/stable on Windows; users can opt in explicitly with --allow-windows-workers.
    p.add_argument("--num-workers", type=int, default=(0 if os.name == "nt" else min(8, max(0, (os.cpu_count() or 1) - 1))))
    p.add_argument("--allow-windows-workers", action="store_true", help="Honor --num-workers > 0 on Windows. Off by default because spawn workers caused memory errors/freezes in this probe.")
    p.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=False, help="Use persistent DataLoader workers when num_workers > 0. Usually leave off on Windows unless testing.")
    p.add_argument("--max-epochs", type=int, default=50)
    p.add_argument("--log-every-n-steps", type=int, default=10)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--force-train", action="store_true", help="Train a fresh model even if a previous best_inference_model exists.")
    p.add_argument("--train-only", action="store_true", help="Stop after training/loading the model; useful before expensive analysis.")
    p.add_argument("--matmul-precision", choices=["highest", "high", "medium"], default="high", help="Enable Tensor Core friendly float32 matmul precision on RTX GPUs.")
    p.add_argument("--model-path", default="")

    # Analysis.
    p.add_argument("--eval-frames", type=int, default=32)
    p.add_argument("--analysis-frames", type=int, default=24)
    p.add_argument("--response-frames", type=int, default=2)
    p.add_argument("--rfit-min", type=float, default=5.0)
    p.add_argument("--rfit-max", type=float, default=10.0)

    # Adil-calibrated filters/tests:
    # - compare short-range vs long-range pairs,
    # - test response witness after the expected stiffness decay R^-8 is removed.
    p.add_argument("--distance-bins", type=float, nargs="+", default=[0.0, 3.0, 5.0, 10.0, 15.0],
                   help="Contiguous distance cut points in Angstrom for binned interaction/response summaries.")
    p.add_argument("--min-distances", type=float, nargs="+", default=[5.0, 10.0],
                   help="Additional hard filters R >= value, e.g. Adil's >5A and >10A suggestions.")
    p.add_argument("--min-pairs-per-filter", type=int, default=20,
                   help="Skip distance filters with fewer valid pairs than this.")
    p.add_argument("--response-decay-power", type=float, default=8.0,
                   help="Expected stiffness decay power for response witness; R^-8 follows from force ~ R^-7.")
    p.add_argument("--response-rfit-min", type=float, default=5.0,
                   help="Lower distance bound for fitting c*R^-response_decay_power.")
    p.add_argument("--response-rfit-max", type=float, default=10.0,
                   help="Upper distance bound for fitting c*R^-response_decay_power.")
    p.add_argument("--ranking-key", choices=["deviation", "magnitude"], default="deviation")
    p.add_argument("--topk-frac", type=float, default=0.02)
    p.add_argument("--gauge-strengths", type=float, nargs="+", default=[0.0, 0.01, 0.03, 0.1, 0.3])
    p.add_argument("--gauge-trials", type=int, default=3)
    p.add_argument("--fij-method", choices=["auto", "autograd", "finite-diff"], default="auto", help="How to compute Fij=-dE_j/dR_i. auto tries autograd then falls back to finite differences.")
    p.add_argument("--fij-fd-delta", type=float, default=1e-3, help="Central finite-difference step in Angstrom for Fij fallback.")
    p.add_argument("--response-method", choices=["auto", "autograd", "finite-diff"], default="finite-diff", help="How to compute force-response witness. finite-diff is most robust across SchNetPack builds.")
    p.add_argument("--response-fd-delta", type=float, default=1e-3, help="Central finite-difference step in Angstrom for response witness.")
    p.add_argument("--show-fd-progress", action="store_true", help="Show nested finite-difference progress bars. Useful for timing, noisy for normal runs.")
    p.add_argument("--verbose-fallback", action=argparse.BooleanOptionalAction, default=True, help="Print when auto Fij falls back to finite differences.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_all(args.seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable. Falling back to CPU.")
        args.device = "cpu"

    if os.name == "nt" and args.num_workers > 0 and not args.allow_windows_workers:
        print(f"[TRAIN] Windows-safe mode: clamping num_workers {args.num_workers} -> 0. "
              "Lightning's worker-count hint is generic; this run already showed spawn workers causing MemoryError/freezes. "
              "Use --allow-windows-workers to override.")
        args.num_workers = 0
        args.persistent_workers = False

    if args.device.startswith("cuda"):
        try:
            torch.set_float32_matmul_precision(args.matmul_precision)
            print(f"[TORCH] float32 matmul precision = {args.matmul_precision}")
        except Exception as exc:
            print(f"[WARN] Could not set matmul precision: {exc}")

    run_dir = ensure_dir(Path(args.run_root) / time.strftime("dha_painn_probe_%Y%m%d_%H%M%S"))
    json_dump(vars(args), run_dir / "args.json")

    print("=" * 100)
    print("DHA SchNetPack PaiNN decomposition probe")
    print("=" * 100)
    print(f"[RUN] {run_dir}")
    print(f"[DEVICE] {args.device}")
    print(f"[DATALOADER] num_workers={args.num_workers} persistent_workers={args.persistent_workers}")

    R, F, E, Z, meta = load_or_build_npz_cache(args)
    print(f"[DATA] R={R.shape} F={F.shape} E={E.shape} Z={Z.shape} source={meta.get('source')}")
    unique, counts = np.unique(Z, return_counts=True)
    print(f"[DATA] Z counts: {dict(zip(unique.tolist(), counts.tolist()))}")
    json_dump(meta, run_dir / "data_metadata.json")

    if args.num_test is None:
        args.num_test = max(0, len(E) - args.num_train - args.num_val)
    print(f"[SPLIT] train={args.num_train} val={args.num_val} test={args.num_test}")

    # Avoid the expensive SchNetPack datamodule/statistics pass when we are only loading
    # an existing inference model for analysis. On Windows, many workers can otherwise
    # spawn lots of Python processes just to calculate stats we do not need.
    dm = None
    explicit_model = bool(args.model_path)
    latest_model = find_latest_inference_model(args.run_root) if not args.force_train else None
    can_load_without_training = explicit_model or (latest_model is not None and not args.force_train)

    if can_load_without_training:
        model, model_path = train_or_load_model(args, dm, run_dir)
    else:
        db_path = build_ase_db(args, R, F, E, Z)
        dm = build_datamodule(args, db_path)
        # Prepare once here so metadata transforms/offsets are available before training.
        dm.prepare_data()
        dm.setup()
        model, model_path = train_or_load_model(args, dm, run_dir)
    print(f"[MODEL] Active model path: {model_path}")

    if args.train_only:
        json_dump({"model_path": str(model_path), "args": vars(args)}, run_dir / "train_only_report.json")
        print(f"[TRAIN-ONLY] Stopping before evaluation/analysis. Model: {model_path}")
        print(f"[OUT] {run_dir}")
        return

    eval_summary = evaluate_model_subset(model, R, F, E, Z, args, run_dir)
    analysis_summary, frame_records = run_interaction_analysis(model, R, F, Z, args, run_dir)

    final = {
        "args": vars(args),
        "data": {"R_shape": list(R.shape), "F_shape": list(F.shape), "E_shape": list(E.shape), "Z": Z.tolist(), "metadata": meta},
        "model_path": str(model_path),
        "eval": eval_summary,
        "analysis": analysis_summary,
    }
    json_dump(final, run_dir / "final_report.json")

    print("\n[SUMMARY]")
    for k, v in eval_summary.items():
        print(f"  {k:55s}: {v}")
    for k, v in analysis_summary.items():
        print(f"  {k:55s}: {v}")
    print(f"\n[OUT] {run_dir}")


if __name__ == "__main__":
    main()
