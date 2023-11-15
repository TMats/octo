from enum import IntEnum
import hashlib
import inspect
import json
import logging
import os
from typing import Any, Callable, Dict, List

import dlimp as dl
import numpy as np
import tensorflow as tf
from tensorflow_datasets.core.dataset_builder import DatasetBuilder
import tqdm


def tree_map(fn: Callable, tree: dict) -> dict:
    """Maps a function over a nested dictionary."""
    return {
        k: tree_map(fn, v) if isinstance(v, dict) else fn(v) for k, v in tree.items()
    }


class StateEncoding(IntEnum):
    """Defines supported proprio state encoding schemes for different datasets."""

    NONE = -1  # no state provided
    POS_EULER = 1  # EEF XYZ + roll-pitch-yaw + 1 x pad + gripper open/close
    POS_QUAT = 2  # EEF XYZ + quaternion + gripper open/close
    JOINT = 3  # 7 x joint angles (padding added if fewer) + gripper open/close


class ActionEncoding(IntEnum):
    """Defines supported action encoding schemes for different datasets."""

    EEF_POS = 1  # EEF delta XYZ + roll-pitch-yaw + gripper open/close
    JOINT_POS = 2  # 7 x joint delta position + gripper open/close


def pprint_data_mixture(
    dataset_kwargs_list: List[Dict[str, Any]], dataset_weights: List[int]
) -> None:
    print(
        "\n######################################################################################"
    )
    print(
        f"# Loading the following {len(dataset_kwargs_list)} datasets (incl. sampling weight):{'': >24} #"
    )
    for dataset_kwargs, weight in zip(dataset_kwargs_list, dataset_weights):
        pad = 80 - len(dataset_kwargs["name"])
        print(f"# {dataset_kwargs['name']}: {weight:=>{pad}f} #")
    print(
        "######################################################################################\n"
    )


def get_dataset_statistics(
    builder: DatasetBuilder,
    state_obs_keys: List[str],
    restructure_fn: Callable,
    transform_fn: Callable,
) -> dict:
    """Either computes the statistics of a dataset or loads them from a cache file if this function
    has been called before with the same arguments. Currently, the statistics include the
    min/max/mean/std of the actions and proprio as well as the number of transitions and
    trajectories in the dataset.
    """
    # compute a hash of the dataset info, state observation keys, and transform function
    # to determine the name of the cache file
    data_info_hash = hashlib.sha256(
        (
            str(builder.info)
            + str(state_obs_keys)
            + str(inspect.getsource(restructure_fn))
            + str(inspect.getsource(transform_fn))
        ).encode("utf-8")
    ).hexdigest()
    path = tf.io.gfile.join(
        builder.info.data_dir, f"dataset_statistics_{data_info_hash}.json"
    )
    # fallback local path for when data_dir is not writable
    local_path = os.path.expanduser(
        os.path.join(
            "~",
            ".cache",
            "orca",
            builder.name,
            f"dataset_statistics_{data_info_hash}.json",
        )
    )

    # check if cache file exists and load
    if tf.io.gfile.exists(path):
        logging.info(f"Loading existing dataset statistics from {path}.")
        with tf.io.gfile.GFile(path, "r") as f:
            metadata = json.load(f)
        return metadata

    if os.path.exists(local_path):
        logging.info(f"Loading existing dataset statistics from {local_path}.")
        with open(local_path, "r") as f:
            metadata = json.load(f)
        return metadata

    if "val" not in builder.info.splits:
        split = "train[:95%]"
    else:
        split = "train"
    dataset = (
        dl.DLataset.from_rlds(builder, split=split, shuffle=False)
        .map(restructure_fn)
        .map(
            lambda traj: {
                "action": traj["action"],
                "proprio": traj["observation"]["proprio"],
            }
        )
    )
    logging.info(
        f"Computing dataset statistics for {builder.name}. This may take awhile, "
        "but should only need to happen once."
    )
    actions = []
    proprios = []
    num_transitions = 0
    num_trajectories = 0
    for traj in tqdm.tqdm(
        dataset.iterator(), total=builder.info.splits["train"].num_examples
    ):
        actions.append(traj["action"])
        proprios.append(traj["proprio"])
        num_transitions += traj["action"].shape[0]
        num_trajectories += 1
    actions = np.concatenate(actions)
    proprios = np.concatenate(proprios)
    metadata = {
        "action": {
            "mean": actions.mean(0).tolist(),
            "std": actions.std(0).tolist(),
            "max": actions.max(0).tolist(),
            "min": actions.min(0).tolist(),
        },
        "proprio": {
            "mean": proprios.mean(0).tolist(),
            "std": proprios.std(0).tolist(),
            "max": proprios.max(0).tolist(),
            "min": proprios.min(0).tolist(),
        },
        "num_transitions": num_transitions,
        "num_trajectories": num_trajectories,
    }

    try:
        with tf.io.gfile.GFile(path, "w") as f:
            json.dump(metadata, f)
    except tf.errors.PermissionDeniedError:
        logging.warning(
            f"Could not write dataset statistics to {path}. "
            f"Writing to {local_path} instead."
        )
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "w") as f:
            json.dump(metadata, f)

    return metadata


def normalize_action_and_proprio(traj, metadata, normalization_type):
    # maps keys of `metadata` to corresponding keys in `traj`
    keys_to_normalize = {
        "action": "action",
        "proprio": "observation/proprio",
    }
    if normalization_type == "normal":
        # normalize to mean 0, std 1
        for key, traj_key in keys_to_normalize.items():
            traj = dl.transforms.selective_tree_map(
                traj,
                match=traj_key,
                map_fn=lambda x: (x - metadata[key]["mean"])
                / (metadata[key]["std"] + 1e-8),
            )
        return traj

    if normalization_type == "bounds":
        # normalize to [-1, 1]
        for key, traj_key in keys_to_normalize.items():
            traj = dl.transforms.selective_tree_map(
                traj,
                match=traj_key,
                map_fn=lambda x: tf.clip_by_value(
                    2
                    * (x - metadata[key]["min"])
                    / (metadata[key]["max"] - metadata[key]["min"] + 1e-8)
                    - 1,
                    -1,
                    1,
                ),
            )
        return traj

    raise ValueError(f"Unknown normalization type {normalization_type}")