import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from functools import cache
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from desilofhe import Engine
from safetensors.torch import load_file as load_safetensor

from .model_encoder import (
    encode_b,
    encode_b_cls,
    encode_b_pooler,
    encode_w_att,
    encode_w_cls,
    encode_w_ff,
    encode_w_pooler,
    encode_w_qkv,
    get_classifier_prefix,
    get_weight_array,
)
from .paths import get_light_plaintext_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./finetuned_models/mrpc/model.safetensors")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--compact", action="store_true", help="Use a memory-optimized execution mode with reduced memory footprint."
    )
    return parser.parse_args()


def write_light_plaintext_to_file(engine, messages, level, path):
    print("file path:", path, "shape:", messages.shape, "level:", level)
    for index, message in np.ndenumerate(messages):
        postfix = "_".join(map(str, index))
        light_plaintext = engine.encode_to_light_plaintext(message, level)
        engine.write_light_plaintext(light_plaintext, str(path) + postfix)


def pre_encode_masks(engine, light_plaintext_path):
    path = light_plaintext_path / "masks"
    path.mkdir(parents=True, exist_ok=True)

    ccmm_path = path / "ccmm"
    transpose_path = path / "transpose"
    rotate_internal_path = path / "rotate_internal"

    for index in range(4):
        (ccmm_path / str(index)).mkdir(parents=True, exist_ok=True)
        (transpose_path / str(index)).mkdir(parents=True, exist_ok=True)
    for name in ("attention", "block_diag_1", "block_diag_2"):
        (rotate_internal_path / name).mkdir(parents=True, exist_ok=True)

    array = np.full((2**15,), 1)
    array[np.arange(2**15) % (2**12) >= 2**11] = 0
    engine.write_light_plaintext(engine.encode_to_light_plaintext(array), str(path / "make_copies_0"))

    array = np.full((2**15,), 1)
    array[np.arange(2**15) % (2**12) < 2**11] = 0
    engine.write_light_plaintext(engine.encode_to_light_plaintext(array), str(path / "make_copies_1"))

    array = np.ones((2**15,), dtype=int)
    array[np.arange(2**15) % 16 < 6] = 0
    engine.write_light_plaintext(engine.encode_to_light_plaintext(array), str(path / "attention_dense"))

    array = np.full((2**15,), 1, dtype=int)
    array[np.arange(2**15) % 16 >= 6] = 0
    engine.write_light_plaintext(engine.encode_to_light_plaintext(array), str(path / "intermediate_dense"))

    array = np.zeros((2**15,), dtype=int)
    array[np.arange(2**15) % (2**11) < 6] = 1
    engine.write_light_plaintext(engine.encode_to_light_plaintext(array), str(path / "pooler_dense"))

    for i in range(1, 128):
        array = np.ones((2**15,), dtype=int)
        array[np.arange(2**15) % (2**11) >= 16 * i] = 0
        engine.write_light_plaintext(
            engine.encode_to_light_plaintext(array),
            str(rotate_internal_path / "attention" / str(i)),
        )

    for i in range(1, 16):
        array = np.ones((2**15,), dtype=int)
        array[np.arange(2**15) % 16 >= i] = 0
        engine.write_light_plaintext(
            engine.encode_to_light_plaintext(array),
            str(rotate_internal_path / "block_diag_1" / str(i)),
        )

    for i in range(1, 8):
        array = np.ones((2**15,), dtype=int)
        array[np.arange(2**15) % 8 >= i] = 0
        engine.write_light_plaintext(
            engine.encode_to_light_plaintext(array),
            str(rotate_internal_path / "block_diag_2" / str(i)),
        )

    for i in range(8):
        array = np.zeros((2**15,), dtype=int)
        array[2**12 * i : 2**12 * (i + 1)] = 1
        engine.write_light_plaintext(
            engine.encode_to_light_plaintext(array * (1 / 4)),
            str(path / f"make_copies_2_{i}"),
        )

    for i in range(4):
        diag_index = 16 * i
        arr0 = np.array([1] * 16 * (64 + (diag_index - 16) % 64 + 16))
        arr1 = np.array(
            [0] * (2**15 - 16 * (64 - ((diag_index - 16) % 64 + 16))) + [1] * 16 * (64 - ((diag_index - 16) % 64 + 16))
        )
        engine.write_light_plaintext(engine.encode_to_light_plaintext(arr0), str(transpose_path / "0" / str(i)))
        engine.write_light_plaintext(engine.encode_to_light_plaintext(arr1), str(transpose_path / "1" / str(i)))
        for j in range(16 * i + 1, 16 * (i + 1)):
            l = 64 - j  # noqa: E741
            arr2 = np.array([0] * 2**11 * (16 - j % 16) + [1] * (128 - l) * 2**4)
            arr3 = np.array([0] * 2**11 * (16 - j % 16 - 1) + [0] * (128 - l) * 2**4 + [1] * 16 * l)
            engine.write_light_plaintext(engine.encode_to_light_plaintext(arr2), str(transpose_path / "2" / str(j)))
            engine.write_light_plaintext(engine.encode_to_light_plaintext(arr3), str(transpose_path / "3" / str(j)))

    for n in range(1, 128):
        rot = n
        j = n % 16
        arr0 = np.full((2**15,), 1, dtype=float)
        arr0[np.arange(2**15) % (2**11) >= (2**11 - 16 * rot)] = 0

        arr1 = np.full((2**15,), 0, dtype=float)
        arr1[np.arange(2**15) % (2**11) >= (2**11 - 16 * rot)] = 1

        if j == 0:
            engine.write_light_plaintext(engine.encode_to_light_plaintext(arr0), str(ccmm_path / "0" / str(n)))
            engine.write_light_plaintext(engine.encode_to_light_plaintext(arr1), str(ccmm_path / "1" / str(n)))
        else:
            arr0[: (2**11) * j] = 0
            engine.write_light_plaintext(engine.encode_to_light_plaintext(arr0), str(ccmm_path / "0" / str(n)))

            arr1[-(2**11) :] = 0
            if j > 1:
                arr1[: (2**11) * (j - 1)] = 0
            engine.write_light_plaintext(engine.encode_to_light_plaintext(arr1), str(ccmm_path / "1" / str(n)))

            arr2 = np.full((2**15,), 1, dtype=float)
            arr2[np.arange(2**15) % (2**11) >= (2**11 - 16 * rot)] = 0
            arr2[(2**11) * j :] = 0
            engine.write_light_plaintext(engine.encode_to_light_plaintext(arr2), str(ccmm_path / "2" / str(n)))

            arr3 = np.full((2**15,), 1, dtype=float)
            arr3 = arr3 - arr0 - arr1 - arr2
            engine.write_light_plaintext(engine.encode_to_light_plaintext(arr3), str(ccmm_path / "3" / str(n)))


def pre_encode_stage_03(engine, weights, layer_index, light_plaintext_path):
    level = 8 if layer_index == 0 else 11

    bert_prefix = f"bert.encoder.layer.{layer_index}.attention.self"

    query_weight = encode_w_qkv(get_weight_array(weights, f"{bert_prefix}.query.weight"))
    query_bias = encode_b(get_weight_array(weights, f"{bert_prefix}.query.bias"), n_blocks=12, n_out=64)

    path = light_plaintext_path / "stage_03" / f"layer_{layer_index}"
    path.mkdir(parents=True, exist_ok=True)

    write_light_plaintext_to_file(engine, query_weight, level, path / "w_")
    write_light_plaintext_to_file(engine, query_bias, level - 1, path / "b_")


def pre_encode_stage_04(engine, weights, layer_index, light_plaintext_path):
    level = 8 if layer_index == 0 else 11

    bert_prefix = f"bert.encoder.layer.{layer_index}.attention.self"

    softmax_scale = 1 / 1024 if layer_index == 2 else 1 / 512
    key_weight = encode_w_qkv(get_weight_array(weights, f"{bert_prefix}.key.weight"), scale=softmax_scale)
    key_bias = encode_b(
        get_weight_array(weights, f"{bert_prefix}.key.bias"), n_blocks=12, n_out=64, scale=softmax_scale
    )

    path = light_plaintext_path / "stage_04" / f"layer_{layer_index}"
    path.mkdir(parents=True, exist_ok=True)

    write_light_plaintext_to_file(engine, key_weight, level, path / "w_")
    write_light_plaintext_to_file(engine, key_bias, level - 1, path / "b_")


def pre_encode_stage_05(engine, weights, layer_index, light_plaintext_path):
    level = 8 if layer_index == 0 else 11

    bert_prefix = f"bert.encoder.layer.{layer_index}.attention.self"

    value_weight = encode_w_qkv(get_weight_array(weights, f"{bert_prefix}.value.weight"))
    value_bias = encode_b(get_weight_array(weights, f"{bert_prefix}.value.bias"), n_blocks=12, n_out=64)

    path = light_plaintext_path / "stage_05" / f"layer_{layer_index}"
    path.mkdir(parents=True, exist_ok=True)

    write_light_plaintext_to_file(engine, value_weight, level, path / "w_")
    write_light_plaintext_to_file(engine, value_bias, level - 1, path / "b_")


def pre_encode_stage_10(engine, weights, layer_index, light_plaintext_path):
    level = 14

    bert_prefix = f"bert.encoder.layer.{layer_index}.attention.output.dense"
    weight = encode_w_att(
        get_weight_array(weights, f"{bert_prefix}.weight"), n_in=64, n_out=128, block_shape=(128, 64)
    )
    bias = encode_b(get_weight_array(weights, f"{bert_prefix}.bias"), n_blocks=6, n_out=128)

    path = light_plaintext_path / "stage_10" / f"layer_{layer_index}"
    path.mkdir(parents=True, exist_ok=True)

    write_light_plaintext_to_file(engine, weight, level, path / "w_")
    write_light_plaintext_to_file(engine, bias, level - 1, path / "b_")


def pre_encode_stage_11(engine, weights, layer_index, light_plaintext_path):
    level = 8 if layer_index == 0 else 11

    bert_prefix = f"bert.encoder.layer.{layer_index}.attention.output.LayerNorm"
    weight = encode_b(get_weight_array(weights, f"{bert_prefix}.weight"), n_blocks=6, n_out=128)
    bias = encode_b(get_weight_array(weights, f"{bert_prefix}.bias"), n_blocks=6, n_out=128)

    path = light_plaintext_path / "stage_11" / f"layer_{layer_index}"
    path.mkdir(parents=True, exist_ok=True)

    write_light_plaintext_to_file(engine, weight, level, path / "w_")
    write_light_plaintext_to_file(engine, bias, level - 1, path / "b_")


def pre_encode_stage_12(engine, weights, layer_index, light_plaintext_path):
    level = 4 if layer_index == 0 else 7

    bert_prefix = f"bert.encoder.layer.{layer_index}.intermediate.dense"
    weight = encode_w_ff(
        get_weight_array(weights, f"{bert_prefix}.weight"),
        n_in=128,
        n_out=128,
        block_shape=(128, 128),
        vsplit=4,
        scale=1 / 64,
    )
    bias_chunks = np.split(get_weight_array(weights, f"{bert_prefix}.bias"), 2)
    bias = np.full((2, 8), None, dtype=object)
    for rep in range(2):
        bias[rep] = encode_b(
            bias_chunks[rep],
            n_blocks=12,
            n_out=128,
            pad_index=(6, 7, 14, 15),
            scale=1 / 64,
        )

    path = light_plaintext_path / "stage_12" / f"layer_{layer_index}"
    path.mkdir(parents=True, exist_ok=True)

    write_light_plaintext_to_file(engine, weight, level, path / "w_")
    write_light_plaintext_to_file(engine, bias, level - 1, path / "b_")


def pre_encode_stage_14(engine, weights, layer_index, light_plaintext_path):
    level = 3

    bert_prefix = f"bert.encoder.layer.{layer_index}.output.dense"
    weight = encode_w_ff(
        get_weight_array(weights, f"{bert_prefix}.weight"),
        n_in=128,
        n_out=128,
        block_shape=(128, 128),
        hsplit=4,
    )
    bias = encode_b(get_weight_array(weights, f"{bert_prefix}.bias"), n_blocks=6, n_out=128)

    path = light_plaintext_path / "stage_14" / f"layer_{layer_index}"
    path.mkdir(parents=True, exist_ok=True)

    write_light_plaintext_to_file(engine, weight, level, path / "w_")
    write_light_plaintext_to_file(engine, bias, level - 1, path / "b_")


def pre_encode_stage_16(engine, weights, layer_index, light_plaintext_path):
    level = 14

    bert_prefix = f"bert.encoder.layer.{layer_index}.output.LayerNorm"
    weight = encode_b(get_weight_array(weights, f"{bert_prefix}.weight"), n_blocks=6, n_out=128)
    bias = encode_b(get_weight_array(weights, f"{bert_prefix}.bias"), n_blocks=6, n_out=128)

    path = light_plaintext_path / "stage_16" / f"layer_{layer_index}"
    path.mkdir(parents=True, exist_ok=True)

    write_light_plaintext_to_file(engine, weight, level, path / "w_")
    write_light_plaintext_to_file(engine, bias, level - 1, path / "b_")


def pre_encode_stage_17(engine, weights, light_plaintext_path):
    level = 14

    bert_prefix = "bert.pooler.dense"
    weight = encode_w_pooler(get_weight_array(weights, f"{bert_prefix}.weight"))
    bias = encode_b_pooler(get_weight_array(weights, f"{bert_prefix}.bias"), n_blocks=6)

    path = light_plaintext_path / "stage_17"
    path.mkdir(parents=True, exist_ok=True)

    write_light_plaintext_to_file(engine, weight, level, path / "w_")
    write_light_plaintext_to_file(engine, bias, level - 1, path / "b_")


def pre_encode_stage_18(engine, weights, light_plaintext_path):
    level = 14

    classifier_prefix = get_classifier_prefix(weights)
    weight = encode_w_cls(get_weight_array(weights, f"{classifier_prefix}.weight"))
    bias = encode_b_cls(get_weight_array(weights, f"{classifier_prefix}.bias"))

    path = light_plaintext_path / "stage_18"
    path.mkdir(parents=True, exist_ok=True)

    write_light_plaintext_to_file(engine, weight, level, path / "w_")
    write_light_plaintext_to_file(engine, bias, level - 1, path / "b_")


def get_worker_count(requested_workers):
    if requested_workers is not None:
        return max(1, requested_workers)
    return min(8, os.cpu_count() or 1)


def get_jobs():
    jobs = []
    jobs.append(("mask", None))
    for layer_index in range(12):
        jobs.append(("attention", layer_index))
        jobs.append(("feed_forward_12", layer_index))
        jobs.append(("feed_forward_14", layer_index))
        jobs.append(("feed_forward_16", layer_index))
    jobs.append(("pooler", None))
    jobs.append(("classifier", None))
    return jobs


@cache
def get_weights(model_path):
    return load_safetensor(model_path)


def run_job(job, model_path, light_plaintext_path, compact):
    job_type, layer_index = job
    weights = get_weights(model_path)

    if compact:
        engine = Engine(use_bootstrap_to_17_levels=True, compact=True)
    else:
        engine = Engine(use_bootstrap_to_14_levels=True, compact=False)

    if job_type == "mask":
        pre_encode_masks(engine, light_plaintext_path)
        return

    if job_type == "attention":
        pre_encode_stage_03(engine, weights, layer_index, light_plaintext_path)
        pre_encode_stage_04(engine, weights, layer_index, light_plaintext_path)
        pre_encode_stage_05(engine, weights, layer_index, light_plaintext_path)
        pre_encode_stage_10(engine, weights, layer_index, light_plaintext_path)
        pre_encode_stage_11(engine, weights, layer_index, light_plaintext_path)
        return

    if job_type == "feed_forward_12":
        pre_encode_stage_12(engine, weights, layer_index, light_plaintext_path)
        return

    if job_type == "feed_forward_14":
        pre_encode_stage_14(engine, weights, layer_index, light_plaintext_path)
        return

    if job_type == "feed_forward_16":
        pre_encode_stage_16(engine, weights, layer_index, light_plaintext_path)
        return

    if job_type == "pooler":
        pre_encode_stage_17(engine, weights, light_plaintext_path)
        return

    if job_type == "classifier":
        pre_encode_stage_18(engine, weights, light_plaintext_path)
        return

    raise ValueError(f"Unknown job type: {job_type}")


def main():
    args = parse_args()

    model_path = Path(args.model_path)
    compact = args.compact
    light_plaintext_path = get_light_plaintext_path(compact)
    worker_count = get_worker_count(args.workers)
    jobs = get_jobs()

    if worker_count == 1:
        for job in jobs:
            run_job(job, model_path, light_plaintext_path, compact)
        return

    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=get_context("spawn"),
    ) as executor:
        futures = [executor.submit(run_job, job, model_path, light_plaintext_path, compact) for job in jobs]
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
