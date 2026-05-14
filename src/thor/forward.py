import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from transformers import BertForNextSentencePrediction

from .data_encoder import DataEncoder
from .he import HE
from .timer import Timer
from .utils import load_model, matrix_ld


def parse_args():
    parser = argparse.ArgumentParser(description="Run the forward pass from forward.ipynb as a script.")
    parser.add_argument("--dataset-type", default="mrpc", help="Dataset type used by the notebook.")
    parser.add_argument("--target-idx", type=int, default=0, help="Validation sample index to run.")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index.")
    parser.add_argument(
        "--compact", action="store_true", help="Use a memory-optimized execution mode with reduced memory footprint."
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip the per-layer comparison plots.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save plots and the result JSON for this run.",
    )
    parser.add_argument("--print-rotate-levels", action="store_true", help="Print rotate levels.")
    return parser.parse_args()


def load_encrypted_input(dataset_type, target_idx, he):
    data_encryptor = DataEncoder(
        dataset_type,
        embedding_model=BertForNextSentencePrediction.from_pretrained("bert-base-uncased").bert.embeddings,
        he=he,
    )
    data_loader = data_encryptor.eval_dataloader

    idx = 0
    for batch in data_loader:
        if idx < target_idx:
            idx += 1
            continue
        if idx == target_idx:
            data = dict((k, v) for k, v in batch.items() if k in ["input_ids", "token_type_ids"])
            embedding = data_encryptor.embed_data(data)
            x = data_encryptor.encrypt_embedding(embedding, level=9)
            attention_mask = batch["attention_mask"]
            thor_attention_mask, clear_attention_mask = data_encryptor.encode_attention_mask(
                attention_mask.cpu().numpy().squeeze().T,
                level=14,
            )
            return (
                data_loader,
                x,
                attention_mask,
                thor_attention_mask,
                clear_attention_mask,
            )

    raise IndexError(f"target_idx={target_idx} is out of range for the evaluation dataloader")


def load_plain_reference(dataset_type, target_idx, data_loader):
    model_plain = load_model(dataset_type, f"./finetuned_models/{dataset_type}/model.safetensors")
    model_plain.eval()
    device = torch.device("cpu")
    model_plain.to(device)

    idx = 0
    for batch in data_loader:
        if idx < target_idx:
            idx += 1
            continue
        if idx == target_idx:
            batch = dict((k, v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items())
            with torch.no_grad():
                outputs = model_plain(**batch)
                plain_logits = outputs.logits.squeeze()
                plain_pred = plain_logits.argmax().item()
                plain_probs = torch.softmax(plain_logits, dim=0).numpy()

            print("=== Final Prediction ===")
            print(f"  Plain PyTorch logits : {plain_logits.tolist()}")
            print(f"  Plain PyTorch pred   : {plain_pred}")
            print(f"  Plain PyTorch probs  : {plain_probs.tolist()}")
            return model_plain, outputs, device, plain_logits.tolist(), plain_pred

    raise IndexError(f"target_idx={target_idx} is out of range for the plain model reference run")


def get_nonlinear_reference(model_plain, outputs, attention_mask, device):
    def get_nonlinear_in_out(hidden_states, layer_idx):
        with torch.no_grad():
            bert_layer_m = model_plain.bert.encoder.layer[layer_idx]
            attention_m = bert_layer_m.attention.self
            bert_output_m = model_plain.bert.encoder.layer[layer_idx].attention.output

            def project_to_heads(x):
                new_shape = x.size()[:-1] + (attention_m.num_attention_heads, attention_m.attention_head_size)
                return x.view(*new_shape).transpose(1, 2)

            q = project_to_heads(attention_m.query(hidden_states))
            k = project_to_heads(attention_m.key(hidden_states))
            v = project_to_heads(attention_m.value(hidden_states))
            attention_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(attention_m.attention_head_size)
            extended_att_mask = model_plain.get_extended_attention_mask(attention_mask, 768).to(device)
            sftmx_in = attention_scores + extended_att_mask
            att_probs_m = torch.nn.functional.softmax(sftmx_in, dim=-1)
            sftmx_out = att_probs_m
            att_context_m = torch.matmul(att_probs_m, v)
            context_layer = att_context_m.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (attention_m.all_head_size,)
            context_layer = context_layer.view(new_context_layer_shape)
            att_dense = bert_output_m.dense(context_layer)
            ln1_in = att_dense + hidden_states
            layernorm_1_output = bert_output_m.LayerNorm(ln1_in)
            intermediate_dense = bert_layer_m.intermediate.dense(layernorm_1_output)
            gelu_out = bert_layer_m.intermediate.intermediate_act_fn(intermediate_dense)
            output_dense = bert_layer_m.output.dense(gelu_out)
            layernorm_2_input = output_dense + layernorm_1_output
            layernorm_2_output = bert_layer_m.output.LayerNorm(layernorm_2_input)
            pooler_m = model_plain.bert.pooler
            pooler_dense_output = pooler_m.dense(layernorm_2_output[:, 0])
            pooler_output = pooler_m.activation(pooler_dense_output)

        return (
            hidden_states.cpu().numpy().squeeze(),
            q.cpu().numpy().squeeze(),
            sftmx_in.cpu().numpy().squeeze(),
            sftmx_out.cpu().numpy().squeeze(),
            att_context_m.cpu().numpy().squeeze(),
            att_dense.cpu().numpy().squeeze(),
            layernorm_1_output.cpu().numpy().squeeze(),
            intermediate_dense.cpu().numpy().squeeze(),
            gelu_out.cpu().numpy().squeeze(),
            output_dense.cpu().numpy().squeeze(),
            layernorm_2_input.cpu().numpy().squeeze(),
            layernorm_2_output.cpu().numpy().squeeze(),
            pooler_dense_output.cpu().numpy().squeeze(),
            pooler_output.cpu().numpy().squeeze(),
        )

    hidden_states = []
    qs = []
    sftmx_ins = []
    sftmx_outs = []
    att_contexts = []
    att_denses = []
    layernorm_1_outputs = []
    intermediate_denses = []
    gelu_outs = []
    output_denses = []
    layernorm_2_inputs = []
    layernorm_2_outputs = []

    for layer in range(12):
        (
            hidden_state,
            q,
            sftmx_in,
            sftmx_out,
            att_context,
            att_dense,
            layernorm_1_output,
            intermediate_dense,
            gelu_out,
            output_dense,
            layernorm_2_input,
            layernorm_2_output,
            _pooler_dense_out,
            _pooler_out,
        ) = get_nonlinear_in_out(outputs.hidden_states[layer], layer)
        hidden_states.append(hidden_state)
        qs.append(q)
        sftmx_ins.append(sftmx_in)
        sftmx_outs.append(sftmx_out)
        att_contexts.append(att_context)
        att_denses.append(att_dense)
        layernorm_1_outputs.append(layernorm_1_output)
        intermediate_denses.append(intermediate_dense)
        gelu_outs.append(gelu_out)
        output_denses.append(output_dense)
        layernorm_2_inputs.append(layernorm_2_input)
        layernorm_2_outputs.append(layernorm_2_output)

    return dict(
        hidden_states=hidden_states,
        qs=qs,
        sftmx_ins=sftmx_ins,
        sftmx_outs=sftmx_outs,
        att_contexts=att_contexts,
        att_denses=att_denses,
        layernorm_1_outputs=layernorm_1_outputs,
        intermediate_denses=intermediate_denses,
        gelu_outs=gelu_outs,
        output_denses=output_denses,
        layernorm_2_inputs=layernorm_2_inputs,
        layernorm_2_outputs=layernorm_2_outputs,
    )


def forward_layer(x, layer_idx, clear_attention_mask, he):
    print("layer_idx:", layer_idx)
    print("now:", datetime.now())

    timer = he.timer
    timer.print_legend()

    with timer.stage(1, "complexify x"):
        x, x_cplx = he.stage_01_complexify_x(x, layer_idx)

    with timer.stage(2, "make rotated copies"):
        x_cplx_rots = he.stage_02_make_rotated_copies(x_cplx)

    with timer.stage(3, "query"):
        q_wo_rescale = he.stage_03_query(x_cplx_rots, layer_idx)

    with timer.stage(4, "key"):
        k = he.stage_04_key(x_cplx_rots, layer_idx)

    with timer.stage(5, "value"):
        v = he.stage_05_value(x_cplx_rots, layer_idx)

    with timer.stage(6, "attention score"):
        sftmx_in = he.stage_06_attention_score(q_wo_rescale, k)

    with timer.stage(7, "softmax"):
        sftmx_out = he.stage_07_softmax(sftmx_in, clear_attention_mask, layer_idx)

    with timer.stage(8, "attention context"):
        att_context = he.stage_08_attention_context(v, sftmx_out)

    with timer.stage(9, "make rotated copies"):
        att_context_rots = he.stage_02_make_rotated_copies(att_context)

    with timer.stage(10, "dense output"):
        att_dense = he.stage_10_attention_dense(att_context_rots, layer_idx)

    with timer.stage(11, "layernorm 1"):
        layernorm_1_output = he.stage_11_attention_layernorm(x, att_dense, layer_idx)

    with timer.stage(12, "intermediate dense"):
        intermediate_dense = he.stage_12_intermediate_dense(layernorm_1_output, layer_idx)

    with timer.stage(13, "gelu"):
        gelu_output = he.stage_13_gelu(intermediate_dense)

    with timer.stage(14, "output dense"):
        output_dense = he.stage_14_output_dense(gelu_output, layer_idx)

    with timer.stage(15, "prepare layernorm 2"):
        layernorm_2_input = he.stage_15_prepare_layernorm(layernorm_1_output, output_dense)

    with timer.stage(16, "layernorm 2"):
        layernorm_2_output = he.stage_16_output_layernorm(layernorm_2_input, layer_idx)

    return layernorm_2_output, (
        x,
        q_wo_rescale,
        sftmx_in,
        sftmx_out,
        att_context,
        att_dense,
        layernorm_1_output,
        intermediate_dense,
        gelu_output,
        output_dense,
        layernorm_2_input,
        layernorm_2_output,
    )


def plot_variables(variables, plain_reference, he, layer_idx, output_dir, i=0, j=0, h=0):
    variable_names = [
        "x",
        "q",
        "sftmx_in",
        "sftmx_out",
        "att_context",
        "att_dense",
        "layernorm_1_output",
        "intermediate_dense",
        "gelu_out",
        "output_dense",
        "layernorm_2_input",
        "layernorm_2_output",
    ]
    global_vars = [
        plain_reference["hidden_states"],
        plain_reference["qs"],
        plain_reference["sftmx_ins"],
        plain_reference["sftmx_outs"],
        plain_reference["att_contexts"],
        plain_reference["att_denses"],
        plain_reference["layernorm_1_outputs"],
        plain_reference["intermediate_denses"],
        plain_reference["gelu_outs"],
        plain_reference["output_denses"],
        plain_reference["layernorm_2_inputs"],
        plain_reference["layernorm_2_outputs"],
    ]
    h_indices = [np.where(np.arange(0, 2**11) % 16 == head) for head in range(12)]

    fig, axs = plt.subplots(4, 3, figsize=(15, 15))
    fig.suptitle(f"Variables Plot (Layer {layer_idx})", fontsize=16)

    for index, (var, name, global_var) in enumerate(zip(variables, variable_names, global_vars)):
        row = index // 3
        col = index % 3

        if isinstance(var, np.ndarray) and var.ndim > 1:
            var = var[0]

        if len(var) <= i:
            print(f"{name} is not available: shape is {len(var)}")
            continue

        current_var = he.decrypt(var[i])[2**11 * j : 2**11 * (j + 1)][h_indices[h]]
        global_var = global_var[layer_idx]

        if global_var.ndim == 3:
            global_var = global_var[h].T
        elif name in ["intermediate_dense", "gelu_out"]:
            global_var = np.vsplit(global_var.T, 24)[0]
        else:
            global_var = np.vsplit(global_var.T, 6)[h]

        global_var_layer = matrix_ld(global_var, i * 16 + j)

        if name == "sftmx_in":
            global_var_layer = global_var_layer[:40]
            current_var = current_var[:40]
            current_var = current_var * (128 if layer_idx == 2 else 64)
        elif name == "intermediate_dense":
            current_var *= 64
        elif name == "layernorm_2_input":
            current_var /= 2

        mse = np.mean((current_var - global_var_layer) ** 2)
        print(f"{name} MSE: {mse:.4e}")
        axs[row, col].plot(current_var, label=f"HE {name}")
        axs[row, col].plot(global_var_layer, label=f"Plain {name}", linestyle="--")
        axs[row, col].set_title(f"{name} (MSE: {mse:.4e})")
        axs[row, col].grid(True)
        axs[row, col].legend()

    for ax in axs.flat:
        ax.set(xlabel="Index", ylabel="Decoded Value")

    axs[-1, -1].axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / f"layer-{layer_idx:02d}.png")
    plt.close(fig)


def run_forward(args):
    print(args)

    compact = args.compact
    if compact:
        key_size = "medium"
    else:
        key_size = "large"

    output_dir = Path(args.output_dir) if args.output_dir is not None else Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)

    timer = Timer()
    with timer.setup():
        he = HE(args.device, compact, key_size, timer)
        data_loader, x, attention_mask, thor_attention_mask, clear_attention_mask = load_encrypted_input(
            args.dataset_type,
            args.target_idx,
            he,
        )
        model_plain, outputs, device, plain_logits, plain_pred = load_plain_reference(
            args.dataset_type, args.target_idx, data_loader
        )
        plain_reference = get_nonlinear_reference(model_plain, outputs, attention_mask, device)

    for layer_idx in range(12):
        with timer.layer(layer_idx):
            x, variables = forward_layer(x, layer_idx, clear_attention_mask, he)
        if not args.no_plots:
            with timer.paused():
                plot_variables(variables, plain_reference, he, layer_idx, output_dir)

    print("all layers done")
    print("now:", datetime.now())

    timer.print_legend()

    with timer.stage(17, "pooler"):
        x = he.stage_17_pooler(x)

    with timer.stage(18, "classifier"):
        x = he.stage_18_classifier(x)

    dataset = load_dataset("glue", args.dataset_type)
    val_set = dataset["validation"]

    a = he.decrypt(x[0])[0]
    b = he.decrypt(x[1])[0]
    pred = 0 if a > b else 1
    label = val_set["label"][args.target_idx]
    print(f"Predicted by HE: {pred}, Ground Truth: {label}")
    print(f"HE A [{a}] B [{b}]")
    print(f"PT A [{plain_logits[0]}] B [{plain_logits[1]}]")

    print("now:", datetime.now())
    timer.print_legend()

    if args.print_rotate_levels:
        for delta, level in he.rotate_levels.items():
            print(f"Rotate delta {delta} max level {level}")

        print("[")
        for delta, level in he.rotate_levels.items():
            print(f"({delta}, {level}),")
        print("]")

    result = dict(
        dataset_type=args.dataset_type,
        target_idx=args.target_idx,
        device=args.device,
        compact=compact,
        key_size=key_size,
        pred=pred,
        plain_pred=plain_pred,
        label=label,
        he_logits=[a, b],
        plain_logits=plain_logits,
    )
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    args = parse_args()
    run_forward(args)


if __name__ == "__main__":
    main()
