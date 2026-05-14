# THOR

THOR is a secure Transformer inference framework that uses homomorphic encryption to run a BERT sequence-classification forward pass over encrypted data.
It is built on the [DESILO FHE library](https://fhe.desilo.dev/latest).

The repository exposes three CLI commands:

- `encode_weights`: generates encoded plaintext weights and masks under `light_plaintexts/`
- `forward`: runs one validation example and compares the result with the plain PyTorch model
- `forward_batch`: runs `forward` for a range of validation indices across one or more GPUs

## Quickstart

### Requirements

- desilofhe 1.13+ (CUDA version 12.1 to 13.0).
- GPU with at least 36 GB of VRAM (the default), or 32 GB when using the `--compact` flag.

### 1. Installation

THOR works with Python 3.14+, and any standard Python package manager can be used; the examples below use Poetry.

```bash
poetry install
```

### 2. Prepare Files

Before running the encrypted forward pass, you need to prepare the following files:

2-1. A fine-tuned BERT checkpoint file including `model.safetensors`

Download `finetuned_models.tar` [Google Drive Link](https://drive.google.com/file/d/1gC6-C2bCf-bEALhQtzk2s3MGSuHpcGAh), then extract the file into the repository root.
The code also loads `bert-base-uncased` components from Hugging Face at runtime.

2-2. Encoded checkpoint files under `light_plaintexts/`, generated from model weights

`encode_weights` writes the weights, biases, and masks used by the encrypted forward pass.
To generate the light plaintexts for the model:

```bash
poetry run encode_weights \
  --model_path ./finetuned_models/mrpc/model.safetensors
```

This writes default-mode files to `./light_plaintexts/default/`. With `--compact`, it writes compact-mode files to `./light_plaintexts/compact/`.

2-3. (Recommended) Cache the encoded files for your selected mode with `vmtouch -t light_plaintexts/default/` or `vmtouch -t light_plaintexts/compact/`. Each directory is around 110 GB, so make sure you have enough RAM for the mode you cache.

You can install `vmtouch` from your package manager or from source: [vmtouch](https://github.com/hoytech/vmtouch)

### 3. Forward Pass

- Single encrypted forward pass: `poetry run forward`
- With the memory-efficient engine: `poetry run forward --compact`

You can also run a batch over a range of validation indices:

```bash
poetry run forward_batch \
  --start-idx 0 \
  --end-idx 10 \
  --devices 0 1 \
  --output-dir ./forward-batch-results
```

`forward_batch` creates one subdirectory per target index and skips indices that already have results in the output directory.

### 4. (Optional) The Compact Mode

All scripts (forward, forward_batch, encode_weights) support a `--compact` flag that uses a more compact encoding for the internal data structures, which can reduce memory usage during the forward pass and enable it to run on GPUs with 32 GB of VRAM.

Note that the compact encoding is not compatible with the non-compact forward pass, so you must use the `--compact` flag for both encoding and forward steps if you choose to use it.

```bash
# With compact encoding
poetry run encode_weights --compact
poetry run forward --compact
```

## Results

Each `forward` run writes:

- `result.json`, which includes the dataset type, target index, device, key size, prediction, plain-model prediction, label, HE logits, and plain logits
- Optional per-layer plots such as `layer-00.png` through `layer-11.png`

During execution, the script also prints per-stage timing information from `thor.timer.Timer`.

### Example Output

- HE and PT denote logits from the homomorphically encrypted forward pass and the plain PyTorch model, respectively.
- `compute time` measures the core encrypted inference execution time only, while `total time` includes end-to-end overhead such as preprocessing, data transfer, and visualization.

```
Predicted by HE: 1, Ground Truth: 1
HE A [-3.007829226318608] B [5.926385952893445]
PT A [-3.12514591217041] B [6.013195514678955]
now: 2026-05-13 02:24:13.853284
----------------------------------------------------------------------------------
           stage time    compute time       total time    stage name
----------------------------------------------------------------------------------
                           6m  2.666s     11m 26.870s
----------------------------------------------------------------------------------
```

### Benchmark

The accuracy of the total run of MRPC examples is 84.07% (343/408), and the average compute time is 590.6 seconds on an NVIDIA A100-SXM4-80GB GPU.
Note that the original THOR paper reports 84.80% accuracy and 602 seconds compute time.

| Mode | CPU | GPU | Compute Time |
|---|---|---|---|
| Default | Intel Xeon Platinum 8462Y+ | NVIDIA A100-SXM4-80GB | 590.6s |
| Compact | Intel Xeon Platinum 8462Y+ | NVIDIA A100-SXM4-80GB | 650.3s |
| Compact | Intel Core i7-10700K @ 3.80GHz | NVIDIA GeForce RTX 5090 | 362.6s |

For optimal performance, cache the encoded files for your selected mode with `vmtouch -t light_plaintexts/default/` or `vmtouch -t light_plaintexts/compact/` before running the forward pass.
