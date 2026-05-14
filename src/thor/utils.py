import numpy as np
from safetensors.torch import load_file as load_safetensors
from transformers import (
    AutoModelForSequenceClassification,
    BertForNextSentencePrediction,
)


def load_model(data_type: str, model_dir: str, type: str = "default"):
    state_dict = load_safetensors(model_dir)
    if data_type in {"rte", "mrpc"}:
        model = BertForNextSentencePrediction.from_pretrained("bert-base-uncased", output_hidden_states=True)
    elif data_type == "sst2":
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", output_hidden_states=True)
    elif data_type == "stsb":
        model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=1, output_hidden_states=True
        )
    model.load_state_dict(state_dict)
    print(f"Model loaded for {data_type}")
    return model


def ld_entry(matrix: np.ndarray, l: int, i: int):  # noqa: E741
    """
    Get the i th entry of the l th lower diagonal of the matrix
    """
    b, c = matrix.shape
    return matrix[(l + i) % b, i % c]


def matrix_ld(matrix: np.ndarray, l: int):  # noqa: E741
    """
    Get the l th lower diagonal of the matrix
    """
    a, b = matrix.shape
    dim = max(a, b)
    return np.array([ld_entry(matrix, l, i) for i in range(dim)])


def to_blocks(
    matrix: np.ndarray, block_shape: tuple[int, int], diag: bool = True
) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Convert the matrix to a list of block matrices.
    Return the blocks in diagonal form if diag is True
    """
    rows, cols = matrix.shape
    block_rows, block_cols = block_shape
    if rows % block_rows != 0 or cols % block_cols != 0:
        raise ValueError("Matrix shape should be divisible by block shape")

    vertical = rows // block_rows
    horizontal = cols // block_cols
    blocks = matrix.reshape(vertical, block_rows, horizontal, block_cols).transpose(0, 2, 1, 3)

    if not diag:
        return blocks, (vertical, horizontal)

    diag_rows = min(vertical, horizontal)
    diag_cols = max(vertical, horizontal)
    diag_blocks = np.empty((diag_rows, diag_cols, block_rows, block_cols), dtype=matrix.dtype)
    diag_row_indices = np.arange(diag_rows)
    for diag_col in range(diag_cols):
        diag_blocks[:, diag_col] = blocks[(diag_row_indices + diag_col) % vertical, diag_col % horizontal]
    return diag_blocks, (diag_rows, diag_cols)
