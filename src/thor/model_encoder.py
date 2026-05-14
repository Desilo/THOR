import numpy as np

from .utils import to_blocks

SLOT_COUNT = 2**15
GROUP_SIZE = 2**11
PACK = 16
DIM = 128
DIM_RANGE = np.arange(DIM)
PACK_RANGE = np.arange(PACK)
DIM_SLOT_BASE = 16 * DIM_RANGE[:, None]
PACK_GROUP_BASE = GROUP_SIZE * PACK_RANGE[:, None, None]
ATT_SLOT_INDICES = np.arange(12)
FF_SLOT_INDICES = np.array([0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13])


def get_weight_array(weights, name):
    return weights[name].detach().cpu().numpy()


def get_classifier_prefix(weights) -> str:
    return "cls.seq_relationship" if "cls.seq_relationship.weight" in weights else "classifier"


def gather_upper_diagonal_batch(
    blocks: np.ndarray, input_indices: np.ndarray, rotations: np.ndarray, n_in: int, n_in_complex: int
) -> np.ndarray:
    offsets = rotations[:, None] + DIM_RANGE[None, :]
    row_indices = offsets % blocks.shape[1]
    real_col_indices = (input_indices[:, None] + offsets) % blocks.shape[2]
    imag_col_indices = (((input_indices + n_in_complex) % n_in)[:, None] + offsets) % blocks.shape[2]

    block_indices = np.arange(blocks.shape[0])[None, :, None]
    real = blocks[block_indices, row_indices[:, None, :], real_col_indices[:, None, :]]
    imag = blocks[block_indices, row_indices[:, None, :], imag_col_indices[:, None, :]]
    return (real - 1j * imag) / 2


def assign_packed_dense_block(msg: np.ndarray, values: np.ndarray, slot_indices: np.ndarray):
    positions = PACK_GROUP_BASE + DIM_SLOT_BASE[None, :, :] + slot_indices[None, None, :]
    msg[positions] = values.transpose(0, 2, 1)


def encode_w_att(w: np.ndarray, n_in: int, n_out: int, block_shape: tuple[int, int], scale: float = 1) -> np.ndarray:
    if w.shape[0] % block_shape[0] != 0 or w.shape[1] % block_shape[1] != 0:
        raise ValueError("Dimension does not match")
    if n_out % PACK != 0:
        raise ValueError("Number of n_out should be divisible by pack")

    n_in_complex = n_in // 2
    diag_blocks, (diag_count, _) = to_blocks(w, block_shape, diag=True)
    n_out_packed = n_out // PACK
    messages = np.full((n_out_packed, diag_count, n_in_complex), None, dtype=object)

    for diag_index in range(diag_count):
        diagonal = diag_blocks[diag_index]
        for n in range(n_in_complex):
            for out_index in range(n_out_packed):
                msg = np.zeros((SLOT_COUNT,), dtype=complex)
                rotations = out_index * 16 + PACK_RANGE
                input_indices = ((n // 16) * 16 + out_index * 16 + (n + PACK_RANGE) % 16) % n_in_complex
                input_indices = (input_indices - rotations) % n_in
                values = scale * gather_upper_diagonal_batch(diagonal, input_indices, rotations, n_in, n_in_complex)
                assign_packed_dense_block(msg, values, ATT_SLOT_INDICES)
                messages[out_index, diag_index, n] = msg
    return messages


def encode_w_qkv(w: np.ndarray, scale: float = 1) -> np.ndarray:
    if w.shape != (768, 768):
        raise ValueError("Shape of Wq, Wk, Wv matrices should be (768, 768)")
    return encode_w_att(w, n_in=128, n_out=64, block_shape=(64, 128), scale=scale)


def encode_w_ff(
    w: np.ndarray,
    n_in: int,
    n_out: int,
    block_shape: tuple[int, int],
    vsplit: int = 0,
    hsplit: int = 0,
    scale: float = 1,
) -> np.ndarray:
    n_in_complex = n_in // 2

    if w.shape[0] % block_shape[0] != 0 or w.shape[1] % block_shape[1] != 0:
        raise ValueError("Dimension does not match")
    if n_out % PACK != 0:
        raise ValueError("Number of n_out should be divisible by pack")

    if vsplit:
        weight_splits = np.vsplit(w, vsplit)
    elif hsplit:
        weight_splits = np.hsplit(w, hsplit)
    else:
        weight_splits = [w]

    diag_blocks_list = []
    for weight_split in weight_splits:
        diag_blocks, (diag_count, _) = to_blocks(weight_split, block_shape, diag=True)
        diag_blocks_list.append(diag_blocks)

    n_out_packed = n_out // PACK
    messages = np.full((2, n_out_packed, diag_count, n_in_complex), None, dtype=object)

    for rep in range(2):
        for diag_index in range(diag_count):
            combined_diagonal = np.concatenate(
                (diag_blocks_list[rep * 2][diag_index], diag_blocks_list[rep * 2 + 1][diag_index]),
                axis=0,
            )
            for n in range(n_in_complex):
                for out_index in range(n_out_packed):
                    msg = np.zeros((SLOT_COUNT,), dtype=complex)
                    rotations = out_index * 16 + PACK_RANGE
                    input_indices = ((n // 16) * 16 + out_index * 16 + (n + PACK_RANGE) % 16) % n_in_complex
                    input_indices = (input_indices - rotations) % n_in
                    values = scale * gather_upper_diagonal_batch(
                        combined_diagonal, input_indices, rotations, n_in, n_in_complex
                    )
                    assign_packed_dense_block(msg, values, FF_SLOT_INDICES)
                    messages[rep, out_index, diag_index, n] = msg
    return messages


def encode_w_pooler(w: np.ndarray, block_shape: tuple[int, int] = (128, 128)) -> np.ndarray:
    if w.shape[0] % block_shape[0] != 0 or w.shape[1] % block_shape[1] != 0:
        raise ValueError("Dimension does not match")

    diag_blocks, (diag_count, _) = to_blocks(w, block_shape, diag=True)
    messages = np.full((diag_count, 4), None, dtype=object)
    for diag_index in range(diag_count):
        blocks = diag_blocks[diag_index]
        for n in range(4):
            msg = np.zeros((SLOT_COUNT,), dtype=complex)
            for pack_offset in range(PACK):
                input_index = n * 16 + pack_offset
                group_offset = pack_offset * GROUP_SIZE
                values = (blocks[:, :, input_index] - 1j * blocks[:, :, input_index + 64]) / 2
                msg[group_offset + DIM_SLOT_BASE + np.arange(values.shape[0])[None, :]] = values.T
            messages[diag_index, n] = msg
    return messages


def encode_w_cls(w: np.ndarray) -> np.ndarray:
    if w.shape[1] != 768:
        raise ValueError(f"Shape of classifier weight should be (class_count, 768). Shape is {w.shape}")

    class_count = w.shape[0]
    messages = np.full((class_count,), None, dtype=object)
    positions = DIM_SLOT_BASE + np.arange(6)[None, :]
    for class_index in range(class_count):
        msg = np.zeros((SLOT_COUNT,), dtype=float)
        blocks = w[class_index].reshape(6, DIM)
        msg[positions] = blocks.T
        messages[class_index] = msg
    return messages


def encode_b(
    b: np.ndarray,
    n_blocks: int,
    n_out: int,
    pack: int = 16,
    n_slot: int = 16,
    pad_index=None,
    scale: float = 1,
) -> np.ndarray:
    if pad_index is None:
        pad_index = [slot for slot in range(n_slot) if slot >= n_blocks]
    elif n_blocks + len(pad_index) != n_slot:
        raise ValueError("Parameters do not match")

    if not isinstance(b, np.ndarray):
        raise ValueError("Input should be a numpy array")
    if b.shape[0] % n_blocks != 0:
        raise ValueError("Block size does not match")

    blocks = np.stack(np.split(b, n_blocks), axis=0)
    n_out_packed = n_out // pack
    messages = np.full((n_out_packed,), None, dtype=object)
    active_slots = np.array([slot for slot in range(n_slot) if slot not in pad_index])
    output_positions = DIM_SLOT_BASE + active_slots[None, :]
    pack_group_base = GROUP_SIZE * np.arange(pack)[:, None, None]

    for out_index in range(n_out_packed):
        msg = np.zeros((SLOT_COUNT,), dtype=float)
        rotations = out_index * pack + np.arange(pack)
        gather_indices = (rotations[:, None] + DIM_RANGE[None, :]) % blocks.shape[1]
        rotated = np.take_along_axis(blocks[:, None, :], gather_indices[None, :, :], axis=2).transpose(1, 2, 0)
        msg[pack_group_base + output_positions[None, :, :]] = (scale * rotated) / 2
        messages[out_index] = msg
    return messages


def encode_b_pooler(b: np.ndarray, n_blocks: int, n_slot: int = 16, pad_index=None) -> np.ndarray:
    if pad_index is None:
        pad_index = [slot for slot in range(n_slot) if slot >= n_blocks]
    elif n_blocks + len(pad_index) != n_slot:
        raise ValueError("Parameters do not match")

    if not isinstance(b, np.ndarray):
        raise ValueError("Input should be a numpy array")
    if b.shape[0] % n_blocks != 0:
        raise ValueError("Block size does not match")

    blocks = np.stack(np.split(b, n_blocks), axis=1)
    msg = np.zeros((GROUP_SIZE,), dtype=float)
    active_slots = np.array([slot for slot in range(n_slot) if slot not in pad_index])
    msg[DIM_SLOT_BASE + active_slots[None, :]] = blocks / 2

    messages = np.full((1,), None, dtype=object)
    messages[0] = np.tile(msg, 2**4)
    return messages


def encode_b_cls(b: np.ndarray) -> np.ndarray:
    class_count = b.shape[0]
    messages = np.full((class_count,), None, dtype=object)
    for class_index in range(class_count):
        msg = np.zeros((SLOT_COUNT,), dtype=float)
        msg[0] = b[class_index]
        messages[class_index] = msg
    return messages
