from pathlib import Path

LIGHT_PLAINTEXT_BASE_PATH = Path("./light_plaintexts")


def get_light_plaintext_path(compact):
    return LIGHT_PLAINTEXT_BASE_PATH / ("compact" if compact else "default")
