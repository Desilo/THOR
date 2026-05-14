import time
from contextlib import contextmanager


def to_elapsed_time(diff) -> str:
    if diff >= 60:
        minutes = int(diff // 60)
        seconds = diff - minutes * 60
        return f"{minutes:2d}m {seconds:6.3f}s"
    return f"    {diff:6.3f}s"


class Timer:
    def __init__(self):
        self.start = time.perf_counter()
        self.paused_time = 0.0

    @property
    def elapsed(self) -> str:
        return to_elapsed_time(time.perf_counter() - self.start)

    @property
    def true_elapsed(self) -> str:
        return to_elapsed_time(time.perf_counter() - self.start - self.paused_time)

    def reset(self):
        self.start = time.perf_counter()
        self.paused_time = 0.0

    def pause(self):
        self.pause_start = time.perf_counter()

    def resume(self):
        self.paused_time += time.perf_counter() - self.pause_start

    @contextmanager
    def paused(self):
        self.pause()
        try:
            yield
        finally:
            self.resume()

    @contextmanager
    def setup(self):
        setup_start = time.perf_counter()
        try:
            yield
        finally:
            setup_time = to_elapsed_time(time.perf_counter() - setup_start)
            print("----------------------------------------------------------------------------------")
            print(f"Setup: {setup_time}")
            print("----------------------------------------------------------------------------------")
            self.reset()

    @contextmanager
    def stage(self, stage_index, stage_name):
        stage_start = time.perf_counter()
        try:
            yield
        finally:
            stage_time = to_elapsed_time(time.perf_counter() - stage_start)
            print(
                f"          {stage_time}     {self.true_elapsed}     {self.elapsed}    {stage_index:>2}: {stage_name}"
            )

    @contextmanager
    def layer(self, layer_index):
        layer_start = time.perf_counter()
        try:
            yield
        finally:
            layer_time = to_elapsed_time(time.perf_counter() - layer_start)
            print("----------------------------------------------------------------------------------")
            print(f"Layer {layer_index:2}  {layer_time}     {self.true_elapsed}     {self.elapsed}")
            print("----------------------------------------------------------------------------------")

    def print_legend(self):
        print("----------------------------------------------------------------------------------")
        print("           stage time    compute time       total time    stage name")
        print("----------------------------------------------------------------------------------")
        print(f"                          {self.true_elapsed}     {self.elapsed}")
        print("----------------------------------------------------------------------------------")
