import time

import glob
import os


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        print(f"Process time: {func.__name__}: {time.perf_counter() - start_time} [second]")
        return result

    return wrapper


