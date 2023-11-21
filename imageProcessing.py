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


def find_files_with_extension(directory: str, extension: str) -> list:
    extension = extension.lower()
    directory_path = os.path.abspath(directory)
    search_path = os.path.join(directory_path, f"**/*.{extension}")
    files = glob.glob(search_path, recursive=True)
    return files
