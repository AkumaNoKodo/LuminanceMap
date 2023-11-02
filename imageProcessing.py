import numpy as np
import re
import glob
import os

# def timing_decorator(func):
#     def wrapper(*args, **kwargs):
#         start_time = time.time()
#         result = func(*args, **kwargs)
#         end_time = time.time()
#         print(f"Process time: {func.__name__}: {end_time - start_time} [second]")
#         return result
#
#     return wrapper

def cfa2channels(cfa: np.ndarray) -> np.ndarray:
    return np.stack((cfa[::2, ::2], cfa[::2, 1::2], cfa[1::2, ::2], cfa[1::2, 1::2]), axis=2)


def rescale_channels(channels: np.ndarray, black_level: np.ndarray, white_level: np.ndarray) -> np.ndarray:
    return (channels - black_level) / (white_level - black_level)


def channels2rgb(channels: np.ndarray, cfa_pattern: str) -> np.ndarray:
    pattern = cfa_pattern.upper()
    indices_of_r = np.array([match.start() for match in re.finditer('R', pattern)])
    indices_of_g = np.array([match.start() for match in re.finditer('G', pattern)])
    indices_of_b = np.array([match.start() for match in re.finditer('B', pattern)])

    return np.stack((
        channels[:, :, indices_of_r].squeeze(),
        np.mean(channels[:, :, indices_of_g], axis=2).squeeze(),
        channels[:, :, indices_of_b].squeeze()
    ), axis=2)


def apply_tone_curve(cfa: np.ndarray, tone_curve: np.ndarray) -> np.ndarray:
    return np.take(tone_curve, cfa)


def rgb2y(rgb: np.ndarray, conversion_matrix: np.ndarray) -> np.ndarray:
    return np.einsum('ijk,k->ij', rgb, conversion_matrix)


def cut_image(image: np.ndarray, cut_size: np.ndarray) -> np.ndarray:
    return image[cut_size[0, 0]:(cut_size[0, 0] + cut_size[0, 2]),
                 cut_size[0, 1]:(cut_size[0, 1] + cut_size[0, 3])
                 ]


def find_files_with_extension(directory: str, extension: str) -> list:
    extension = extension.lower()
    directory_path = os.path.abspath(directory)
    search_path = os.path.join(directory_path, f"**/*.{extension}")
    files = glob.glob(search_path, recursive=True)
    return files
