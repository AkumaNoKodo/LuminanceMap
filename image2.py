import os
import pickle
import re
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any
import attr
import exifread
import numpy as np
import psutil
import rawpy


class Converter:
    @staticmethod
    def _check_instance(value: Any, control_class: Any, np_ndarray_precision: str) -> Any:
        if value is None:
            return None

        if not isinstance(value, control_class):
            try:
                value = control_class(value)
            except Exception as e:
                print(e)
                return None

        if np_ndarray_precision:
            try:
                value = value.astype(np_ndarray_precision)
            except Exception as e:
                print(e)
                return None

        return value

    @staticmethod
    def _check_size(value: Any, expected_shape: int | tuple[int, ...]) -> Any:
        if value is None:
            return None

        try:
            if isinstance(value, np.ndarray):
                size = value.shape
            else:
                size = len(value)
        except Exception as e:
            print(e)
            return None

        if size != expected_shape:
            print("Failed")
            return None

        return value

    def define_converter(self, control_class: Any = None, expected_size: int | tuple[int, ...] = 0,
                         np_ndarray_precision: str = ""):
        def converter(value):
            if control_class:
                value = self._check_instance(value, control_class, np_ndarray_precision)
            if expected_size:
                value = self._check_size(value, expected_size)
            return value

        return converter


@attr.define
class Image:
    converter = Converter().define_converter

    path: Path | None = \
        attr.field(converter=converter(control_class=Path))

    _cfa: np.ndarray | None = None

    _rgb: np.ndarray | None = None

    is_temped: bool = False

    tag: str | None = \
        attr.field(default=None,
                   converter=converter(control_class=str))

    black_level: np.ndarray | None = \
        attr.field(default=None,
                   converter=converter(control_class=np.ndarray,
                                       expected_size=(1, 1, 4),
                                       np_ndarray_precision="uint16"))

    white_level: np.ndarray | None = \
        attr.field(default=None,
                   converter=converter(control_class=np.ndarray,
                                       expected_size=(1, 1, 4),
                                       np_ndarray_precision="uint16"))

    bayer_pattern: str | None = \
        attr.field(default=None,
                   converter=converter(control_class=str,
                                       expected_size=4))

    rgb2xyz_matrix: np.ndarray | None = \
        attr.field(default=None,
                   converter=converter(control_class=np.ndarray,
                                       expected_size=(3, 3),
                                       np_ndarray_precision="float32"))

    camera_model: str | None = \
        attr.field(default=None,
                   converter=converter(control_class=str))

    exposure_time: float | None = \
        attr.field(default=None,
                   converter=converter(control_class=float))

    f_number: float | None = \
        attr.field(default=None,
                   converter=converter(control_class=float))

    iso: float | None = \
        attr.field(default=None,
                   converter=converter(control_class=float))

    capture_time: str | None = \
        attr.field(default=None,
                   converter=converter(control_class=str))

    observer: float | None = \
        attr.field(default=None,
                   converter=converter(control_class=float))

    def __attrs_post_init__(self):
        self.tag = self.path.parts[-2]

    @property
    def cfa(self):
        if self._cfa and isinstance(self._cfa, str):
            with open(self._cfa, 'rb') as temporary_file:
                return pickle.load(temporary_file)
        else:
            return self._cfa

    @cfa.setter
    def cfa(self, value):
        converter = self.converter(control_class=np.ndarray, np_ndarray_precision="uint16")
        self._cfa = converter(value)
        self.make_temporary("_cfa")

    @property
    def rgb(self):
        if self._cfa and isinstance(self._rgb, str):
            with open(self._rgb, 'rb') as temporary_file:
                return pickle.load(temporary_file)
        else:
            return self._rgb

    @rgb.setter
    def rgb(self, value):
        converter = self.converter(control_class=np.ndarray, np_ndarray_precision="float32")
        self._rgb = converter(value)
        self.make_temporary("_rgb")

    @property
    def rescale_cfa(self):
        cfa = self.cfa
        if cfa is not None:
            channels = self.cfa2channels(cfa)
            channels = self.rescale_channels(channels, self.black_level, self.white_level)
            reconstructed_cfa = self.channels2cfa(channels, self.bayer_pattern)

            return reconstructed_cfa
        else:
            return None

    def make_temporary(self, attribute: str):
        if self.is_temped:
            primary_folder = "TEMP"
            secondary_folder = os.path.join(primary_folder, "IMAGE")

            if not os.path.exists(primary_folder):
                os.makedirs(primary_folder)

            if not os.path.exists(secondary_folder):
                os.makedirs(secondary_folder)

            time_stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
            temporary_name = os.path.join(secondary_folder, f"temp_{time_stamp}.pkl")

            with open(temporary_name, 'wb') as f:
                pickle.dump(getattr(self, attribute), f)

            setattr(self, attribute, temporary_name)

    @staticmethod
    def channels2cfa(channels, cfa_pattern: str) -> np.ndarray:
        height, width, _ = channels.shape[0] * 2, channels.shape[1] * 2, channels.shape[2]
        reconstructed_cfa = np.zeros((height, width), dtype=channels.dtype)

        indexes = Image.get_patter_indexes(cfa_pattern)

        reconstructed_cfa[::2, ::2] = np.squeeze(channels[:, :, indexes[0]])
        reconstructed_cfa[::2, 1::2] = np.squeeze(channels[:, :, indexes[1][0]])
        reconstructed_cfa[1::2, ::2] = np.squeeze(channels[:, :, indexes[1][1]])
        reconstructed_cfa[1::2, 1::2] = np.squeeze(channels[:, :, indexes[2]])

        return reconstructed_cfa

    @staticmethod
    def cfa2channels(cfa: np.ndarray) -> np.ndarray:
        return np.stack((cfa[::2, ::2], cfa[::2, 1::2], cfa[1::2, ::2], cfa[1::2, 1::2]), axis=2, dtype="float32")

    @staticmethod
    def rescale_channels(channels: np.ndarray, black_level: np.ndarray, white_level: np.ndarray) -> np.ndarray:
        channels = (channels - black_level) / white_level
        channels[(channels <= 0) | (channels >= 1)] = np.nan
        return channels

    @staticmethod
    def get_patter_indexes(cfa_pattern: str) -> list[list[int] | int]:
        pattern = cfa_pattern.upper()
        letters = ["R", "G", "B"]
        return [[match.start() for match in re.finditer(letter, pattern)] for letter in letters]

    @staticmethod
    def channels2rgb(channels: np.ndarray, cfa_pattern: str) -> np.ndarray:
        indexes = Image.get_patter_indexes(cfa_pattern)

        return np.stack((
            channels[:, :, indexes[0]].squeeze(),
            np.mean(channels[:, :, indexes[1]], axis=2).squeeze(),
            channels[:, :, indexes[2]].squeeze()
        ), axis=2)

    @staticmethod
    def rgb2y(rgb: np.ndarray, conversion_matrix: np.ndarray) -> np.ndarray:
        return np.einsum('ijk,k->ij', rgb, conversion_matrix)

    @staticmethod
    def cut_image(image: np.ndarray, cut_size: np.ndarray) -> np.ndarray:
        return image[cut_size[0, 0]:(cut_size[0, 0] + cut_size[0, 2]),
                     cut_size[0, 1]:(cut_size[0, 1] + cut_size[0, 3])]


class NEFImage(Image):
    def __init__(self, file_path: Path, is_temped: bool = False):
        super().__init__(path=file_path, is_temped=is_temped)

        if not self.path or self.path.suffix != '.nef':
            print("Wrong or missing file")
            return

        with rawpy.imread(str(self.path)) as raw:
            self.black_level = np.array(raw.black_level_per_channel, dtype="float32").reshape(1, 1, 4)
            self.white_level = np.array(raw.camera_white_level_per_channel, dtype="float32").reshape(1, 1, 4)
            self.bayer_pattern = raw.color_desc.decode('utf-8')
            self.rgb2xyz_matrix = raw.rgb_xyz_matrix[:3, :3]
            self.cfa = raw.raw_image_visible.copy()

        with open(file_path, 'rb') as exif_data_file:
            exif_data = exifread.process_file(exif_data_file)

            self.camera_model = str(exif_data["Image Model"].values).upper()
            self.exposure_time = float(exif_data["EXIF ExposureTime"].values[0])
            self.f_number = float(exif_data["EXIF FNumber"].values[0])
            try:
                self.iso = float(exif_data["EXIF ISOSpeedRatings"].values[0])
            except Exception as e:
                # print(f"{e} --- set default value 100 --- {str(self.path)}")
                self.iso = 100
            self.capture_time = str(exif_data["EXIF DateTimeOriginal"].values).upper()
            self.observer = float(exif_data["Image PhotometricInterpretation"].values[0])


class ImageLoader:
    def __init__(self, is_temped: bool = False):
        self.is_temped = is_temped

    def load_images_sets_from_folder(self, folder: str):
        sub_folders = [Path(item) for item in Path(folder).iterdir() if item.is_dir()]
        files = [file for sub_folder in sub_folders for file in list(sub_folder.glob("*.*"))]

        return self.read_images_data(list(map(str, files)))

    def load_images_from_folder(self, folder: str):
        files = list(map(str, Path(folder).glob("*.*")))

        return self.read_images_data(files)

    def read_images_data(self, file_paths: list[str]):
        start_time = time.perf_counter()

        with ProcessPoolExecutor(max_workers=psutil.cpu_count(logical=False)) as executor:
            print(f"Start multiprocessing image loading: Total files for execute {len(file_paths)}")
            images = list(executor.map(self.process_file, file_paths))
        images = [image for image in images if image]
        print(f"Loading end: {len(images)}, Process time --- {time.perf_counter() - start_time}")

        return images

    def process_file(self, path: str):
        path = Path(path)

        if path.suffix.lower() == ".nef":
            try:
                image = NEFImage(path, self.is_temped)
                print(f"Loading complete: {path}")
                return image
            except Exception as e:
                print(f"Failed file --- {path} --- {e}")
        else:
            print(f"Failed file --- {path} --- invalid extension --- {path.suffix.lower()}")
            return None
