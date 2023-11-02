import numpy as np
import rawpy
import exifread
import os
from sample import Sample


class RawImageReader:
    def __init__(self, filename: str, cut_size: np.ndarray | None = None):
        self._filename = filename
        self._cut_size = cut_size
        self._file_manager: RawFileManager = RawFileManager(filename)

    @property
    def filename(self):
        return self._filename

    def load_data(self) -> Sample:
        data = Sample()
        data.cut_size = self._cut_size
        data.filename = self.filename
        data.black_level = self._get_black_level()
        data.white_level = self._get_white_level()
        data.bayer_pattern = self._get_bayer_pattern()
        data.rgb2xyz_matrix = self._get_rgb2xyz_matrix()
        data.tone_curve = self._get_tone_curve()
        data = self._get_exif_data(data)
        data.cfa = self._get_cfa()

        return data

    def _get_black_level(self) -> np.ndarray:
        with self._file_manager as raw:
            return np.array(raw.black_level_per_channel, dtype=int).reshape(1, 1, 4)

    def _get_white_level(self) -> np.ndarray:
        with self._file_manager as raw:
            raw_white_level = np.array(raw.white_level, dtype=int)
            return np.expand_dims(raw_white_level, axis=-1).repeat(4, axis=-1)

    def _get_bayer_pattern(self) -> str:
        with self._file_manager as raw:
            return raw.color_desc.decode('utf-8')

    def _get_rgb2xyz_matrix(self) -> np.ndarray:
        with self._file_manager as raw:
            return np.array(raw.rgb_xyz_matrix, dtype=float)

    def _get_tone_curve(self) -> np.ndarray:
        with self._file_manager as raw:
            return np.array(raw.tone_curve, dtype=int).reshape(1, -1)

    def _get_cfa(self) -> np.ndarray:
        with self._file_manager as raw:
            return np.array(raw.raw_image_visible.copy(), dtype=int)

    def _get_exif_data(self, data) -> Sample:
        with open(self._filename, 'rb') as file:
            tags = exifread.process_file(file)

        data.exposure_time = float(tags["EXIF ExposureTime"].values[0])
        data.f_number = float(tags["EXIF FNumber"].values[0])
        data.camera_model = str(tags["Image Model"].values).upper()

        try:
            data.iso = float(tags["EXIF ISOSpeedRatings"].values[0])
        except (KeyError, ValueError):
            data.iso = 100.

        return data


class RawFileManager:
    def __init__(self, filename):
        if not self._is_valid_file(filename):
            raise RuntimeError("Invalid file.")
        self.filename = filename
        self.raw = None

    def __enter__(self):
        self.raw = rawpy.imread(self.filename)
        return self.raw

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            self.raw.close()

    def __del__(self):
        if self.raw is not None:
            self.raw.close()
            self.raw = None

    @staticmethod
    def _is_valid_file(path) -> bool:
        extension = os.path.splitext(path)[1].lower()
        return extension in ['.nef', '.cr2']
