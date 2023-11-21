import re
from pprint import pprint
from typing import Tuple, List
import rawpy
import exifread
import os
from abc import ABC
import numpy as np
from imageProcessing import timing_decorator
import glob


def safe_initialization(decorated_class):
    def wrapper(*args, **kwargs):
        try:
            out = decorated_class(*args, **kwargs)
            return out
        except Exception as e:
            print(f"{decorated_class.__name__} Failed: {e}")
            return EmptyData()

    return wrapper


class Data(ABC):
    def __init__(self, input_python_class, input_data_name):
        self.python_class: type = input_python_class
        self.data_name: str | None = input_data_name

        python_to_database_type_dict = {"str": "TEXT", "float": "REAL", "ndarray": "NPARRAY"}
        self.database_data_type: str | None = python_to_database_type_dict[self.python_class.__name__]

        self.class_and_precision: str | None = None
        self._data: str | float | np.ndarray | None = None
        self.expected_size: tuple | int | None = None

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value: str | float | int | np.ndarray):
        if not isinstance(value, self.python_class):
            print(f"Expected type: {self.python_class.__name__}, "
                  f"but received type: {type(value).__name__}, "
                  f"for value: {value}")
            self._all_to_none()
            return

        if self.expected_size is not None:
            if isinstance(value, str) and len(value) != self.expected_size:
                print(f"Expected size: {self.expected_size}, "
                      f"but received size: {len(value)}, "
                      f"for value: {value}")
                self._all_to_none()
                return
            elif isinstance(value, np.ndarray) and value.shape != self.expected_size:
                print(f"Expected size: {self.expected_size}, "
                      f"but received size: {value.shape}, "
                      f"for value: {value}")
                self._all_to_none()
                return

        if isinstance(self.python_class, float):
            try:
                value = float(value)
            except Exception as e:
                print(f"Expected float or converted to float: "
                      f"but received class: {type(value).__name__}, "
                      f"with value: {value}, "
                      f"with except: {e}")
                self._all_to_none()
                return

        if isinstance(value, np.ndarray):
            try:
                if self.class_and_precision is None:
                    value = value.astype("float32")
                else:
                    value = value.astype(self.class_and_precision)
            except Exception as e:
                print(f"Expected np.ndarray convertible to: {self.class_and_precision}, "
                      f"but received class: {type(value).__name__}, "
                      f"with value: {value}, "
                      f"with except: {e}")
                self._all_to_none()
                return

        self._data = value

    def _all_to_none(self) -> None:
        for key in self.__dict__.keys():
            setattr(self, key, None)


@safe_initialization
class EmptyData(Data):
    def __init__(self):
        super().__init__(None, None)


@safe_initialization
class Filename(Data):
    def __init__(self, file_path):
        super().__init__(str, "filename")
        name, _ = os.path.splitext(os.path.basename(file_path))
        self.data = name


@safe_initialization
class Extension(Data):
    def __init__(self, file_path):
        super().__init__(str, "extension")
        _, extension = os.path.splitext(os.path.basename(file_path))
        self.data = extension


@safe_initialization
class BlackLevel(Data):
    def __init__(self, raw_image: rawpy):
        super().__init__(np.ndarray, "black_level")
        self.expected_size = (1, 1, 4)
        self.class_and_precision = "uint16"
        self.data = np.array(raw_image.black_level_per_channel, dtype="float32").reshape(1, 1, 4)


@safe_initialization
class WhiteLevel(Data):
    def __init__(self, raw_image: rawpy):
        super().__init__(np.ndarray, "white_level")
        self.expected_size = (1, 1, 4)
        self.class_and_precision = "uint16"
        self.data = np.array(raw_image.camera_white_level_per_channel, dtype="float32").reshape(1, 1, 4)


@safe_initialization
class BayerPattern(Data):
    def __init__(self, raw_image: rawpy):
        super().__init__(str, "bayer_pattern")
        self.expected_size = 4
        self.data = raw_image.color_desc.decode('utf-8')


@safe_initialization
class RGB2XYZMatrix(Data):
    def __init__(self, raw_image: rawpy):
        super().__init__(np.ndarray, "rgb2xyz_matrix")
        self.expected_size = (3, 3)
        self.data = raw_image.rgb_xyz_matrix[:3, :3]


@safe_initialization
class ToneCurve(Data):
    def __init__(self, raw_image: rawpy):
        super().__init__(np.ndarray, "tone_curve")
        self.expected_size = (1, 65536)
        self.class_and_precision = "uint16"
        tone_curve = raw_image.tone_curve.copy()
        self.data = np.array(tone_curve).reshape(1, -1).copy()


@safe_initialization
class ExposureTime(Data):
    def __init__(self, exif_data: dict):
        super().__init__(float, "exposure_time")
        self.expected_size = 1
        self.data = float(exif_data["EXIF ExposureTime"].values[0])


@safe_initialization
class FNumber(Data):
    def __init__(self, exif_data: dict):
        super().__init__(float, "f_number")
        self.expected_size = 1
        self.data = float(exif_data["EXIF FNumber"].values[0])


@safe_initialization
class Iso(Data):
    def __init__(self, exif_data: dict):
        super().__init__(float, "iso")
        self.expected_size = 1
        try:
            data = float(exif_data["EXIF ISOSpeedRatings"].values[0])
        except (KeyError, ValueError):
            data = 100.
        self.data = data


@safe_initialization
class CameraModel(Data):
    def __init__(self, exif_data: dict):
        super().__init__(str, "camera_model")
        self.data = str(exif_data["Image Model"].values).upper()


@safe_initialization
class CaptureTime(Data):
    def __init__(self, exif_data: dict):
        super().__init__(str, "capture_time")
        self.data = str(exif_data["EXIF DateTimeOriginal"].values).upper()


@safe_initialization
class Observer(Data):
    def __init__(self, exif_data: dict):
        super().__init__(float, "camera_model")
        self.data = float(exif_data["Image PhotometricInterpretation"].values[0])


@safe_initialization
class Cfa(Data):
    def __init__(self, raw_image: rawpy):
        super().__init__(np.ndarray, "cfa")
        self.class_and_precision = "uint16"
        self.data = raw_image.raw_image_visible.copy()


@timing_decorator
class RawImage:
    def __init__(self, file_path):
        if not os.path.splitext(file_path)[1].lower() in ['.nef', '.cr2']:
            print("Wrong file, bad extension.")
            return

        self.filename: Filename = Filename(file_path)
        self.extension: Extension = Extension(file_path)

        with rawpy.imread(file_path) as raw:
            self.black_level: BlackLevel = BlackLevel(raw)
            self.white_level: WhiteLevel = WhiteLevel(raw)
            self.bayer_pattern: BayerPattern = BayerPattern(raw)
            self.rgb2xyz_matrix: RGB2XYZMatrix = RGB2XYZMatrix(raw)
            self.tone_curve: ToneCurve = ToneCurve(raw)
            self.cfa: Cfa = Cfa(raw)

        with open(file_path, 'rb') as file:
            exif_data = exifread.process_file(file)

            self.camera_model: CameraModel = CameraModel(exif_data)
            self.exposure_time: ExposureTime = ExposureTime(exif_data)
            self.f_number: FNumber = FNumber(exif_data)
            self.iso: Iso = Iso(exif_data)
            self.capture_time: CaptureTime = CaptureTime(exif_data)
            self.observer: Observer = Observer(exif_data)

    def get_data_like_dict(self, print_: bool = False):
        data_dict = {attr_name: attr_value.data for attr_name, attr_value in self.__dict__.items()
                     if isinstance(attr_value, Data)}
        if print_:
            pprint(data_dict)
        return data_dict

    def new_table_command(self, table_name) -> str:
        columns_setting = [f"{value.data_name} {value.database_data_type}" for value in self.__dict__.values() if
                           isinstance(value, Data)]
        return (f"CREATE TABLE IF NOT EXISTS {table_name} ( id INTEGER PRIMARY KEY AUTOINCREMENT, " +
                ", ".join(columns_setting) + ")")

    def new_insert_command(self, table_name) -> Tuple[str, List]:
        columns, values = map(list, zip(*self.__dict__.items()))
        command = (f"INSERT INTO {table_name} ({', '.join(columns)}) "
                   f"VALUES ({', '.join(['?' for _ in columns])})")
        values = [value.data for value in values]
        return command, values


class RawImageConvertor:
    def apply_post_processing_rgb_algorythm(self, raw_image: RawImage) -> dict:
        image_data_dict = raw_image.get_data_like_dict()

        cfa = image_data_dict.pop("cfa", None)
        tone_curve = image_data_dict.pop("tone_curve", None)
        cfa = self.apply_tone_curve(cfa, tone_curve)
        del tone_curve

        channels = self.cfa2channels(cfa)
        del cfa

        black_level = image_data_dict.pop("black_level", None)
        white_level = image_data_dict.pop("white_level", None)
        channels = self.rescale_channels(channels, black_level, white_level)

        bayer_pattern = image_data_dict.pop("bayer_pattern", None)
        rgb = self.channels2rgb(channels, bayer_pattern)
        image_data_dict["rgb"] = rgb

        return image_data_dict

    @staticmethod
    def cfa2channels(cfa: np.ndarray) -> np.ndarray:
        return np.stack((cfa[::2, ::2], cfa[::2, 1::2], cfa[1::2, ::2], cfa[1::2, 1::2]), axis=2, dtype="float32")

    @staticmethod
    def rescale_channels(channels: np.ndarray, black_level: np.ndarray, white_level: np.ndarray) -> np.ndarray:
        channels = (channels - black_level) / (white_level - black_level)
        channels = np.where(channels < 0, 0, channels)
        channels = np.where(channels > 1, 1, channels)
        return channels

    @staticmethod
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

    @staticmethod
    def apply_tone_curve(cfa: np.ndarray, tone_curve: np.ndarray) -> np.ndarray:
        return np.take(tone_curve, cfa)

    @staticmethod
    def rgb2y(rgb: np.ndarray, conversion_matrix: np.ndarray) -> np.ndarray:
        return np.einsum('ijk,k->ij', rgb, conversion_matrix)

    @staticmethod
    def cut_image(image: np.ndarray, cut_size: np.ndarray) -> np.ndarray:
        return image[cut_size[0, 0]:(cut_size[0, 0] + cut_size[0, 2]),
                     cut_size[0, 1]:(cut_size[0, 1] + cut_size[0, 3])]

    @staticmethod
    def find_files_with_extension(directory: str, extension: str) -> list:
        extension = extension.lower()
        directory_path = os.path.abspath(directory)
        search_path = os.path.join(directory_path, f"**/*.{extension}")
        files = glob.glob(search_path, recursive=True)
        return files


