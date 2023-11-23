import datetime
import json
import pickle
import re
from pprint import pprint
import rawpy
import exifread
import os
from abc import ABC
import numpy as np
from imageProcessing import find_files_with_extension
import glob
import shutil


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


class RawImageConvertor:
    @staticmethod
    def apply_post_processing_rgb_algorythm(raw_image: RawImage) -> dict:
        image_data_dict = raw_image.get_data_like_dict()

        cfa = image_data_dict.pop("cfa", None)
        tone_curve = image_data_dict.pop("tone_curve", None)
        cfa = RawImageConvertor.apply_tone_curve(cfa, tone_curve)
        channels = RawImageConvertor.cfa2channels(cfa)

        black_level = image_data_dict.pop("black_level", None)
        white_level = image_data_dict.pop("white_level", None)
        channels = RawImageConvertor.rescale_channels(channels, black_level, white_level)
        image_data_dict["channels"] = channels

        bayer_pattern = image_data_dict.pop("bayer_pattern", None)
        rgb = RawImageConvertor.channels2rgb(image_data_dict.pop("channels", None), bayer_pattern)
        image_data_dict["rgb"] = rgb

        return image_data_dict

    @staticmethod
    def cfa_analyze(image_data_dict: dict) -> dict:
        cfa = image_data_dict.pop("cfa", None)
        tone_curve = image_data_dict.pop("tone_curve", None)
        cfa = RawImageConvertor.apply_tone_curve(cfa, tone_curve)
        channels = RawImageConvertor.cfa2channels(cfa)

        black_level = image_data_dict.pop("black_level", None)
        white_level = image_data_dict.pop("white_level", None)
        channels = RawImageConvertor.rescale_channels(channels, black_level, white_level)
        reconstructed_cfa = RawImageConvertor.channels2cfa(channels, image_data_dict["bayer_pattern"])
        image_data_dict["reconstructed_cfa"] = reconstructed_cfa
        return image_data_dict

    @staticmethod
    def channels2cfa(channels, cfa_pattern: str) -> np.ndarray:
        height, width, _ = channels.shape[0] * 2, channels.shape[1] * 2, channels.shape[2]
        reconstructed_cfa = np.zeros((height, width), dtype=channels.dtype)

        indexes = RawImageConvertor.get_patter_indexes(cfa_pattern)

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
        indexes = RawImageConvertor.get_patter_indexes(cfa_pattern)

        return np.stack((
            channels[:, :, indexes[0]].squeeze(),
            np.mean(channels[:, :, indexes[1]], axis=2).squeeze(),
            channels[:, :, indexes[2]].squeeze()
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
        return glob.glob(search_path, recursive=True)


class TempRawImages:
    def __init__(self):
        self.main_folder = "TEMP"
        self.tem_json_file = os.path.join(self.main_folder, "temp_information.json")
        self.temp_files_list = []
        if not os.path.exists(self.main_folder):
            os.makedirs(self.main_folder)
        else:
            if os.path.exists(self.tem_json_file):
                with open(self.tem_json_file, 'r') as file:
                    self.temp_files_list = json.load(file)

    def create_from_folder(self, folder: str, extension: str, clean_option: bool = True):
        sub_folders = \
            [os.path.join(folder, item) for item in os.listdir(folder) if os.path.isdir(os.path.join(folder, item))]

        if clean_option:
            self.clean()

        for sub_folder in sub_folders:
            raw_files = find_files_with_extension(sub_folder, extension)
            tag = os.path.basename(os.path.dirname(raw_files[0]))

            self.temp_files_list.extend([self.get_image_data(file, self.main_folder, tag) for file in raw_files])
        self.save_content()

    @staticmethod
    def get_image_data(file: str, main_folder: str, tag: str):
        im = RawImage(file)
        process_im = im.get_data_like_dict()
        time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
        path = os.path.join(main_folder,
                            f"temp_{time_stamp}.pkl")

        information_dict = {"tag": tag,
                            "src_file_name": path,
                            "iso": process_im["iso"],
                            "f_number": process_im["f_number"],
                            "exposure_time": process_im['exposure_time']
                            }

        with open(information_dict["src_file_name"], 'wb') as f:
            pickle.dump(process_im, f)

        return information_dict

    def save_content(self):
        with open(self.tem_json_file, 'w') as file:
            json.dump(self.temp_files_list, file, indent=4)

    def clean(self):
        if os.path.exists(self.main_folder):
            shutil.rmtree(self.main_folder)
        os.makedirs(self.main_folder)
