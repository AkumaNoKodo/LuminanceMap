from typing import Tuple, List
import numpy as np
from dataclasses import dataclass, field
from data import Data


@dataclass
class Sample:
    filename: Data = field(default_factory=lambda: Data(str, dtype="str"))
    black_level: Data = field(default_factory=lambda: Data(np.ndarray, (1, 1, 4)))
    white_level: Data = field(default_factory=lambda: Data(np.ndarray, (1, 1, 4)))
    bayer_pattern: Data = field(default_factory=lambda: Data(str, 4, "str"))
    rgb2xyz_matrix: Data = field(default_factory=lambda: Data(np.ndarray, (4, 3)))
    tone_curve: Data = field(default_factory=lambda: Data(np.ndarray, (1, 65536)))
    exposure_time: Data = field(default_factory=lambda: Data(float, 1))
    f_number: Data = field(default_factory=lambda: Data(float, 1))
    iso: Data = field(default_factory=lambda: Data(float, 1))
    camera_model: Data = field(default_factory=lambda: Data(str, dtype="str"))
    cfa: Data = field(default_factory=lambda: Data(np.ndarray, dtype="float32"))
    cut_size: Data = field(default_factory=lambda: Data(np.ndarray, (2, 2)))

    def __setattr__(self, name, value):
        if isinstance(value, Data):
            value._data_name = name
        super().__setattr__(name, value)

    def new_table_command(self) -> str:
        columns_setting = [f"{value.data_name} {value.database_type}" for value in self.__dict__.values() if
                           isinstance(value, Data)]
        return (f"CREATE TABLE IF NOT EXISTS sample ( id INTEGER PRIMARY KEY AUTOINCREMENT, " +
                ", ".join(columns_setting) + ")")

    def new_insert_command(self) -> Tuple[str, List]:
        columns, values = map(list, zip(*self.__dict__.items()))
        command = (f"INSERT INTO sample ({', '.join(columns)}) "
                   f"VALUES ({', '.join(['?' for _ in columns])})")
        return command, values
