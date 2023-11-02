import numpy as np
from dataclasses import dataclass
from typing import Union, Type


@dataclass
class Data:
    python_type: Union[Type[str], Type[float], Type[np.ndarray]]
    expected_size: Union[tuple, int] = None
    dtype: str = "float64"
    _data_name: str = None
    _database_type: str = None
    _data: Union[str, float, np.ndarray] = None

    @property
    def data_name(self):
        return self._data_name

    @property
    def database_type(self):
        return self._database_type

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        if not (value is None or isinstance(value, self.python_type)):
            print(f"Invalid data type -- {value.__class__.__name__}")
            return
        self._data = value

    def __post_init__(self):
        python_database_type_pairs = {"str": "TEXT",
                                      "float": "REAL",
                                      "ndarray": "NPARRAY"
                                      }

        self._database_type = python_database_type_pairs[self.python_type.__name__]

    def column_command(self) -> str:
        return f"{self._data_name} {self._database_type}"

