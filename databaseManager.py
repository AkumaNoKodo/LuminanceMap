import zlib
import re
import numpy as np
from sample import Sample
from image import RawImageReader
from imageProcessing import find_files_with_extension
import sqlite3
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import Queue
import threading


class DatabaseManager:
    def __init__(self, filename: str):
        sqlite3.register_adapter(np.ndarray, self._adapt_nparray)
        sqlite3.register_converter("NPARRAY", self._convert_nparray)

        self.database = sqlite3.connect(filename, detect_types=sqlite3.PARSE_DECLTYPES)
        self.table_name = "raw_samples"

        sample = Sample()
        sample.create_new_table(self)

    def __del__(self):
        self.database.commit()
        self.database.close()

    def insert_sample(self, sample) -> None:
        sample.insert_to_database(self)

    def insert_from_folder(self, folder: str, extension: str) -> None:
        files = find_files_with_extension(folder, extension)
        # inserter = ParallelInserter(files, self)
        # inserter.process_files()

    def load_samples_from_ids(self, row_ids: list[int]) -> list[Sample]:
        cursor = self.database.cursor()
        if 0 in row_ids:
            sql_command = f"SELECT * FROM {self.table_name}"
            cursor.execute(sql_command)
        else:
            placeholders = ','.join('?' for _ in row_ids)
            sql_command = f"SELECT * FROM {self.table_name} WHERE id IN ({placeholders})"
            cursor.execute(sql_command, row_ids)

        rows = cursor.fetchall()

        if not rows:
            print(f"No rows found with ids {row_ids}")
            return []

        samples = []
        columns = [desc[0] for desc in cursor.description]

        for row in rows:
            new_option = Sample()
            for column, value in zip(columns, row):
                if hasattr(new_option, column):
                    getattr(new_option, column).data = value
            samples.append(new_option)

        return samples

    @staticmethod
    def _adapt_nparray(arr) -> bytes:
        data_bytes = arr.tobytes()
        shape_dtype_str = f"{arr.shape}{arr.dtype.descr}"
        shape_dtype_bytes = shape_dtype_str.encode()
        length = len(shape_dtype_bytes)
        header = length.to_bytes(4, 'big') + shape_dtype_bytes
        if arr.size > 1000:
            compressed = zlib.compress(data_bytes)
            return b"1" + header + compressed  # "1" state with compression
        else:
            return b"0" + header + data_bytes  # "0" state without compression

    @staticmethod
    def _convert_nparray(text) -> np.ndarray:
        is_compressed = text[0] == ord('1')
        length = int.from_bytes(text[1:5], 'big')
        shape_dtype_str, data = text[5:5 + length].decode(), text[5 + length:]
        shape_str, dtype_str = re.search(r"(\(.*\))(\[.*])", shape_dtype_str).groups()
        shape = tuple(map(int, re.findall(r"\d+", shape_str)))
        dtype = np.dtype(eval(dtype_str))
        if is_compressed:
            data = zlib.decompress(data)
        return np.frombuffer(data, dtype=dtype).reshape(shape)
