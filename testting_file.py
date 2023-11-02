import zlib
import re
from image import RawImageReader
from sample import Sample
from imageProcessing import find_files_with_extension
from queue import Queue
import threading
import sqlite3
import numpy as np


class ParallelInserter:
    def __init__(self, database_file_path: str, target):
        sqlite3.register_adapter(np.ndarray, self._adapt_nparray)
        sqlite3.register_converter("NPARRAY", self._convert_nparray)

        self.database_path = database_file_path
        self.queue_maxsize = 6
        self.target_class = target
        self.local_thread = threading.local()
        self.progress_output_lock = threading.Lock()
        self.queue = Queue(maxsize=self.queue_maxsize)

    def producer(self, items) -> None:
        for item in items:
            job_product = self.job_producer(item)
            self.queue.put(job_product)
            self.print_progress("Producing")
        self.queue.put(None)

    def print_progress(self, message: str) -> None:
        with self.progress_output_lock:
            print(f"{message} --> {self.queue.qsize()}/{self.queue_maxsize}")

    def consumer(self) -> None:
        if not hasattr(self.local_thread, "database"):
            self.local_thread.database = sqlite3.connect(self.database_path)
        self.create_new_table_for_class()
        break_condition = True
        while break_condition:
            items = []
            for _ in range(self.queue.qsize()):
                items.append(self.queue.get())
                self.queue.task_done()

            if None in items:
                items = [item for item in items if item is not None]
                break_condition = False

            if items:
                self.print_progress("Done")
                self.job_consumer(items)

    @staticmethod
    def job_producer(item: str) -> Sample:
        file_reader = RawImageReader(item)
        return file_reader.load_data()

    def job_consumer(self, item) -> None:
        self.insert_new_rows(item)

    def insert_from_folder(self, target_folder: str, extension: str) -> None:
        self.local_thread.database = sqlite3.connect(self.database_path, detect_types=sqlite3.PARSE_DECLTYPES)

        files = find_files_with_extension(target_folder, extension)
        producer_thread = threading.Thread(target=self.producer, args=(files,))
        consumer_thread = threading.Thread(target=self.consumer)
        producer_thread.start()
        consumer_thread.start()
        producer_thread.join()
        self.queue.join()

        self.local_thread.database.commit()
        self.local_thread.database.close()

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

    def create_new_table_for_class(self) -> None:
        command = self.target_class.new_table_command()
        cursor = self.local_thread.database.cursor()
        cursor.execute(command)
        cursor.close()
        self.local_thread.database.commit()

    def insert_new_rows(self, items: [Sample]) -> None:
        if not items:
            return

        command, _ = items[0].new_insert_command()
        values_list = [item.new_insert_command()[1] for item in items]

        cursor = self.local_thread.database.cursor()
        cursor.executemany(command, values_list)
        cursor.close()
        self.local_thread.database.commit()


# st = time.time()
# folder = r"C:\Users\marku\Downloads\Kalibrace_NIKON_90D\Test"
# inserter = ParallelInserter("test.db", Sample())
# inserter.insert_from_folder(folder, "nef")
# print(f"{time.time() - st}")
