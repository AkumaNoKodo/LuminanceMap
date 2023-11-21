from multiprocessing import Process, Manager, Lock
import sqlite3
import zlib
import re
import numpy as np
from imageProcessing import find_files_with_extension


class DatabaseInserter:
    def __init__(self, database_file_path: str):
        self.database_path = database_file_path
        self.queue_maxsize = 6
        self.progress_output_lock = Lock()
        self.manager = Manager()
        self.queue = self.manager.Queue(self.queue_maxsize)

    def producer(self, items):
        for item in items:
            job_product = self.job_producer(item)
            self.queue.put(job_product)
            self.print_progress("Producing")
        self.queue.put(None)

    def print_progress(self, message):
        with self.progress_output_lock:
            print(f"{message} --> {self.queue.qsize()}/{self.queue_maxsize}")

    def consumer(self):
        db = sqlite3.connect(self.database_path)
        sqlite3.register_adapter(np.ndarray, self._adapt_nparray)
        sqlite3.register_converter("NPARRAY", self._convert_nparray)
        self.create_new_table_for_class(db)

        while True:
            item = self.queue.get()
            if item is None:
                self.queue.task_done()
                self.print_progress("Done all")
                break
            self.job_consumer(item, db)
            self.queue.task_done()
            self.print_progress("Done")
        db.close()

    @staticmethod
    def job_producer(item):
        file_reader = RawImageReader(item)
        return file_reader.load_data()

    def job_consumer(self, item, db):
        self.insert_new_row(item, db)

    def insert_from_folder(self, target_folder, extension):
        files = find_files_with_extension(target_folder, extension)
        producer_process = Process(target=self.producer, args=(files,))
        consumer_process = Process(target=self.consumer)

        producer_process.start()
        consumer_process.start()

        producer_process.join()
        consumer_process.join()

    @staticmethod
    def _adapt_nparray(arr):
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
    def _convert_nparray(text):
        is_compressed = text[0] == ord('1')
        length = int.from_bytes(text[1:5], 'big')
        shape_dtype_str, data = text[5:5 + length].decode(), text[5 + length:]
        shape_str, dtype_str = re.search(r"(\(.*\))(\[.*])", shape_dtype_str).groups()
        shape = tuple(map(int, re.findall(r"\d+", shape_str)))
        dtype = np.dtype(eval(dtype_str))
        if is_compressed:
            data = zlib.decompress(data)
        return np.frombuffer(data, dtype=dtype).reshape(shape)

    def create_new_table_for_class(self, db):
        command = self.target_class.new_table_command()
        cursor = db.cursor()
        cursor.execute(command)
        db.commit()

    def insert_new_row(self, item, db):
        command, values = item.new_insert_command()
        cursor = db.cursor()
        cursor.execute(command, values)
        db.commit()

