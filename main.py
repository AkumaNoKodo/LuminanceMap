import time
from sample import Sample
from testting_file import ParallelInserter

if __name__ == "__main__":
    st = time.time()
    folder = r"C:\Users\marku\Downloads\Kalibrace_NIKON_90D\Test"
    inserter = ParallelInserter("test.db", Sample())
    inserter.insert_from_folder(folder, "nef")
    print(f"{time.time() - st}")





