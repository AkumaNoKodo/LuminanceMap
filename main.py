from itertools import groupby
from operator import itemgetter
from pprint import pprint
import pickle
from image import RawImage, RawImageConvertor
from imageProcessing import find_files_with_extension

if __name__ == "__main__":
    folder = r"C:\Users\marku\Downloads\Kalibrace_NIKON_90D\Kalibrace_NIKON_90D_1"
    raw_files = find_files_with_extension(folder, "nef")

    # # raw_files = raw_files[2:3]
    # x = [RawConvertor.apply_post_processing_rgb_algorythm(RawImage(x)) for x in raw_files]
    # x = sorted(x, key=lambda y: (y["iso"], y["f_number"], y["exposure_time"]))
    # grouped = {iso: {f_number: list(items_f)
    #                  for f_number, items_f in groupby(items_iso, key=itemgetter('f_number'))}
    #            for iso, items_iso in groupby(x, key=itemgetter('iso'))}
    #
    # with open('large_dict.pkl', 'wb') as f:
    #     pickle.dump(grouped, f)
    #
    # with open('large_dict.pkl', 'rb') as f:
    #     grouped = pickle.load(f)
    #
    # pprint(grouped)
