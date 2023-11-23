from pprint import pprint
from calibration import complete_statistic_data_per_exposure_time, complete_statistic_data_per_f_number, \
    complete_statistic_data_per_iso, get_statistic_data_per_exposure_time, group_samples_per_tags
from image import TempRawImages

if __name__ == "__main__":
    extension = "nef"
    folder = r"C:\Users\marku\Downloads\Kalibrace_NIKON_90D"
    images = TempRawImages()
    images.create_from_folder(folder, extension)
    data = group_samples_per_tags(images.temp_files_list)
    data = get_statistic_data_per_exposure_time(data)
    # file = "all_image_statistic2023_11_23_01_28_20_996298.pkl"
    # with open(file, "rb") as a_file:
    #     data = pickle.load(a_file)
    data = complete_statistic_data_per_exposure_time(data)
    data = complete_statistic_data_per_f_number(data)
    data = complete_statistic_data_per_iso(data)
    pprint(data)
