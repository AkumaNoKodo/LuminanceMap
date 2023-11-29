import pickle
import shutil
from concurrent.futures import ProcessPoolExecutor
from pprint import pprint
import numpy as np
import psutil
from image2 import Image, ImageLoader


def calibration_statistic_data_from_sets_folder(folder: str) -> None:
    loader = ImageLoader(True)
    all_images = loader.load_images_sets_from_folder(folder)
    stat = get_statistic_data_exposure_time(all_images)

    with open("exposure_time_stat.pkl", "wb") as a_file:
        pickle.dump(stat, a_file)

    shutil.rmtree("TEMP")
    pprint(stat)


def get_statistic_data_exposure_time(data: Image | list[Image]):
    stat = {}
    data = group_samples_per_tags(data)
    data = get_statistic_data_per_exposure_time(data)
    stat["per_exposure_time"] = data.copy()
    data1 = complete_statistic_data_per_exposure_time(data)
    stat["per_exposure_time_complete"] = data1.copy()
    data2 = complete_statistic_data_per_f_number(data1)
    stat["per_f_number"] = data2.copy()
    data3 = complete_statistic_data_per_iso(data2)
    stat["per_iso"] = data3.copy()

    return stat


def group_samples_per_tags(data: Image | list[Image]):
    grouped = {}
    for record in data:
        tag = record.tag
        iso = record.iso
        f_number = record.f_number
        exposure_time = record.exposure_time

        if tag not in grouped:
            grouped[tag] = {}

        if iso not in grouped[tag]:
            grouped[tag][iso] = {}

        if f_number not in grouped[tag][iso]:
            grouped[tag][iso][f_number] = {}

        if exposure_time not in grouped[tag][iso][f_number]:
            grouped[tag][iso][f_number][exposure_time] = {}

        grouped[tag][iso][f_number][exposure_time] = record
    return grouped


def group_samples_per_iso(data):
    grouped = {}
    for record in data:
        iso = record["iso"]
        f_number = record["f_number"]
        exposure_time = record["A_exposure_time"]

        if iso not in grouped:
            grouped[iso] = {}

        if f_number not in grouped[iso]:
            grouped[iso][f_number] = {}

        if exposure_time not in grouped[iso][f_number]:
            grouped[iso][f_number][exposure_time] = []

        grouped[iso][f_number][exposure_time].append(record)
    return grouped


def prepare_data(source, key) -> np.ndarray | None:
    image = source[key]
    data = image.rescale_cfa
    data[(data < 0.15) | (data > 0.95)] = np.nan
    return data


def prepare_combination(data: dict):
    combinations = []
    for tag, iso_dict in data.items():
        for iso, f_number_dict in iso_dict.items():
            for f_number, exposure_times_dict in f_number_dict.items():
                sorted_keys = sorted(exposure_times_dict.keys(), key=lambda x: float(x))
                for a, b in zip(sorted_keys[:-1], sorted_keys[1:]):
                    combinations.append([tag, iso, f_number, exposure_times_dict, a, b])

    return combinations


def get_relation_statistics(combination):
    tag, iso, f_number, exposure_times_dict, a, b = combination
    a_data = prepare_data(exposure_times_dict, a)
    b_data = prepare_data(exposure_times_dict, b)

    if a_data is None or b_data is None:
        return None

    relation = np.abs(a_data / b_data)
    std_of_relation = np.nanstd(relation, axis=(0, 1))
    if np.any(std_of_relation > 0.15) or np.any(std_of_relation < 0.001) or np.any(
            np.isnan(std_of_relation)):
        return None

    not_nan_count = np.count_nonzero(~np.isnan(relation))
    good_percent_count = 100 * not_nan_count / relation.size
    if good_percent_count < 1:
        return None

    calibration_set = {
        "tag": tag,
        "iso": iso,
        "f_number": f_number,
        "A_exposure_time": a,
        "B_exposure_time": b,
        "mean_of_relation": np.nanmean(relation, axis=(0, 1)),
        "std_of_relation": std_of_relation,
        "set_size": relation.size,
        "not_nan_size": not_nan_count,
        "good_pixel_percent": f"({not_nan_count} / {relation.size}) = {good_percent_count:.3f}"
    }
    print(calibration_set["mean_of_relation"],
          calibration_set["std_of_relation"],
          calibration_set["good_pixel_percent"])
    return calibration_set


def get_statistic_data_per_exposure_time(data: dict) -> list:
    combinations = prepare_combination(data)
    with ProcessPoolExecutor(max_workers=psutil.cpu_count(logical=False)) as executor:
        print(f"Start multiprocessing statistic combinations test: Total combinations for execute {len(combinations)}")
        calibration_sets = list(executor.map(get_relation_statistics, combinations))

    return [calibration_set for calibration_set in calibration_sets if calibration_set is not None]


def complete_statistic_data_per_exposure_time(samples):
    samples = group_samples_per_iso(samples)
    for iso, f_number_dict in samples.items():
        for f_number, exposure_time_dict in f_number_dict.items():
            for exposure_time, exposure_time_list in exposure_time_dict.items():
                means = [item["mean_of_relation"] for item in exposure_time_list]
                stds = [item["std_of_relation"] for item in exposure_time_list]
                weights = [item["not_nan_size"] for item in exposure_time_list]

                samples[iso][f_number][exposure_time] = get_weighted_mean_and_std(means, stds, weights)
    return samples


def complete_statistic_data_per_f_number(samples):
    for iso, f_number_dict in samples.items():
        for f_number, exposure_time_dict in f_number_dict.items():
            means = [item["mean"] for item in exposure_time_dict.values()]
            stds = [item["std"] for item in exposure_time_dict.values()]
            weights = [item["set_size"] for item in exposure_time_dict.values()]

            samples[iso][f_number] = get_weighted_mean_and_std(means, stds, weights)
    return samples


def complete_statistic_data_per_iso(samples):
    for iso, f_number_dict in samples.items():
        means = [item["mean"] for item in f_number_dict.values()]
        stds = [item["std"] for item in f_number_dict.values()]
        weights = [item["set_size"] for item in f_number_dict.values()]

        samples[iso] = get_weighted_mean_and_std(means, stds, weights)
    return samples


def get_weighted_mean_and_std(means: list[float], stds: list[float], weights: list[int]) -> dict:
    """
    Calculates the weighted mean and weighted standard deviation using NumPy,
    converting input lists to NumPy arrays.

    Args:
    means (list[float]): List of mean values.
    stds (list[float]): List of standard deviations.
    weights (list[int]): List of weights corresponding to the sample sizes.

    Returns:
    dict: A dictionary containing the weighted mean, weighted standard deviation,
          and the total sample size.
    """

    means_np = np.array(means)
    stds_np = np.array(stds)
    weights_np = np.array(weights)

    # Calculate the weighted mean using NumPy's average function with weights
    weighted_mean = np.average(means_np, weights=weights_np)

    # Calculate the partial variances for each element
    partial_variances = (weights_np - 1) * stds_np ** 2 + weights_np * (means_np - weighted_mean) ** 2

    # Sum up the partial variances and divide by the sum of weights to get the weighted variance
    weighted_variance = np.sum(partial_variances) / np.sum(weights_np)

    # The weighted standard deviation is the square root of the weighted variance
    weighted_std = np.sqrt(weighted_variance)

    # Return the results in a dictionary
    return {"mean": weighted_mean, "std": weighted_std, "set_size": np.sum(weights_np)}
