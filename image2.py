import os
import subprocess
import json
import pprint
import tempfile

import numpy as np


def get_metadata(filename):
    # Spustí ExifTool a získá metadata ve formátu JSON
    result = subprocess.run(["C:\\Users\\marku\\PycharmProjects\\LuminanceMap\\ExifTool\\exiftool.exe", "-j", filename],
                            text=True, capture_output=True)

    # Kontroluje, zda byl příkaz úspěšný
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return None

    # Převede výstup na slovník Pythonu
    metadata = json.loads(result.stdout)[0]

    return metadata


def extract_linearization_table(filename):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        # Spustí ExifTool s přepínačem -b a extrahuje NEFLinearizationTable do dočasného souboru
        result = subprocess.run(["C:\\Users\\marku\\PycharmProjects\\LuminanceMap\\ExifTool\\exiftool.exe",
                                 "-b", "-NEFLinearizationTable", filename],
                                stdout=temp_file)

        # Kontroluje, zda byl příkaz úspěšný
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return None

    # Načte binární data z dočasného souboru
    with open(temp_file.name, 'rb') as file:
        binary_data = file.read()

    # Odstraní dočasný soubor
    os.unlink(temp_file.name)

    # Konvertuje binární data na numpy pole celých čísel
    int_array = np.frombuffer(binary_data, dtype=np.uint16)

    return int_array


# Použití
filename = r"C:\Users\marku\Downloads\Kalibrace_NIKON_90D\Kalibrace_NIKON_90D_1\IMG_55.nef"
linearization_table = extract_linearization_table(filename)

# Vypíše prvních 10 hodnot z tabulky linearizace
print(linearization_table[:10])