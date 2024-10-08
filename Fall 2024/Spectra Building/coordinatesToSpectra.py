# Skye Goetz 10/06/24

import os
import re
import sys
import subprocess
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
from pyspark.sql import DataFrame

FILE_NAME = os.path.basename(sys.argv[0])
BIN = os.path.dirname(sys.argv[0])
CWD = os.getcwd()

spark = SparkSession.builder \
    .appName("Haynes Lab Spark Spectra") \
    .getOrCreate()

def main(INPUT_FILE: str, CHARGE: str, MULTIPLICITY: str) -> None: 

    FIRST_XYZ = []
    with open(INPUT_FILE, "rt") as FIRST_COORDINATES:
        
        for line in FIRST_COORDINATES:
            FIRST_XYZ.append(line.strip())
    
    FIRST_ORCA_INPUT_FILE = "orca.opt." + \
        os.path.splitext(os.path.basename(INPUT_FILE))[0] + ".inp"
    FIRST_ORCA_OUTPUT_FILE = "orca.opt." + \
        os.path.splitext(os.path.basename(INPUT_FILE))[0] + ".out"

    FIRST_ORCA_COMMANDS_BEFORE_COORDINATES = [
        "! PBE0 Def2-SVPD TightOpt SMD(ethanol)\n",
        "\n",
    ]

    CHARGE_MULTIPLICITY = "* xyz " + CHARGE + " " + MULTIPLICITY + "\n" 

    FIRST_ORCA_INPUT_PATH = os.path.join(CWD, FIRST_ORCA_INPUT_FILE)

    with open(FIRST_ORCA_INPUT_PATH, "w") as FIRST_QUERY:

        for FIRST_COMMAND_BEFORE_COORDINATES in FIRST_ORCA_COMMANDS_BEFORE_COORDINATES:
            FIRST_QUERY.write(FIRST_COMMAND_BEFORE_COORDINATES)

        FIRST_QUERY.write(CHARGE_MULTIPLICITY)

        for FIRST_COORDINATE in FIRST_XYZ:
            FIRST_QUERY.write(FIRST_COORDINATE + "\n")

        FIRST_QUERY.write("*\n")

    if not os.path.isfile(FIRST_ORCA_OUTPUT_FILE): 
        try:

            subprocess.run(["orca", FIRST_ORCA_INPUT_PATH], \
                stdout=open(FIRST_ORCA_OUTPUT_FILE, "w"), check=True)
        except subprocess.CalledProcessError as e:

            print(f"error.CRITICAL | ORCA execution failed: {e}")
            sys.exit()

    FIRST_QUERY_XYZ_FILE = os.path.splitext(os.path.basename \ 
        (FIRST_ORCA_OUTPUT_FILE))[0] + ".xyz"

    SECOND_XYZ = []
    with open(FIRST_QUERY_XYZ_FILE, "rt") as SECOND_COORDINATES:
        lineCount = -1
        for line in SECOND_COORDINATES:
            lineCount += 1
            if lineCount > 1: 
                SECOND_XYZ.append(re.sub(r"^\s+|\s+$", "", line))

    SECOND_ORCA_INPUT_FILE = "orca.tddft." + \
        os.path.splitext(os.path.basename(INPUT_FILE))[0] + ".inp"
    SECOND_ORCA_OUTPUT_FILE = "orca.tddft." + \
        os.path.splitext(os.path.basename(INPUT_FILE))[0] + ".out"

    SECOND_ORCA_COMMANDS_BEFORE_COORDINATES = [
        "! CAM-B3LYP Def2-TZVPD CPCM(ethanol)\n"
        "\n"
        "%tddft\n"
        "  nroots 10\n"
        "  maxdim 100\n"
        "end\n"
        "\n"
    ]

    SECOND_ORCA_INPUT_PATH = os.path.join(CWD, SECOND_ORCA_INPUT_FILE)

    with open(SECOND_ORCA_INPUT_PATH, "w") as SECOND_QUERY:

        for SECOND_COMMAND_BEFORE_COORDINATES in SECOND_ORCA_COMMANDS_BEFORE_COORDINATES:
            SECOND_QUERY.write(SECOND_COMMAND_BEFORE_COORDINATES)

        SECOND_QUERY.write(CHARGE_MULTIPLICITY)

        for SECOND_COORDINATE in SECOND_XYZ:
            SECOND_QUERY.write(SECOND_COORDINATE + "\n") 

        SECOND_QUERY.write("*\n")

    if not os.path.isfile(SECOND_ORCA_OUTPUT_FILE): 

        try:

            subprocess.run(["orca", SECOND_ORCA_INPUT_PATH], \
                stdout=open(SECOND_ORCA_OUTPUT_FILE, "w"), check=True)
        except subprocess.CalledProcessError as e:

            print(f"error.CRITICAL | ORCA execution failed: {e}")
            sys.exit()

    STRING_TO_BEGIN_SECTION = "ABSORPTION SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS"
    STRING_TO_END_SECTION = "CD SPECTRUM"

    WAVELENGTH = []
    FOSC = []

    isCorrectSection = False
    with open(SECOND_ORCA_OUTPUT_FILE, "rt") as SECOND_OUTPUT:
        for line in SECOND_OUTPUT:
        
            if STRING_TO_BEGIN_SECTION in line: 
                isCorrectSection = True
            elif STRING_TO_END_SECTION in line: 
                isCorrectSection = False

            if isCorrectSection:
                LINE_LIST = line.split()

                if len(LINE_LIST) > 5:
                
                    try:

                        wavelength = float(LINE_LIST[5])
                        fosc = float(LINE_LIST[6])
                        WAVELENGTH.append(wavelength)
                        FOSC.append(fosc)

                    except ValueError: continue

    DATA = {
    
        "Compound": (os.path.splitext(os.path.basename(INPUT_FILE))[0]), 
        "Wavelength": WAVELENGTH,
        "Fosc": FOSC
        
    }

    if WAVELENGTH and FOSC:

        SPARK_SPECTRA = spark.createDataFrame([(DATA["Compound"], w, f) \
            for w, f in zip(WAVELENGTH, FOSC)], ["Compound", "Wavelength", "Fosc"])

        PARQUET_FILE = os.path.join(BIN, "sparkSpectra.parquet")
        
        if os.path.exists(PARQUET_FILE):

            EXISTING_SPARK_SPECTRA = spark.read.parquet(PARQUET_FILE)
            COMBINED_SPARK_SPECTRA = EXISTING_SPARK_SPECTRA.union(SPARK_SPECTRA).dropDuplicates()
            COMBINED_SPARK_SPECTRA.write.mode("overwrite").parquet(PARQUET_FILE)

        else:

            SPARK_SPECTRA.write.parquet(PARQUET_FILE)

    else:
    
        print(f"error.CRITICAL | NO ABSORBANCE DATA FOUND")
        sys.exit()

    X = SPARK_SPECTRA.select("Wavelength").rdd.flatMap(lambda x: x).collect()
    Y = SPARK_SPECTRA.select("Fosc").rdd.flatMap(lambda x: x).collect()

    plt.bar(X, Y, width=0.5, color="blue", edgecolor="blue")
    plt.title("Absorbance Spectra")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel(r"Fosc $(au^{2})$")
    plt.xticks(rotation=45) 
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    PNG_FILE = os.path.join(CWD, \
        (os.path.splitext(os.path.basename(INPUT_FILE))[0] + ".png"))

    plt.savefig(PNG_FILE, format="png")

try:
    
    ARGUMENT_ONE = sys.argv[1]
    ARGUMENT_TWO = sys.argv[2]
    ARGUMENT_THREE = sys.argv[3]

    main(ARGUMENT_ONE, ARGUMENT_TWO, ARGUMENT_THREE)

except IndexError:

    print(f"error.CRITICAL | USAGE | python3 \
        {FILE_NAME} <INPUT_FILE> <CHARGE> <MULTIPLICITY>")
        
    sys.exit()
