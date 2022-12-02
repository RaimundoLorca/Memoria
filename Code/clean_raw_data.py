# -*- coding: utf-8 -*-
"""
"""
# =============== Imports ===============
from pathlib import Path

import pandas as pd

from moncon.pipelines.rawdata_process import clean_data

# =============== Script Parameters ===============
# Settings
SAVE = True
YEAR = 2022
FREQ = None
# Column names
TIMECOL = "Timestamp"
TAGS_ID_COL = "pump_id"
# Folders and files paths
MAIN_DATA_FOLDER = Path("data")
CLEAN_DATA_FOLDER = MAIN_DATA_FOLDER.joinpath("data_clean")
TAGS_FILE = MAIN_DATA_FOLDER.joinpath("DrivePumps_tags.xlsx")
RAW_DATA_FOLDER = MAIN_DATA_FOLDER.joinpath("raw_data")
FILE_NAME = "DrivePumps2_pruebadescarga.xlsx"



# =============== Preprocessing ===============
raw_data_xls = pd.ExcelFile(
    RAW_DATA_FOLDER.joinpath(
        FILE_NAME
    )
)

for sheet in raw_data_xls.sheet_names:
    print(f"{sheet}")
    # Load pump tags and raw data excel files
    tags_Serie = pd.read_excel(
        TAGS_FILE,
        index_col=TAGS_ID_COL
    ).loc[sheet].dropna()
    # Load raw data 
    raw_data = pd.read_excel(
        raw_data_xls,
        sheet_name=sheet
    )

    data = clean_data(
        raw_data=raw_data,
        tags_Serie=tags_Serie,
        timecol=TIMECOL,
        raw_data_folder=RAW_DATA_FOLDER,
        save=SAVE,
        asset_id=sheet,
        clean_data_folder=CLEAN_DATA_FOLDER,
    ) 
    
    unique_cols = pd.Series(data.columns).unique()

    print(f"{sheet} ready, total sensors: {data.shape[1]}")
    print(f"Total unique columns: {unique_cols.shape[0]}")
        
        
        
