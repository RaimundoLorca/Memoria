# -*- coding: utf-8 -*-
"""

"""
from pathlib import Path

import pandas as pd

def clean_data(
        raw_data,
        tags_Serie,
        timecol,
        raw_data_folder,
        save,
        data_source="PI",
        asset_id=None,
        clean_data_folder=None,
        freq=None
):
    """
    Cleans empty rows and columns, and change data's column names as TAGs
    --------------------------------------------------------------------------
    Parameters
    
    raw_data
    
    timecol: (str)
    
    
    """
    # Drop empty cols (empty if more than 99% of values are NaN)
    data = drop_unuseful_cols(raw_data)
    
    # Set DataFrame columns as sensor tags
    data.columns = get_tags(
        data=data,
        timecol=timecol,
        data_source=data_source
    )
    # Drop uneuseful cols. Due to the format stablished for IMA-MCM-Calmly team
    # rows from 0 to 4 have uneuseful cols, unless this download data format
    # is changed this drop will work.
    data.drop(
        [0, 1, 2, 3, 4],
        inplace=True
    )
    # Set the timecol as DataFrame's index
    data.set_index(
        timecol,
        inplace=True
    )
    data = set_sensor_names(
            data=data,
            tags_Serie=tags_Serie
    )
    
    
    
    if save:
        if isinstance(asset_id, str) and isinstance(clean_data_folder, Path):
            save_data(
                    data=data,
                    clean_data_folder=clean_data_folder,
                    asset_id=asset_id,
                    timecol=timecol,
                    freq=freq
            )
        else:
            msg = "Missing asset_id and/or clean_data_folder"
            raise NotImplementedError(msg)
            
        
    return data

def drop_unuseful_cols(
        raw_data
):
    cols2drop = [
        col for col in raw_data.columns\
            if not isinstance(raw_data.loc[4, col], str)
    ]
    cols2drop.pop(1) 
    data = raw_data.drop(
        columns=cols2drop
    )
    return data

def drop_empty_cols(
        raw_data
):
    """
    Drop all columns that have greater than 99% of NaNs.
    ---------------------------------------------------------------------------
    Parameters
    
    raw_data:
    --------------------------------------------------------------------------
    """
    cols2drop = [
        col for col in raw_data.columns\
            if (raw_data[col].isna().sum()/raw_data.shape[0]) > 0.99
    ]
    data = raw_data.drop(
        columns=cols2drop
    )
    return data

def set_sensor_names(
        data,
        tags_Serie
):
    tags_dict = {}
    for sensor_tag, sensor_name in tags_Serie.to_dict().items():
        tags_dict[sensor_name] = sensor_tag
    data.rename(
        columns=tags_dict,
        inplace=True
    )
    return data

def save_data(
        data,
        clean_data_folder,
        asset_id,
        timecol,
        freq=None
):
    """
    Saves new data and check for duplicates on old data if exists. Also moves
    the raw data excel file to another folder.
    --------------------------------------------------------------------------
    Parameters
    
    
    
    
    """
    # Load old preprocessed data and concat to the current data if
    # exists, otherwise pass
    try:
        old_data = pd.read_pickle(
            clean_data_folder.joinpath(
                file_name(asset_id, freq)
            )
        )
        data = pd.concat(
            [old_data, data],
        )
    except FileNotFoundError:
        pass

    # Drop duplicates timestamps (if exists)
    data = data.reset_index(
    ).drop_duplicates(
        subset=timecol,
        keep="first"
    ).set_index(
        timecol
    )
    # Save data
    data.to_pickle(
        clean_data_folder.joinpath(
            file_name(asset_id, freq)
        )
    )


def get_tags(
        data,
        timecol,
        data_source
):
    """
    Returns as list the file's TAGs
    -------------------------------------------------------------------------
    Parameters
    
    data
    
    timecol
    
    """
    # (Fourth row in raw data has sensor names)
    new_cols = data.iloc[[4]].values[0].tolist()
    new_cols = [
        col_name for col_name in new_cols if isinstance(col_name, str)
    ]
    n_cols = data.shape[1]
    
    if data_source == "uniformance":
        # Iter over TAGs and errase " - Snapshot"
        new_cols = [
            col_name.replace(" - Snapshot", "") if " - Snapshot" in col_name\
                else col_name for col_name in new_cols
        ]
    elif data_source == "PI":
        pass
    else:
        raise NotImplementedError("Data source not implemented")
        

          
    if n_cols == len(new_cols):
        return new_cols
    elif n_cols == len(new_cols) + 1 and timecol not in new_cols:
        new_cols.insert(0, timecol)
        return new_cols
    
    else: 
        # (n_cols > len(new_cols) + 1) or 
        # (n_cols == len(new_cols) + 1 and timecol in new_cols)
        raise NotImplementedError("Missing TAG")


def file_name(
        asset_id,
        freq=None
):
    """
    Returns preprocessed file name. If "freq" argument is given, includes the
    sample frequency on the name.
    --------------------------------------------------------------------------
    Parameters
    
    """
    if isinstance(freq, str):
        file_name = f"data_{asset_id}_{freq}.pkl"
    else:
        file_name = f"data_{asset_id}.pkl"
    return file_name
        
    
    