# coding: utf-8

'''
Input: ChargeSessions_raw.csv

This file reads the downloaded CSV from DWH and convets
the RFIDs to identify car2go and other user types.

Output: ChargeSessions_raw.pkl
'''

import pandas as pd
import os 

cwd = os.getcwd()
df = pd.read_csv(cwd + "/ChargeSessions_raw.csv")

df_info = pd.read_csv(cwd + "/RFID_DETAILS.csv")
df_info = df_info.drop(df_info.columns[0], axis=1)
df_info = df_info.drop(df_info.columns[2], axis=1)

df_merged = pd.merge(df, df_info, on="RFID")
df_merged['RFID'] = df_merged.RFID.astype(str)
df_merged = df_merged.rename(index=str, columns={"UseType_y": "UseType"})

def process_ids(row):
    if row['UseType'] == 'Car2GO':
        new_id = 'car2go' 
        return new_id
    elif row['UseType'] == 'SHARE2USE':
        new_id = 'SHARE2USE_' + row.RFID 
        return new_id 
    elif row['UseType'] == 'UBER':
        new_id = 'UBER_' + row.RFID 
        return new_id 
    elif row['UseType'] == 'taxi':
        new_id = 'taxi_' + row.RFID 
        return new_id 
    elif row['UseType'] == 'connexxion':
        new_id = 'connexxion_' + row.RFID 
        return new_id     
    else:
        new_id = 'car_' + row.RFID
        return new_id

df_merged['RFID'] = df_merged.apply(process_ids, axis=1)
df_merged.to_pickle("ChargeSessions_raw.pkl")