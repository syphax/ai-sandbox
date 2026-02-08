# %%
# FCP Parts Summarizer

import os
import pathlib

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import re
import json
import time
import datetime
import warnings

from pathlib import Path

#warnings.filterwarnings('ignore')

# %%
# Parameters

path = Path('..') / 'data'
fn = 'fcp-euro-parts.csv'

# %%
# Read in the data
df_fcp = pd.read_csv(path / fn)

print(f'df_fcp has {len(df_fcp):,} rows and {len(df_fcp.columns)} columns.')

df_cnts_by_brand = df_fcp.groupby(['source_brand'], as_index=False).agg(cnt_parts=('sku','nunique'), cnt_rows=('sku','count')).sort_values('cnt_parts', ascending=False)

display(df_cnts_by_brand)

# %%
# Add rank by product line
# This logic assumes the data is ranked by popularity/volume by product line

df_fcp['rank'] = df_fcp.groupby((df_fcp['product_line'] != df_fcp['product_line'].shift()).cumsum()).cumcount() + 1
df_fcp['rank_pct'] = ((df_fcp['rank'] - 1) / (df_fcp.groupby((df_fcp['product_line'] != df_fcp['product_line'].shift()).cumsum())['rank'].transform('max') - 1)).fillna(0)

agg_pn = dict(
    cnt_parts=('sku', 'nunique'),
    rank_pct_2th=('rank_pct', lambda x: x.quantile(0.02)),
    rank_pct_5th=('rank_pct', lambda x: x.quantile(0.05)),
    rank_pct_50th=('rank_pct', lambda x: x.quantile(0.50)),
    rank_pct_95th=('rank_pct', lambda x: x.quantile(0.95)),
)

# %%
# Fill in missing types

df_fcp['part_type_clean'] = np.where(df_fcp['part_type'].isna(), 'Not Specified', df_fcp['part_type'])

list_genuine = ['Genuine Audi', 'Genuine BMW', 
'Genuine European BMW', 
'Genuine Jaguar', 'Genuine Land Rover', 
'Genuine Mercedes', 'Genuine Mini', 
'Genuine Porsche', 'Genuine Saab', 
'Genuine Volvo', 'Genuine VW', 
'Genuine VW Audi', 'Mercedes StarParts']

df_fcp['part_type_clean'] = np.where(df_fcp['brand'].isin(list_genuine), 'Genuine', df_fcp['part_type_clean'])


# %%
# Get the unique part numbers
part_numbers = df_fcp['sku'].unique()
print(f'There are {len(part_numbers):,} unique part numbers.')

# %%
# Get the unique part categories
product_lines = df_fcp['product_line'].unique()
print(f'There are {len(product_lines)} unique part categories.')

# %% 
# Examine brand vs part type:

df_fcp_cnts_br_pt = df_fcp.groupby(['brand','part_type_clean'],
                                   as_index=False, dropna=False).agg(
                                       **agg_pn
                                       ).sort_values(['brand', 'part_type_clean', 'cnt_parts'], ascending=[True, True, False])

display(df_fcp_cnts_br_pt)

df_fcp_cnts_br_pt.to_clipboard()

# %% 
# Examine vehicle brand vs part type:

df_fcp_cnts_br_pt = df_fcp.groupby(['source_brand','part_type_clean'],
                                   as_index=False, dropna=False).agg(
                                       **agg_pn
                                       ).sort_values(['source_brand', 'part_type_clean', 'cnt_parts'], ascending=[True, True, False])

display(df_fcp_cnts_br_pt)

# %%
# Temp to figure out null part_types

part_types = df_fcp.loc[df_fcp['part_type'].isna(), 'brand'].drop_duplicates()

print(part_types.unique())
part_types.to_clipboard()

# %% [markdown]
# # Key Summaries
#
# * Count of parts by: available, in_stock, part_type 

# %%
# By OEM, type, in_stock

df_fcp_cnts_pt_stk_av = df_fcp.groupby(['source_brand', 'part_type_clean','in_stock','available'],
                             as_index=False, dropna=False).agg(**agg_pn)

display(df_fcp_cnts_pt_stk_av)

print(df_fcp_cnts_pt_stk_av['cnt_parts'].sum())

# %%
# By availability

shipping_times = {
    "In Stock": 0,
    "No ETA": None,
    "No Longer Available": None,
    "Ships in 1 business day": 1,
    "Ships in 2 business days": 2,
    "Ships in 2-3 weeks": 17,
    "Ships in 3 business days": 3,
    "Ships in 4 business days": 4,
    "Ships in 4-6 weeks": 35,
    "Ships in 5 business days": 5,
    "Ships in 7 business days": 7,
    "Ships within 1 business day": 0.5,
    "This part has been superseded": None,
    "This product is not available for sale": None
}

df_fcp_cnts_stk_av = df_fcp.groupby(['in_stock','available'],
                             as_index=False, dropna=False).agg(**agg_pn)

# TODO: Move this up tp the master data
df_fcp_cnts_stk_av['shipping_days'] = df_fcp_cnts_stk_av['in_stock'].map(shipping_times)

display(df_fcp_cnts_stk_av.sort_values(['shipping_days']))

#display(df_fcp_cnts_stk_av[['in_stock']])
# %%
