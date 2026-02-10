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

try:
    get_ipython()
    from IPython.display import display
except NameError:
    display = print

#warnings.filterwarnings('ignore')

# %%
# Parameters

path = Path('..') / 'data'
fn = 'fcp-euro-parts.csv'

FLAG_SAVE_CHARTS = True
FLAG_SHOW_CHARTS = True

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

df_shipping_times = pd.DataFrame([
    ("In Stock",                              "In Stock",     0),
    ("No ETA",                                "No ETA",       None),
    ("No Longer Available",                   "Discontinued", None),
    ("Ships in 1 business day",               "1 day",        1),
    ("Ships in 2 business days",              "2 days",       2),
    ("Ships in 2-3 weeks",                    "2-3 weeks",    17),
    ("Ships in 3 business days",              "3 days",       3),
    ("Ships in 4 business days",              "4 days",       4),
    ("Ships in 4-6 weeks",                    "4-6 weeks",    35),
    ("Ships in 5 business days",              "5 days",       5),
    ("Ships in 7 business days",              "7 days",       7),
    ("Ships within 1 business day",           "< 1 day",      0.5),
    ("This part has been superseded",         "Superseded",   None),
    ("This product is not available for sale", "Not For Sale", None),
], columns=["in_stock", "Status", "Shipping Days"])

df_fcp_cnts_stk_av = df_fcp.groupby(['in_stock','available'],
                             as_index=False, dropna=False).agg(**agg_pn)

# TODO: Move this up tp the master data
df_fcp_cnts_stk_av = df_fcp_cnts_stk_av.merge(df_shipping_times[['in_stock', 'Shipping Days']], on='in_stock', how='left')

display(df_fcp_cnts_stk_av.sort_values(['Shipping Days']))

#display(df_fcp_cnts_stk_av[['in_stock']])
# %%
# Chart Setup & Helpers

from matplotlib.ticker import FuncFormatter

def fmt_thousands(x, _):
    return f'{x:,.0f}'

def fmt_pct(x, _):
    return f'{x:.0%}'

# Google Slide side-by-side dimensions
# ~10" slide width → each chart ~4.8" wide, ~4" tall, leaving room for title + notes
CHART_W, CHART_H = 4.8, 4.0
FS_TITLE = 13
FS_AXIS = 11
FS_TICK = 10
FS_BAR = 7.5
FS_LEGEND = 8

path_output = Path('..') / 'img'
path_output.mkdir(exist_ok=True)


def add_category_labels(ax, df_pivot, min_frac=0.06):
    """Label stacked bar segments with their category name when large enough."""
    max_h = df_pivot.sum(axis=1).max()
    for container, col in zip(ax.containers, df_pivot.columns):
        for bar in container:
            h = bar.get_height()
            if h / max_h > min_frac:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + h / 2, col,
                        ha='center', va='center',
                        fontsize=FS_BAR, color='white', fontweight='bold')


def stacked_bar(df_pivot, title, ylabel, colors,
                pct=False, bar_labels=True, legend_ncol=1, save_name=None):
    """Stacked bar chart sized for side-by-side placement on a Google Slide."""
    fig, ax = plt.subplots(figsize=(CHART_W, CHART_H))

    df_plot = df_pivot.div(df_pivot.sum(axis=1), axis=0) if pct else df_pivot
    df_plot.plot.bar(stacked=True, ax=ax, color=colors,
                     width=0.72, edgecolor='white', linewidth=0.5)

    ax.set_title(title, fontsize=FS_TITLE, fontweight='bold', pad=8)
    ax.set_xlabel('')
    ax.set_ylabel(ylabel, fontsize=FS_AXIS)
    ax.tick_params(axis='both', labelsize=FS_TICK)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    if pct:
        ax.yaxis.set_major_formatter(FuncFormatter(fmt_pct))
        ax.set_ylim(0, 1.005)
    else:
        ax.yaxis.set_major_formatter(FuncFormatter(fmt_thousands))

    if bar_labels:
        add_category_labels(ax, df_plot)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1],
              fontsize=FS_LEGEND, bbox_to_anchor=(1.02, 1),
              loc='upper left', frameon=False, ncol=legend_ncol)
    sns.despine(ax=ax)
    plt.tight_layout()

    if FLAG_SAVE_CHARTS and save_name:
        fig.savefig(path_output / save_name, dpi=150, bbox_inches='tight')
    if FLAG_SHOW_CHARTS:
        plt.show()
    plt.close(fig)


# %%
# ─── Charts 1 & 2: Part Count by OEM × Part Type ───

type_order = ['Genuine', 'OE', 'OEM', 'Not Specified']

df_type_pivot = (
    df_fcp.groupby(['source_brand', 'part_type_clean'], as_index=False)
    .agg(cnt=('sku', 'nunique'))
    .pivot(index='source_brand', columns='part_type_clean', values='cnt')
    .fillna(0)
)

# Reorder columns to match desired order
type_cols = [c for c in type_order if c in df_type_pivot.columns]
type_cols += [c for c in df_type_pivot.columns if c not in type_cols]
df_type_pivot = df_type_pivot[type_cols]

# Sort OEMs by total part count descending
df_type_pivot = df_type_pivot.loc[df_type_pivot.sum(axis=1).sort_values(ascending=False).index]

# Light blue palette (<=5 groups)
blues = sns.color_palette('Blues', n_colors=len(type_cols) + 2)[2:]

stacked_bar(df_type_pivot, 'Part Count by OEM & Type', 'Unique Parts',
            colors=blues, bar_labels=False,
            save_name='chart_oem_type_count.png')

stacked_bar(df_type_pivot, 'Part Mix by OEM & Type', '% of Parts',
            colors=blues, pct=True, bar_labels=False,
            save_name='chart_oem_type_pct.png')

# %%
# ─── Charts 3 & 4: Part Count by OEM × Availability Status ───

df_fcp_status = df_fcp.merge(df_shipping_times, on='in_stock', how='left')

df_status_pivot = (
    df_fcp_status.groupby(['source_brand', 'Status'], as_index=False)
    .agg(cnt=('sku', 'nunique'))
    .pivot(index='source_brand', columns='Status', values='cnt')
    .fillna(0)
)

# Order: Shipping Days ascending (None last), secondary sort by Status
status_order = (
    df_shipping_times
    .sort_values(['Shipping Days', 'Status'], ascending=[True, True], na_position='last')
    ['Status'].tolist()
)
status_cols = [s for s in status_order if s in df_status_pivot.columns]
status_cols += [s for s in df_status_pivot.columns if s not in status_cols]
df_status_pivot = df_status_pivot[status_cols]

# Sort OEMs by total part count descending
df_status_pivot = df_status_pivot.loc[df_status_pivot.sum(axis=1).sort_values(ascending=False).index]

# Tableau palette (>5 categories)
colors_status = sns.color_palette('tab20', n_colors=len(status_cols))

stacked_bar(df_status_pivot, 'Part Count by OEM & Status', 'Unique Parts',
            colors=colors_status, bar_labels=False,
            save_name='chart_oem_status_count.png')

stacked_bar(df_status_pivot, 'Part Mix by OEM & Status', '% of Parts',
            colors=colors_status, pct=True, bar_labels=False,
            save_name='chart_oem_status_pct.png')


# %%
