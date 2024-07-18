import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textwrap import wrap
import os
import pandas as pd
import numpy as np
from datetime import datetime
from pandas.tseries.offsets import DateOffset
import openpyxl
import geopandas as gpd
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.font_manager import FontProperties
from matplotlib.patheffects import withStroke
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Tuple
import colorsys
import seaborn as sns
from matplotlib.sankey import Sankey
import holoviews as hv
import hvplot.pandas
from holoviews import opts

def plot_alluvial_diagram_5(data_path, translation_path, dates):
    # Step 1: Read the main CSV data file into a pandas DataFrame
    data = pd.read_csv(data_path, parse_dates=['Date'], index_col='Date')

    # Step 2: Read the translation dictionary CSV file into another DataFrame and create a mapping dictionary
    translation_df = pd.read_csv(translation_path, header=None)
    translation_dict = pd.Series(translation_df[1].values, index=translation_df[0]).to_dict()

    # Step 3: Translate the column names in the main DataFrame using this dictionary
    data.rename(columns=translation_dict, inplace=True)

    # Step 4: Extract the top 5 columns for each date based on their values
    top_columns = {}
    for date in dates:
        top_columns[date] = data.loc[date].nlargest(5).index.tolist()

    # Step 5: Create a DataFrame that keeps track of the rankings of these top 5 columns over the given dates
    all_columns = sorted(list(set(sum(top_columns.values(), []))))  # Convert set to sorted list
    rankings = pd.DataFrame(index=all_columns, columns=dates)

    for date in dates:
        # Create a Series with ranks only for the top columns and NaN for others, then fill NaNs with a default rank
        rank_series = pd.Series(top_columns[date]).rank().reindex(all_columns).fillna(len(all_columns) + 1).astype(int)
        rankings[date] = rank_series

    # Step 6: Use a library like matplotlib and seaborn to create the alluvial diagram with consistent colors
    plt.figure(figsize=(14, 8))

    # Get a distinct color for each column
    colors = sns.color_palette("husl", len(rankings.index))

    # Plot each column's trajectory across time
    for col, color in zip(rankings.index, colors):
        plt.plot(dates, [rankings.loc[col, date] for date in dates], 'o-', label=col, color=color)

    plt.xticks(range(len(dates)), dates, rotation=45)
    plt.xlabel('Date')
    plt.ylabel('Top 5 Columns')
    plt.title('Alluvial Diagram of Top 5 Columns Over Time')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Columns")
    sns.despine(left=True, bottom=True)
    plt.show()


def create_bump_chart(csv_path, dates):
    # Read the CSV file
    df = pd.read_csv(csv_path, parse_dates=['Date'])

    # Filter the dataframe for the specified dates
    df_filtered = df[df['Date'].isin(dates)]

    # Melt the dataframe to long format
    df_melted = df_filtered.melt(id_vars=['Date'], var_name='Column', value_name='Value')

    # Remove rows with NaN values
    df_melted = df_melted.dropna()

    # Get the top 5 columns for each date
    top_5_columns = df_melted.groupby('Date').apply(lambda x: x.nlargest(5, 'Value')['Column'].tolist())

    # Create a set of all unique top 5 columns
    all_top_columns = set([col for cols in top_5_columns for col in cols])

    # Filter the melted dataframe to include only the top columns
    df_top = df_melted[df_melted['Column'].isin(all_top_columns)]

    # Calculate the rank for each column within each date
    df_top['Rank'] = df_top.groupby('Date')['Value'].rank(method='first', ascending=False)

    # Create a pivot table with all dates and columns
    df_pivot = df_top.pivot(index='Date', columns='Column', values='Rank')

    # Fill NaN values with a rank higher than the maximum
    max_rank = df_top['Rank'].max()
    df_pivot = df_pivot.fillna(max_rank + 1)

    # Get the last period's values for color mapping
    last_period_values = df_top[df_top['Date'] == df_top['Date'].max()].set_index('Column')['Value']

    # Create the custom color map
    colors = ["#FFFFD8", "#b9e67f", "#8ECAE6", "#219ebc", "#003049"]
    n_bins = 10000  # Discretizes the interpolation into bins
    custom_cmap = LinearSegmentedColormap.from_list('custom_palette', colors, N=n_bins)

    # Create the bump chart
    plt.figure(figsize=(14, 10))

    # Normalize the last period's values for color mapping
    norm = plt.Normalize(last_period_values.min(), last_period_values.max())

    # Melt the pivot table back to long format
    df_plot = df_pivot.reset_index().melt(id_vars=['Date'], var_name='Column', value_name='Rank')

    for column in df_pivot.columns:
        data = df_plot[df_plot['Column'] == column]
        color = custom_cmap(norm(last_period_values.get(column, last_period_values.min())))
        line, = plt.plot(data['Date'], data['Rank'], linewidth=40, label=column, color=color)

        # Add values on the plot
        for x, y, value in zip(data['Date'], data['Rank'], df_top[df_top['Column'] == column]['Value']):
            if y <= max_rank:  # Only add label if the point is in the top 5
                plt.text(x, y, f'{round(value/1000000, 1)}', ha='right', va='bottom', fontsize=15, color='black',
                         path_effects=[withStroke(linewidth=2, foreground='white')])

    # Customize the plot
    plt.gca().invert_yaxis()  # Invert y-axis to have rank 1 at the top
    plt.title('Bump Chart of Top 5 Columns Over Time', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Rank', fontsize=12)

    # Modify legend to allow text wrapping
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, ['\n'.join(wrap(l, 40)) for l in labels],
               title='КВЕД', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Set y-axis limits
    plt.ylim(max_rank + 1.5, 0.5)

    # Remove x-axis labels and ticks
    plt.xticks(dates, dates, rotation=45, ha='right')

    # Adjust layout to prevent cropping of wrapped legend text
    plt.tight_layout()
    plt.subplots_adjust(right=0.7)  # Adjust this value as needed
    plt.grid(visible=False)
    plt.axis('off')
    plt.grid(False)
    # Show the plot
    plt.show()
