import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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