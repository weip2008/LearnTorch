import pandas as pd
import matplotlib.pyplot as plt

def plot_prices(df):
    """
    Plots the Close price and Normalized price on the same chart with dual y-axes.

    Parameters:
    df (pandas.DataFrame): DataFrame containing 'Close' and 'Normalized_Price' columns.
    """
    # Plotting
    fig, ax1 = plt.subplots()

    # Plot Close prices
    ax1.plot(df.index, df['Close'], color='blue', label='Close Price', linestyle='-', marker='o')
    ax1.set_xlabel('Datetime')
    ax1.set_ylabel('Close Price', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_ylim(df['Close'].min(), df['Close'].max())

    # Create a twin y-axis to plot Normalized Price
    ax2 = ax1.twinx()
    ax2.plot(df.index, df['Normalized_Price'], color='red', label='Normalized Price', linestyle='-', marker='x')
    ax2.set_ylabel('Normalized Price', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(df['Normalized_Price'].min(), df['Normalized_Price'].max())

    # Add a legend to differentiate the plots
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    fig.tight_layout()
    plt.title('Close Price and Normalized Price')
    plt.show()

if __name__ == "__main__":
    data = {
        'Datetime': [
            '2020-01-03 05:26:00', '2020-01-03 05:27:00', '2020-01-03 05:28:00', '2020-01-03 05:29:00', '2020-01-03 05:30:00',
            '2020-01-03 05:31:00', '2020-01-03 05:32:00', '2020-01-03 05:33:00', '2020-01-03 05:34:00', '2020-01-03 05:35:00',
            '2020-01-03 05:36:00', '2020-01-03 05:37:00', '2020-01-03 05:38:00', '2020-01-03 05:39:00', '2020-01-03 05:40:00',
            '2020-01-03 05:41:00', '2020-01-03 05:42:00', '2020-01-03 05:43:00', '2020-01-03 05:44:00', '2020-01-03 05:45:00',
            '2020-01-03 05:46:00', '2020-01-03 05:47:00', '2020-01-03 05:48:00', '2020-01-03 05:49:00', '2020-01-03 05:50:00',
            '2020-01-03 05:51:00', '2020-01-03 05:52:00', '2020-01-03 05:53:00', '2020-01-03 05:54:00', '2020-01-03 05:55:00',
            '2020-01-03 05:56:00', '2020-01-03 05:57:00'
        ],
        'Close': [
            3222.352, 3221.846, 3222.100, 3222.337, 3222.400, 3222.349, 3222.046, 3220.843, 3220.343, 3220.846,
            3220.546, 3220.355, 3220.037, 3219.049, 3218.852, 3218.355, 3218.052, 3218.849, 3218.840, 3218.037,
            3217.849, 3217.855, 3217.100, 3215.049, 3214.052, 3213.334, 3214.334, 3213.040, 3213.346, 3213.046,
            3208.543, 3206.352
        ],
        'Normalized_Price': [
            0.997009, 0.965479, 0.981306, 0.996074, 1.000000, 0.996822, 0.977941, 0.902979, 0.871822, 0.903166,
            0.884472, 0.872570, 0.852754, 0.791189, 0.778913, 0.747944, 0.729063, 0.778726, 0.778166, 0.728128,
            0.716413, 0.716787, 0.669741, 0.541937, 0.479811, 0.435070, 0.497383, 0.416750, 0.435818, 0.417124,
            0.136528, 0.000000
        ]
    }
    df = pd.DataFrame(data)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.set_index('Datetime', inplace=True)

    plot_prices(df)