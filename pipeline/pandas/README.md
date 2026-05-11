# Pandas

DataFrame creation, manipulation, and analysis tasks using the Pandas library. Tasks use Coinbase Bitcoin price data (`coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv`).

## Tasks

| File | Function | Description |
|------|----------|-------------|
| `0-from_numpy.py` | `from_numpy(array)` | Creates a DataFrame from a NumPy array with alphabetically-labeled columns |
| `1-from_dictionary.py` | — | Creates a DataFrame from a hard-coded dictionary with specific index labels |
| `2-from_file.py` | `from_file(filename, delimiter)` | Loads a DataFrame from a delimited file |
| `3-rename.py` | — | Renames the `Timestamp` column to `Datetime` and converts values to datetime objects |
| `4-array.py` | — | Extracts the last 10 rows of the `High` and `Close` columns as a NumPy array |
| `5-slice.py` | — | Slices every 60th row for columns `High`, `Low`, `Close`, `Volume_(BTC)` |
| `6-flip_switch.py` | — | Sorts the DataFrame in reverse chronological order and transposes it |
| `7-high.py` | — | Sorts rows by the `High` column in descending order |
| `8-prune.py` | — | Removes rows where the `Close` price is NaN |
| `9-fill.py` | — | Forward-fills missing values and removes the `Weighted_Price` column |
| `10-index.py` | — | Sets the `Timestamp` column as the DataFrame index |
| `11-concat.py` | — | Concatenates two Coinbase and Bitstamp DataFrames up to a specific timestamp |
| `12-hierarchy.py` | — | Creates a hierarchical index from both exchanges, sorted by timestamp |
| `13-analyze.py` | `analyze(df)` | Returns descriptive statistics for all columns except `Timestamp` |
| `14-visualize.py` | — | Visualizes the Coinbase DataFrame as a line plot using the datetime index |

## Requirements

- Python 3.x
- pandas
- NumPy
- Matplotlib
