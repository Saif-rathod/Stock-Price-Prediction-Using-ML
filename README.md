
# Stock Price Prediction using ML

This repository contains a Python script that analyzes and predicts Tesla's stock prices using linear regression. The analysis includes visualization of stock price trends and the performance evaluation of the linear regression model.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/tesla-stock-analysis.git
   cd tesla-stock-analysis

2. Install the required dependencies:
```bash
pip install pandas numpy matplotlib plotly scikit-learn chart-studio

```
3. Make sure you have the `tesla.csv` file in the repository directory. The CSV file should contain the stock price data with columns such as `Date`, `Open`, `High`, `Low`, `Close`, `Adj Close` and `Volume`.


## Usage

Run the script to perform the analysis and prediction:

```python
python stock.ipynb

```

The script performs the following tasks:

- Loads the Tesla stock price data.
- Displays the date range and total number of days in the dataset.
- Plots a boxplot for the stock price columns.
- Visualizes the stock price trends using Plotly.
- Splits the data into training and testing sets.
- Scales the features.
- Trains a linear regression model.
- Plots the actual vs. predicted stock prices for the training set.
- Prints the R-squared and Mean Squared Error (MSE) for both training and testing sets.


## Project Structure

```bash
├── tesla.csv         # Tesla stock price data file
├── stock_pred.py     # Main script for analysis and prediction
├── README.md         # Project README file
```
## Results

The script outputs the following information:

- Date range and total number of days in the dataset.
- Boxplot for the stock price columns.
- Plotly visualization of stock price trends.
- Plotly visualization of actual vs. predicted stock prices for the training set.
- R-squared and MSE for both training and testing sets.

Example output: 
``` yaml
Dataframe contains stock prices between 2020-01-01 00:00:00 2023-01-01 00:00:00
Total days = 1096 days

Metric         Train               Test               
r2_score       0.95                0.93               
MSE            10.5                12.3               
```

## Contributing

Contributions are always welcome!
Please open an issue or submit a pull request for any improvements or bug fixes.



