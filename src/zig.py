import yfinance as yf
import matplotlib.pyplot as plt

def zigzag(stock_data, percentage_reversal):
    high_points = []
    low_points = []

    i = 0
    while i < len(stock_data) - 1:
        start = i
        end = i
        while end + 1 < len(stock_data) and ((stock_data['Close'][end + 1] - stock_data['Close'][start]) / stock_data['Close'][start]) * 100 <= -percentage_reversal:
            end += 1
        if end > start:
            high_points.append((stock_data.index[start], stock_data['Close'][start]))
            low_points.append((stock_data.index[end], stock_data['Close'][end]))
        i = end + 1

    return high_points, low_points

# Fetch stock data for Apple (AAPL) from Yahoo Finance
stock_data = yf.download('AAPL', start='2023-01-01', end='2024-01-01')

# Percentage reversal
percentage_reversal = 0.3

high_points, low_points = zigzag(stock_data, percentage_reversal)

plt.figure(figsize=(12, 6))
plt.plot(stock_data.index, stock_data['Close'], label='Close Price', color='blue')
plt.scatter([x[0] for x in high_points], [x[1] for x in high_points], color='green', marker='^', label='High Points')
plt.scatter([x[0] for x in low_points], [x[1] for x in low_points], color='red', marker='v', label='Low Points')
plt.title('AAPL Stock Price with Zigzag Diagram')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()
