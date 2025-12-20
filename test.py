from services.data_reader import load_close_series

series = load_close_series(
    symbol="AAPL",
    interval="1d",
    limit=100
)

print(len(series))
print(series[:5])

