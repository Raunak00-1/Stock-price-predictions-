



                            #   STOCK PRICE PREDICTION   #

# Step 1. Import libraries

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.ticker as ticker
# Step 2. Load the CSV data

df= pd.read_csv("AppleStock.csv")
print(df.head(8))  

# Step 3  identify the all data

df = df[["Open","High","Low","Close",]]

# Step 4. Create the target column (next day's closing price)

df["Target"] = df["Close"].shift(-1)

# Step 5. Remove the last row with NaN in 'Target'

df = df.dropna()

# Keep a copy of full data to access 'High' and 'Low' later

full_data = df.copy()

# Step 6. Define features and labels

x = df[["Close"]]        # Today's price
y = df["Target"]         # Tomorrow's price

# Step 7. Train-test split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Step 8. Train the model

model = LinearRegression()
model.fit(x_train, y_train)

# Step 9. Predict the values

predictions = model.predict(x_test)

# Step 10. Plot actual vs predicted

y_test_sorted = y_test.sort_index()
predictions_sorted = pd.Series(predictions, index=y_test.index).sort_index()
plt.figure(figsize=(12, 6))
plt.plot(y_test_sorted.values, label="Actual Close Price", color="red")
plt.plot(predictions_sorted.values, label="Predicted Close Price", color="green")
    
plt.title("Stock Price Prediction using Linear Regression")
plt.xlabel("Days")
plt.ylabel("Stock Price (USD)")
plt.tick_params(axis='y', labelsize=8)   # Shrinks font size on Y-axis

# Fix overlap: rotate and limit ticks

plt.yticks(rotation=0)  # Keep vertical for clarity
plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(nbins=10))

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 11: Plot High and Low prices (from test set)We'll use sorted x_test to match order

high_sorted = full_data.loc[x_test.index, "High"].sort_index()
low_sorted = full_data.loc[x_test.index, "Low"].sort_index()


plt.figure(figsize=(12, 6))
plt.plot(low_sorted.values, label="Low Price", color="blue", linestyle="--")
plt.plot(high_sorted.values, label="High Price", color="green", linestyle="-")
plt.title("High and Low Prices in Test Stock market(Apple)")
plt.xlabel("Days")
plt.ylabel("Stock Price(USD)")
#  Fix overlap: rotate and limit ticks
plt.yticks(rotation=0)
plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(nbins=10))

plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()

