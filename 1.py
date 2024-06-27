import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Fetching data
@st.cache
def load_data(ticker):
    data = yf.download(ticker, period='5y')
    data.reset_index(inplace=True)
    data['Date'] = pd.to_datetime(data['Date'])  # Ensure 'Date' is in datetime format
    data.set_index('Date', inplace=True)  # Set 'Date' as the index
    return data

# Train model and predict target price and stop loss
def predict_prices(data):
    data['Prediction'] = data['Close'].shift(-30)
    X = np.array(data[['Close']])
    X = X[:-30]
    y = np.array(data['Prediction'])
    y = y[:-30]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    X_forecast = np.array(data[['Close']])[-30:]
    forecast = model.predict(X_forecast)
    
    next_target = forecast[-1]
    stop_loss = next_target * 0.95  # Example stop loss at 95% of target price
    
    return next_target, stop_loss

# Visualize stock data
def plot_data(data, ticker):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data.index,
                                 open=data['Open'],
                                 high=data['High'],
                                 low=data['Low'],
                                 close=data['Close'],
                                 name='Candlestick'))
    fig.update_layout(title=f'{ticker} Stock Price',
                      xaxis_title='Date',
                      yaxis_title='Price',
                      xaxis_rangeslider_visible=False)
    return fig

# Streamlit web app
st.title('Stock Analysis Web App (Indian Stocks)')

# App instructions
st.markdown("""
## How to Use This App

1. **Enter Stock Ticker**: 
   - In the input box, enter the ticker symbol of the stock you want to analyze (e.g., `RELIANCE.NS` for Reliance Industries, `TCS.NS` for TCS).
   - Click on the "Analyze" button to fetch and analyze the stock data.

2. **Stock Data**:
   - The app will display the latest stock data for the selected ticker.
   - This includes the last few rows .

3. **Next Target and Stop Loss**:
   - Using machine learning, the app predicts the next target price and a suggested stop loss for the stock.
   - These values are calculated based on historical data and displayed in this section.

4. **Stock Price Chart**:
   - A candlestick chart of the stock's price over time is displayed.
   - This chart helps visualize the open, high, low, and close prices.

5. **Closing Price Over Time**:
   - A line chart showing the closing prices over time.
   - Helps in understanding the trend and movement of the stock.

6. **Moving Averages**:
   - Displays the 50-day and 200-day moving averages along with the closing price.
   - Moving averages help in identifying the direction of the trend and potential reversal points.

## Developed by [Vikash Goyal](https://www.linkedin.com/in/vikash-goyal-20692924b/)
""")

ticker = st.text_input('Enter Stock Ticker (e.g., RELIANCE.NS, TCS.NS)', 'RELIANCE.NS')
data = load_data(ticker)

if st.button('Analyze'):
    st.subheader(f'{ticker} Stock Data')
    st.write(data.tail())

    next_target, stop_loss = predict_prices(data)
    
    st.subheader('Next Target and Stop Loss')
    st.write(f'Next Target Price: ₹{next_target:.2f}')
    st.write(f'Stop Loss: ₹{stop_loss:.2f}')
    
    st.subheader('Stock Price Chart')
    fig = plot_data(data, ticker)
    st.plotly_chart(fig)

    st.subheader('Closing Price Over Time')
    st.line_chart(data['Close'])

    st.subheader('Moving Averages')
    data['MA50'] = data['Close'].rolling(50).mean()
    data['MA200'] = data['Close'].rolling(200).mean()
    st.line_chart(data[['Close', 'MA50', 'MA200']])

# Developer Information
st.markdown("---")
st.markdown("### Developed by [Vikash Goyal](https://www.linkedin.com/in/vikash-goyal-20692924b/)")
st.image("https://avatars.githubusercontent.com/u/115889341?v=4", width=150)

# Run the app with:
# streamlit run your_script.py
