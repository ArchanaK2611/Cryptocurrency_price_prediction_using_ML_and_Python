import streamlit as st
import pandas as pd
from pycoingecko import CoinGeckoAPI
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import datetime as dt
import matplotlib.pyplot as plt

# Create an instance of the CoinGeckoAPI
cg = CoinGeckoAPI()

# Function to get historical data from CoinGecko API
def get_historical_data(coin_id, days):
    try:
        data = cg.get_coin_market_chart_by_id(id=coin_id, vs_currency='usd', days=days)
        prices = data['prices']
        df = pd.DataFrame(prices, columns=['timestamp', 'price'])
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['date_ordinal'] = df['date'].map(dt.datetime.toordinal)
        return df
    except Exception as e:
        st.error(f"An error occurred while fetching historical data: {e}")
        return pd.DataFrame()

# Function to predict future prices using automated linear regression
def predict_price_automated(df, future_days):
    try:
        X = df[['date_ordinal']]
        y = df['price']

        # Normalize the features
        X_mean = X.mean()
        X_std = X.std()
        X_normalized = (X - X_mean) / X_std

        model = LinearRegression().fit(X_normalized, y)

        future_date = dt.datetime.now() + pd.to_timedelta(future_days, unit='D')
        future_date_ordinal = future_date.toordinal()
        future_date_normalized = (future_date_ordinal - X_mean.values[0]) / X_std.values[0]

        # Diagnostics
        st.write("Automated Model Coefficients:", model.coef_)
        st.write("Automated Model Intercept:", model.intercept_)
        st.write("Data range:", df['date'].min(), "to", df['date'].max())
        st.write("Future date ordinal:", future_date_ordinal)
        st.write("Future date normalized:", future_date_normalized)

        future_price = model.predict([[future_date_normalized]])
        return future_price[0], mean_squared_error(y, model.predict(X_normalized))
    except Exception as e:
        st.error(f"An error occurred while predicting the price: {e}")
        return None, None

# Function to predict future prices using manual linear regression calculation
def predict_price_manual(df, future_days):
    try:
        X = df['date_ordinal'].values
        y = df['price'].values

        # Calculate the mean of X and y
        X_mean = X.mean()
        y_mean = y.mean()

        # Calculate the slope (beta_1)
        numerator = ((X - X_mean) * (y - y_mean)).sum()
        denominator = ((X - X_mean) ** 2).sum()
        beta_1 = numerator / denominator

        # Calculate the intercept (beta_0)
        beta_0 = y_mean - beta_1 * X_mean

        # Future date ordinal
        future_date = dt.datetime.now() + pd.to_timedelta(future_days, unit='D')
        future_date_ordinal = future_date.toordinal()

        # Predict the future price
        future_price = beta_0 + beta_1 * future_date_ordinal

        # Diagnostics
        st.write("Manual Model Slope (beta_1):", beta_1)
        st.write("Manual Model Intercept (beta_0):", beta_0)
        st.write("Data range:", df['date'].min(), "to", df['date'].max())
        st.write("Future date ordinal:", future_date_ordinal)

        return future_price, mean_squared_error(y, beta_0 + beta_1 * X)
    except Exception as e:
        st.error(f"An error occurred while predicting the price: {e}")
        return None, None

# Function to predict future prices using Random Forest Regressor
def predict_price_rf(df, future_days):
    try:
        X = df[['date_ordinal']]
        y = df['price']

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        future_date = dt.datetime.now() + pd.to_timedelta(future_days, unit='D')
        future_date_ordinal = future_date.toordinal()

        future_price = model.predict([[future_date_ordinal]])

        mse = mean_squared_error(y, model.predict(X))
        return future_price[0], mse
    except Exception as e:
        st.error(f"An error occurred while predicting the price with Random Forest Regressor: {e}")
        return None, None

# Streamlit app
st.title("Welcome To Crypto World")

# Display the quote using Streamlit's markdown functionality
st.markdown(
    """
    <marquee style='color: red; font-size: 24px;'>
        "The future of money is cryptocurrency - Bill Gates"
    </marquee>
    """, unsafe_allow_html=True
)

# Login section
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.header("Login Details")

    name = st.text_input("Name:")
    email = st.text_input("Email:")
    contact = st.text_input("Contact no:")

    if st.button("Login"):
        if name and email and contact:
            st.session_state.logged_in = True
            st.session_state.name = name
            st.session_state.email = email
            st.session_state.contact = contact
            st.success("Login successful")
        else:
            st.error("Please fill out all fields")

# Main app section
if st.session_state.logged_in:
    st.header(f"Welcome, {st.session_state.name}!")

    # Retrieve and display list of coins
    try:
        coins = cg.get_coins_list()
        coin_names = [coin['name'] for coin in coins]
        coin_dict = {coin['name']: coin['id'] for coin in coins}
    except Exception as e:
        st.error(f"An error occurred while retrieving coins: {e}")
        coins = []
        coin_names = []
        coin_dict = {}

    if coin_names:
        # Dropdown for coin selection
        coin_name = st.selectbox('Select Coin:', coin_names)
        coin_id = coin_dict[coin_name]

        # Input for number of days for historical data
        days = st.number_input('Enter Number of Days for Historical Data:', min_value=1, step=1)

        # Fetch Data Button
        if st.button('Fetch Data'):
            df = get_historical_data(coin_id, days)
            if not df.empty:
                st.session_state.df = df  # Store the dataframe in session state
                st.session_state.coin_name = coin_name  # Store the selected coin name in session state
                st.write(f'Historical data for {coin_name}:')

                # Plot the historical data
                st.line_chart(df[['date', 'price']].set_index('date'))
                
                # Additional diagnostic plot
                plt.figure(figsize=(10, 6))
                plt.scatter(df['date_ordinal'], df['price'])
                plt.xlabel('Date Ordinal')
                plt.ylabel('Price')
                plt.title('Historical Prices')
                st.pyplot(plt)

    # Check if historical data is already in session state
    if 'df' in st.session_state:
        df = st.session_state.df
        coin_name = st.session_state.coin_name

        # Input for number of days into the future for prediction
        future_days = st.number_input('Enter Number of Days into the Future:', min_value=1, step=1)

        # Predict Price Buttons
        if st.button('Predict Price (Automated)'):
            predicted_price_automated, mse_automated = predict_price_automated(df, future_days)
            if predicted_price_automated is not None:
                st.write(f'Predicted price of {coin_name} in {future_days} days (Automated) is {predicted_price_automated:.2f} USD')
                st.write(f'Automated Model Mean Squared Error: {mse_automated:.2f}')
                
        if st.button('Predict Price (Manual)'):
            predicted_price_manual, mse_manual = predict_price_manual(df, future_days)
            if predicted_price_manual is not None:
                st.write(f'Predicted price of {coin_name} in {future_days} days (Manual) is {predicted_price_manual:.2f} USD')
                st.write(f'Manual Model Mean Squared Error: {mse_manual:.2f}')

        if st.button('Predict Price (Random Forest)'):
            predicted_price_rf, mse_rf = predict_price_rf(df, future_days)
            if predicted_price_rf is not None:
                st.write(f'Predicted price of {coin_name} in {future_days} days (Random Forest) is {predicted_price_rf:.2f} USD')
                st.write(f'Random Forest Regressor Mean Squared Error: {mse_rf:.2f}')

        # Compare models based on Mean Squared Error
        if st.button('Compare Models'):
            if 'df' in st.session_state:
                df = st.session_state.df
                future_days = st.number_input('Enter Number of Days into the Future for Comparison:', min_value=1, step=1)

                predicted_price_linear, mse_linear = predict_price_automated(df, future_days)
                predicted_price_rf, mse_rf = predict_price_rf(df, future_days)

                if predicted_price_linear is not None and predicted_price_rf is not None:
                    st.write(f'Predicted price of {coin_name} in {future_days} days using Linear Regression is {predicted_price_linear:.2f} USD')
                    st.write(f'Linear Regression Mean Squared Error: {mse_linear:.2f}')

                    st.write(f'Predicted price of {coin_name} in {future_days} days using Random Forest Regressor is {predicted_price_rf:.2f} USD')
                    st.write(f'Random Forest Regressor Mean Squared Error: {mse_rf:.2f}')

                    # Compare the accuracy of both models
                    if mse_linear < mse_rf:
                        st.write("Linear Regression is more accurate based on Mean Squared Error.")
                    elif mse_rf < mse_linear:
                        st.write("Random Forest Regressor is more accurate based on Mean Squared Error.")
                    else:
                        st.write("Both models have similar accuracy based on Mean Squared Error.")

