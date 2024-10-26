import requests
import yfinance as yf
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.cluster import KMeans
from stable_baselines3 import PPO
import gym
from gym import spaces
import xgboost as xgb
from datetime import datetime, timedelta
import time

# Fetch Poloniex real-time data
def fetch_poloniex_data(symbol="BTC_USDT"):
    url = f"https://api.poloniex.com/markets/{symbol}/price"
    response = requests.get(url)
    if response.status_code == 200:
        return float(response.json()['price'])
    else:
        print("Error fetching data from Poloniex.")
        return None

# Fetch historical data and compute additional features
def fetch_historical_data(symbol="BTC-USD", period='1y', interval='1d'):
    data = yf.download(symbol, period=period, interval=interval)
    data['returns'] = data['Adj Close'].pct_change()
    data['volatility'] = data['returns'].rolling(window=21).std()
    data['ma_50'] = data['Adj Close'].rolling(50).mean()
    data['ma_200'] = data['Adj Close'].rolling(200).mean()
    return data.dropna()

# Generate Buy/Sell Signal
# Generate Buy/Sell Signal
def generate_signals(data):
    signals = pd.DataFrame(index=data.index)
    signals['price'] = data['Adj Close']
    
    # Buy signal: When 50-day MA crosses above 200-day MA
    signals['buy_signal'] = (data['ma_50'] > data['ma_200']) & (data['ma_50'].shift(1) <= data['ma_200'].shift(1))
    
    # Sell signal: When 50-day MA crosses below 200-day MA
    signals['sell_signal'] = (data['ma_50'] < data['ma_200']) & (data['ma_50'].shift(1) >= data['ma_200'].shift(1))
    
    return signals


# 1. Identify Market Regimes
def market_regimes(data):
    features = data[['returns', 'volatility']]
    
    # Initialize the model with 'covariance_type' as 'diag'
    model = GaussianHMM(n_components=3, covariance_type="diag", n_iter=100)  
    
    model.fit(features)
    data['regime'] = model.predict(features)
    return data

# 2. Asset Selection with K-Means Clustering
def kmeans_clustering(data):
    # Calculate features for each data point (row) in the DataFrame
    data['returns_annualized'] = data['returns'] * 252
    data['volatility_annualized'] = data['returns'].rolling(window=21).std() * np.sqrt(252)
    data['sharpe_ratio'] = data['returns_annualized'] / data['volatility_annualized']

    # Drop rows with NaN values resulting from rolling calculations
    data.dropna(inplace=True)

    # Use the calculated features for clustering
    features = data[['returns_annualized', 'volatility_annualized', 'sharpe_ratio']]
    kmeans = KMeans(n_clusters=5)  
    data['cluster'] = kmeans.fit_predict(features)  # Fit and predict on the features
    return data

# 3. XGBoost for Price Prediction
def xgboost_prediction(data):
    data.dropna(inplace=True)
    X = data[['ma_50', 'ma_200', 'returns', 'volatility']]
    y = data['returns'].shift(-1)
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X[:-1], y.dropna())
    data['predicted_returns'] = model.predict(X)
    return data

# Trading Environment
class TradingEnv(gym.Env):
    def __init__(self, data, initial_balance=100, max_position_size=10, stop_loss=0.02, take_profit=0.05):
        super(TradingEnv, self).__init__()
        self.data = data
        self.balance = initial_balance
        self.position = 0
        self.entry_price = 0
        self.current_step = 0
        self.max_position_size = max_position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.action_space = spaces.Discrete(3)  # Buy, Hold, Sell
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
    
    def reset(self):
        self.balance = 100
        self.position = 0
        self.entry_price = 0
        self.current_step = 0
        return self._get_observation()

    def _get_observation(self):
    # Ensure all elements are single values before creating the array
      return np.array([
        float(self.data['Adj Close'].iloc[self.current_step]),  # Ensure this is a float
        float(self.data['returns'].iloc[self.current_step]),     # Ensure this is a float
        float(self.data['volatility'].iloc[self.current_step]),   # Ensure this is a float
        float(self.data['ma_50'].iloc[self.current_step]),        # Ensure this is a float
        float(self.data['ma_200'].iloc[self.current_step]),       # Ensure this is a float
        float(self.data['predicted_returns'].iloc[self.current_step])  # Ensure this is a float
    ], dtype=np.float32).reshape(1, -1)  # Reshape to (1, 6)

    def step(self, action):
      price = self.data['Adj Close'].iloc[self.current_step]
      reward = 0
      done = False

      # Ensure price is a scalar numeric value
      price = float(price)  # Convert to float if necessary

      if action == 0 and self.position == 0:  # Buy action
          self.position = min(self.max_position_size, self.balance)
          self.balance -= self.position  # Deduct position cost from balance
          self.entry_price = price  # Set entry price
          reward = 0  # No reward for initial buy

      elif action == 2 and self.position > 0:  # Sell action
          sell_price = price
          profit = self.position * (sell_price / self.entry_price - 1)
          self.balance += self.position + profit  # Update balance with profits
          reward = profit  # Reward is the profit made from the sale
          self.position = 0  # Reset position after sale

      elif self.position > 0:  # Check stop-loss and take-profit
          if self.entry_price != 0:  # Avoid ZeroDivisionError
              # Stop-loss condition
              if (price / self.entry_price - 1) <= -self.stop_loss:
                  self.balance += self.position * (1 - self.stop_loss)
                  reward = -self.position * self.stop_loss
                  self.position = 0
              
              # Take-profit condition
              elif (price / self.entry_price - 1) >= self.take_profit:
                  self.balance += self.position * (1 + self.take_profit)
                  reward = self.position * self.take_profit
                  self.position = 0

      # Move to the next time step
      self.current_step += 1

      # Check if the episode is done
      if self.current_step >= len(self.data) - 1 or self.balance <= 0:
          done = True

      # Return the observation, reward, done flag, and any additional info
      return self._get_observation(), reward, done, {}


    def render(self):
        print(f'Step: {self.current_step}, Balance: {self.balance}, Position: {self.position}')

# Train the Reinforcement Model
def train_reinforcement_model(data):
    env = TradingEnv(data)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=5000)
    return model

# Execute Trading Bot
def execute_trading_bot():
    poloniex_price = fetch_poloniex_data("BTC_USDT")
    print(f"Current BTC price from Poloniex: {poloniex_price}")

    symbol = 'BTC-USD'
    data = fetch_historical_data(symbol)
    data = market_regimes(data)
    data = kmeans_clustering(data)
    data = xgboost_prediction(data)
    signals = generate_signals(data)

    ppo_model = train_reinforcement_model(data)
    
    env = TradingEnv(data)
    obs = env.reset()
    for _ in range(len(data) - 1):
        action, _ = ppo_model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        # Display action and balance
        env.render()
        if done:
            break

execute_trading_bot()
