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
import logging
from typing import Tuple, Dict, Any, Optional
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)

class DataFetcher:
    """Handles all data fetching operations with proper error handling."""
    
    @staticmethod
    def fetch_poloniex_data(symbol: str = "BTC_USDT") -> Optional[float]:
        """Fetch real-time price data from Poloniex with exponential backoff retry."""
        max_retries = 3
        base_delay = 1
        
        for attempt in range(max_retries):
            try:
                url = f"https://api.poloniex.com/markets/{symbol}/price"
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                return float(response.json()['price'])
            except requests.exceptions.RequestException as e:
                delay = base_delay * (2 ** attempt)
                logging.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {delay} seconds...")
                time.sleep(delay)
        
        logging.error("Failed to fetch data from Poloniex after all retries")
        return None

    @staticmethod
    def fetch_historical_data(
        symbol: str = "BTC-USD",
        period: str = '1y',
        interval: str = '1d'
    ) -> pd.DataFrame:
        """Fetch and process historical data with technical indicators."""
        try:
            data = yf.download(symbol, period=period, interval=interval)
            
            if data.empty:
                raise ValueError("No data retrieved from Yahoo Finance")
            
            # Technical indicators
            data['returns'] = data['Adj Close'].pct_change()
            data['log_returns'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1))
            data['volatility'] = data['returns'].rolling(window=21).std()
            data['ma_50'] = data['Adj Close'].rolling(50).mean()
            data['ma_200'] = data['Adj Close'].rolling(200).mean()
            
            # Additional indicators
            data['rsi'] = DataFetcher._calculate_rsi(data['Adj Close'])
            data['atr'] = DataFetcher._calculate_atr(data)
            
            return data.dropna()
        except Exception as e:
            logging.error(f"Error fetching historical data: {str(e)}")
            raise

    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high = data['High']
        low = data['Low']
        close = data['Adj Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()

class MarketAnalyzer:
    """Handles market analysis and signal generation."""
    
    @staticmethod
    def identify_market_regimes(data: pd.DataFrame, n_regimes: int = 3) -> pd.DataFrame:
        """Identify market regimes using HMM."""
        features = np.column_stack([data['returns'], data['volatility']])
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = GaussianHMM(
                n_components=n_regimes,
                covariance_type="diag",
                n_iter=100,
                random_state=42
            )
            # First fit the model
            model.fit(features)
            # Then predict the regimes
            data['regime'] = model.predict(features)
        
        return data

    @staticmethod
    def cluster_assets(data: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
        """Perform asset clustering based on risk-return characteristics."""
        features = np.column_stack([
            data['returns'] * 252,  # Annualized returns
            data['volatility'] * np.sqrt(252),  # Annualized volatility
            data['returns'] / (data['volatility'] + 1e-6)  # Sharpe ratio approximation
        ])
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        data['cluster'] = kmeans.fit_predict(features)
        return data

class TradingEnvironment(gym.Env):
    """Enhanced trading environment with risk management."""
    
    def __init__(
        self,
        data: pd.DataFrame,
        initial_balance: float = 100000,
        max_position_size: float = 10000,
        stop_loss: float = 0.02,
        take_profit: float = 0.05
    ):
        super().__init__()
        
        self.data = data
        self.initial_balance = initial_balance
        self.max_position_size = max_position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(3)  # Buy, Hold, Sell
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(8,),  # Extended feature set
            dtype=np.float32
        )
        
        self.reset()

    def reset(self) -> np.ndarray:
        """Reset the trading environment."""
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.current_step = 0
        self.trades = []
        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        """Get current market observation with proper scalar conversion."""
        try:
            obs = np.array([
                float(self.data['Adj Close'].iloc[self.current_step]),
                float(self.data['returns'].iloc[self.current_step]),
                float(self.data['volatility'].iloc[self.current_step]),
                float(self.data['ma_50'].iloc[self.current_step]),
                float(self.data['ma_200'].iloc[self.current_step]),
                float(self.data['rsi'].iloc[self.current_step]),
                float(self.data['atr'].iloc[self.current_step]),
                float(self.position / self.max_position_size)  # Position size indicator
            ], dtype=np.float32)
            
            # Ensure there are no NaN values
            if np.any(np.isnan(obs)):
                logging.warning("NaN values detected in observation. Replacing with zeros.")
                obs = np.nan_to_num(obs, 0.0)
            
            return obs
            
        except Exception as e:
            logging.error(f"Error creating observation: {str(e)}")
            # Return zeros as fallback
            return np.zeros(8, dtype=np.float32)   

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one trading step."""
        current_price = float(self.data['Adj Close'].iloc[self.current_step])
        prev_portfolio_value = self.balance + (self.position * current_price)
        
        # Execute trading action
        if action == 0 and self.position == 0:  # Buy
            position_size = min(self.max_position_size, self.balance * 0.95)  # 95% max allocation
            self.position = position_size / current_price
            self.balance -= position_size
            self.entry_price = current_price
            self.trades.append({
                'type': 'buy',
                'price': current_price,
                'size': self.position,
                'timestamp': self.data.index[self.current_step]
            })
            
        elif action == 2 and self.position > 0:  # Sell
            profit = self.position * (current_price - self.entry_price)
            self.balance += (self.position * current_price)
            self.trades.append({
                'type': 'sell',
                'price': current_price,
                'size': self.position,
                'profit': profit,
                'timestamp': self.data.index[self.current_step]
            })
            self.position = 0
        
        # Check stop-loss and take-profit
        elif self.position > 0:
            price_change = (current_price / self.entry_price - 1)
            
            if price_change <= -self.stop_loss:
                self.balance += self.position * current_price
                self.trades.append({
                    'type': 'stop_loss',
                    'price': current_price,
                    'size': self.position,
                    'profit': self.position * (current_price - self.entry_price),
                    'timestamp': self.data.index[self.current_step]
                })
                self.position = 0
                
            elif price_change >= self.take_profit:
                self.balance += self.position * current_price
                self.trades.append({
                    'type': 'take_profit',
                    'price': current_price,
                    'size': self.position,
                    'profit': self.position * (current_price - self.entry_price),
                    'timestamp': self.data.index[self.current_step]
                })
                self.position = 0
        
        # Calculate reward
        current_portfolio_value = self.balance + (self.position * current_price)
        reward = (current_portfolio_value - prev_portfolio_value) / prev_portfolio_value
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1 or self.balance <= 0
        
        info = {
            'portfolio_value': current_portfolio_value,
            'balance': self.balance,
            'position': self.position,
            'trades': self.trades[-1] if self.trades else None
        }
        
        return self._get_observation(), reward, done, info

    def __init__(self, symbol: str = "BTC-USD"):
        self.symbol = symbol
        self.data_fetcher = DataFetcher()
        self.market_analyzer = MarketAnalyzer()
        self.model = None
        
    def validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean the dataset."""
        required_columns = ['Adj Close', 'returns', 'volatility', 'ma_50', 'ma_200', 'rsi', 'atr']
        
        # Check for missing columns
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Handle NaN values
        for col in required_columns:
            null_count = data[col].isna().sum()
            if null_count > 0:
                logging.warning(f"Found {null_count} NaN values in {col}. Filling with forward fill method.")
                data[col] = data[col].ffill().bfill()
        
        # Verify data is not empty
        if len(data.index) == 0:
            raise ValueError("Dataset is empty after cleaning")
        
        return data
        
    def train(self, training_epochs: int = 10000) -> None:
        """Train the trading bot."""
        logging.info("Starting training process...")
        
        try:
            # Fetch and prepare data
            data = self.data_fetcher.fetch_historical_data(self.symbol)
            
            # Check if we have any data
            if len(data.index) == 0:
                raise ValueError("No data available for training")
            
            # Validate and clean data
            data = self.validate_data(data)
            
            # Add market analysis features
            data = self.market_analyzer.identify_market_regimes(data)
            data = self.market_analyzer.cluster_assets(data)
            
            # Final validation
            final_data_length = len(data.index)
            if final_data_length < 100:  # Arbitrary minimum length
                raise ValueError(f"Insufficient data for training: only {final_data_length} samples available")
            
            # Create and train the model
            env = TradingEnvironment(data)
            self.model = PPO(
                "MlpPolicy",
                env,
                verbose=1,
                learning_rate=1e-4,
                batch_size=64,
                n_steps=min(2048, final_data_length - 1),
                ent_coef=0.01,
                n_epochs=10
            )
            
            self.model.learn(total_timesteps=min(training_epochs, final_data_length * 10))
            logging.info("Training completed successfully")
            
        except Exception as e:
            logging.error(f"Error during training: {str(e)}")
            raise

    def execute_trades(self) -> None:
        """Execute live trading."""
        if self.model is None:
            raise ValueError("Model must be trained before executing trades")
            
        while True:
            try:
                current_price = self.data_fetcher.fetch_poloniex_data(
                    self.symbol.replace("-", "_")
                )
                
                if current_price is None:
                    logging.warning("Unable to fetch current price, skipping iteration")
                    time.sleep(60)
                    continue
                
                # Get current market state
                data = self.data_fetcher.fetch_historical_data(
                    self.symbol,
                    period='7d',
                    interval='5m'
                )
                
                # Check if we have data
                if len(data.index) == 0:
                    logging.warning("No valid data available, skipping iteration")
                    time.sleep(60)
                    continue
                
                # Validate and clean data
                data = self.validate_data(data)
                
                # Prepare environment and get action
                env = TradingEnvironment(data)
                obs = env.reset()
                action, _ = self.model.predict(obs, deterministic=True)
                
                # Execute action and log results
                _, reward, done, info = env.step(action)
                
                logging.info(
                    f"Action taken: {['Buy', 'Hold', 'Sell'][action]}, "
                    f"Portfolio Value: ${info['portfolio_value']:.2f}, "
                    f"Current Price: ${current_price:.2f}"
                )
                
                time.sleep(300)  # 5-minute intervals
                
            except Exception as e:
                logging.error(f"Error during trade execution: {str(e)}")
                time.sleep(60)

def main():
    """Main execution function."""
    try:
        bot = TradingBot("BTC-USD")
        bot.train(training_epochs=5000)  # Reduced epochs for testing
        bot.execute_trades()
    except Exception as e:
        logging.error(f"Fatal error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()