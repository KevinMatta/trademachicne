"""
Module for feature engineering from financial data.
"""
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import talib
from .data_cleaning import load_market_data_from_db

logger = logging.getLogger(__name__)

def add_time_features(df):
    """
    Add time-based features to the DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with timestamp column
    
    Returns:
        pd.DataFrame: DataFrame with time features added
    """
    try:
        logger.info("Adding time features")
        
        if df.empty:
            logger.warning("Empty DataFrame provided")
            return df
        
        # Make a copy to avoid modifying the original
        df_copy = df.copy()
        
        # Ensure timestamp column is datetime
        if 'timestamp' in df_copy.columns:
            if not pd.api.types.is_datetime64_any_dtype(df_copy['timestamp']):
                df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
        else:
            logger.warning("No timestamp column found, can't add time features")
            return df_copy
        
        # Extract time features
        df_copy['hour'] = df_copy['timestamp'].dt.hour
        df_copy['day_of_week'] = df_copy['timestamp'].dt.dayofweek
        df_copy['day_of_month'] = df_copy['timestamp'].dt.day
        df_copy['month'] = df_copy['timestamp'].dt.month
        df_copy['quarter'] = df_copy['timestamp'].dt.quarter
        
        # Create cyclical features for hour, day_of_week, day_of_month, month
        df_copy['hour_sin'] = np.sin(2 * np.pi * df_copy['hour'] / 24)
        df_copy['hour_cos'] = np.cos(2 * np.pi * df_copy['hour'] / 24)
        
        df_copy['day_of_week_sin'] = np.sin(2 * np.pi * df_copy['day_of_week'] / 7)
        df_copy['day_of_week_cos'] = np.cos(2 * np.pi * df_copy['day_of_week'] / 7)
        
        df_copy['day_of_month_sin'] = np.sin(2 * np.pi * df_copy['day_of_month'] / 31)
        df_copy['day_of_month_cos'] = np.cos(2 * np.pi * df_copy['day_of_month'] / 31)
        
        df_copy['month_sin'] = np.sin(2 * np.pi * df_copy['month'] / 12)
        df_copy['month_cos'] = np.cos(2 * np.pi * df_copy['month'] / 12)
        
        # Is weekend feature
        df_copy['is_weekend'] = df_copy['day_of_week'].isin([5, 6]).astype(int)
        
        logger.info("Time features added")
        return df_copy
    
    except Exception as e:
        logger.error(f"Error adding time features: {str(e)}")
        return df

def add_price_features(df):
    """
    Add price-based features to the DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
    
    Returns:
        pd.DataFrame: DataFrame with price features added
    """
    try:
        logger.info("Adding price features")
        
        if df.empty:
            logger.warning("Empty DataFrame provided")
            return df
        
        # Make a copy to avoid modifying the original
        df_copy = df.copy()
        
        # Check if required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df_copy.columns]
        
        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")
            return df_copy
        
        # Convert columns to numeric if needed
        for col in required_cols:
            if not pd.api.types.is_numeric_dtype(df_copy[col]):
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
        
        # Calculate returns
        df_copy['return'] = df_copy['close'].pct_change()
        df_copy['log_return'] = np.log(df_copy['close'] / df_copy['close'].shift(1))
        
        # Price differences
        df_copy['open_close_diff'] = df_copy['close'] - df_copy['open']
        df_copy['high_low_diff'] = df_copy['high'] - df_copy['low']
        df_copy['high_close_diff'] = df_copy['high'] - df_copy['close']
        df_copy['close_low_diff'] = df_copy['close'] - df_copy['low']
        
        # Volatility measures
        df_copy['volatility'] = df_copy['log_return'].rolling(window=20).std()
        df_copy['range_pct'] = (df_copy['high'] - df_copy['low']) / df_copy['open']
        
        # Volume features
        df_copy['volume_change'] = df_copy['volume'].pct_change()
        df_copy['volume_ma'] = df_copy['volume'].rolling(window=20).mean()
        df_copy['volume_std'] = df_copy['volume'].rolling(window=20).std()
        df_copy['volume_ma_ratio'] = df_copy['volume'] / df_copy['volume_ma']
        
        # Candle features
        df_copy['body_size'] = abs(df_copy['close'] - df_copy['open'])
        df_copy['upper_shadow'] = df_copy['high'] - df_copy[['open', 'close']].max(axis=1)
        df_copy['lower_shadow'] = df_copy[['open', 'close']].min(axis=1) - df_copy['low']
        
        # Candle direction
        df_copy['direction'] = np.where(df_copy['close'] >= df_copy['open'], 1, -1)
        
        # Fill NaN values
        df_copy = df_copy.fillna(0)
        
        logger.info("Price features added")
        return df_copy
    
    except Exception as e:
        logger.error(f"Error adding price features: {str(e)}")
        return df

def add_technical_indicators(df):
    """
    Add technical indicators to the DataFrame using TA-Lib.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
    
    Returns:
        pd.DataFrame: DataFrame with technical indicators added
    """
    try:
        logger.info("Adding technical indicators")
        
        if df.empty:
            logger.warning("Empty DataFrame provided")
            return df
        
        # Make a copy to avoid modifying the original
        df_copy = df.copy()
        
        # Check if required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df_copy.columns]
        
        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")
            return df_copy
        
        # Convert columns to numeric if needed
        for col in required_cols:
            if not pd.api.types.is_numeric_dtype(df_copy[col]):
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
        
        # Add moving averages
        df_copy['sma_5'] = talib.SMA(df_copy['close'], timeperiod=5)
        df_copy['sma_20'] = talib.SMA(df_copy['close'], timeperiod=20)
        df_copy['sma_50'] = talib.SMA(df_copy['close'], timeperiod=50)
        df_copy['sma_200'] = talib.SMA(df_copy['close'], timeperiod=200)
        
        df_copy['ema_5'] = talib.EMA(df_copy['close'], timeperiod=5)
        df_copy['ema_20'] = talib.EMA(df_copy['close'], timeperiod=20)
        df_copy['ema_50'] = talib.EMA(df_copy['close'], timeperiod=50)
        df_copy['ema_200'] = talib.EMA(df_copy['close'], timeperiod=200)
        
        # Add crossover signals
        df_copy['sma_5_20_cross'] = np.where(
            (df_copy['sma_5'].shift(1) < df_copy['sma_20'].shift(1)) & 
            (df_copy['sma_5'] > df_copy['sma_20']), 
            1, np.where(
                (df_copy['sma_5'].shift(1) > df_copy['sma_20'].shift(1)) & 
                (df_copy['sma_5'] < df_copy['sma_20']), 
                -1, 0
            )
        )
        
        # Add momentum indicators
        df_copy['rsi_14'] = talib.RSI(df_copy['close'], timeperiod=14)
        
        macd, macd_signal, macd_hist = talib.MACD(
            df_copy['close'], 
            fastperiod=12, 
            slowperiod=26, 
            signalperiod=9
        )
        df_copy['macd'] = macd
        df_copy['macd_signal'] = macd_signal
        df_copy['macd_hist'] = macd_hist
        
        # Add volatility indicators
        df_copy['bbands_upper'], df_copy['bbands_middle'], df_copy['bbands_lower'] = talib.BBANDS(
            df_copy['close'], 
            timeperiod=20, 
            nbdevup=2, 
            nbdevdn=2, 
            matype=0
        )
        
        df_copy['atr_14'] = talib.ATR(df_copy['high'], df_copy['low'], df_copy['close'], timeperiod=14)
        
        # Add trend indicators
        df_copy['adx_14'] = talib.ADX(df_copy['high'], df_copy['low'], df_copy['close'], timeperiod=14)
        
        # Add volume indicators
        df_copy['obv'] = talib.OBV(df_copy['close'], df_copy['volume'])
        df_copy['ad'] = talib.AD(df_copy['high'], df_copy['low'], df_copy['close'], df_copy['volume'])
        df_copy['adosc'] = talib.ADOSC(df_copy['high'], df_copy['low'], df_copy['close'], df_copy['volume'], fastperiod=3, slowperiod=10)
        
        # Stochastic oscillator
        df_copy['slowk'], df_copy['slowd'] = talib.STOCH(
            df_copy['high'], 
            df_copy['low'], 
            df_copy['close'], 
            fastk_period=5, 
            slowk_period=3, 
            slowk_matype=0, 
            slowd_period=3, 
            slowd_matype=0
        )
        
        # Fill NaN values
        df_copy = df_copy.fillna(0)
        
        logger.info("Technical indicators added")
        return df_copy
    
    except Exception as e:
        logger.error(f"Error adding technical indicators: {str(e)}")
        return df

def create_prediction_features(symbol, interval, lookback=30):
    """
    Create a feature set for prediction models.
    
    Args:
        symbol (str): Trading pair symbol (e.g., 'BTCUSDT')
        interval (str): Kline interval (e.g., '1h', '4h', '1d')
        lookback (int): Number of historical periods to include
    
    Returns:
        tuple: (X, y) feature matrix and target vector
    """
    try:
        logger.info(f"Creating prediction features for {symbol} {interval} with lookback {lookback}")
        
        # Load market data
        df = load_market_data_from_db(symbol, interval)
        
        if df.empty:
            logger.warning("No market data available")
            return None, None
        
        # Add features
        df = add_price_features(df)
        df = add_technical_indicators(df)
        df = add_time_features(df)
        
        # Select features for prediction
        feature_cols = [
            'open', 'high', 'low', 'close', 'volume',
            'return', 'log_return', 'volatility', 'range_pct',
            'rsi_14', 'macd', 'macd_signal', 'macd_hist',
            'bbands_upper', 'bbands_middle', 'bbands_lower',
            'atr_14', 'adx_14', 'obv', 'slowk', 'slowd'
        ]
        
        # Make sure all selected features exist
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        if not feature_cols:
            logger.error("No valid feature columns found")
            return None, None
        
        # Create target: next period's return
        df['target'] = df['close'].shift(-1) / df['close'] - 1
        
        # Drop rows with NaN values
        df = df.dropna()
        
        if len(df) < lookback:
            logger.warning(f"Not enough data points ({len(df)}) for lookback {lookback}")
            return None, None
        
        # Create sequences for LSTM or similar models
        X_sequences = []
        y = []
        
        for i in range(len(df) - lookback):
            X_sequences.append(df[feature_cols].iloc[i:i+lookback].values)
            y.append(df['target'].iloc[i+lookback])
        
        X = np.array(X_sequences)
        y = np.array(y)
        
        logger.info(f"Created features with shape {X.shape} and targets with shape {y.shape}")
        return X, y
    
    except Exception as e:
        logger.error(f"Error creating prediction features: {str(e)}")
        return None, None