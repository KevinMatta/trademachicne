"""
Module for forecasting models using machine learning.
"""
import logging
import os
import yaml
import joblib
import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from ..preprocessing.data_cleaning import load_market_data_from_db
from ..preprocessing.feature_engineering import add_technical_indicators, add_price_features, add_time_features

logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from config.yaml file."""
    config_path = os.path.join(os.path.dirname(__file__), '../config.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    else:
        logger.warning("Config file not found. Using default configuration.")
        return {
            'analysis': {
                'forecasting': {
                    'model': 'lstm',
                    'lookback': 30,
                    'forecast_horizon': 5,
                    'features': ['close', 'volume', 'rsi', 'macd']
                }
            }
        }

def prepare_data(symbol, interval, lookback=30, horizon=5):
    """
    Prepare data for forecasting models.
    
    Args:
        symbol (str): Trading pair symbol
        interval (str): Kline interval
        lookback (int): Number of historical periods to include
        horizon (int): Number of periods to forecast
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler)
    """
    try:
        logger.info(f"Preparing data for {symbol} {interval} forecasting model")
        
        # Load market data
        df = load_market_data_from_db(symbol, interval)
        
        if df.empty:
            logger.warning("No market data available")
            return None, None, None, None, None
        
        # Add features
        df = add_price_features(df)
        df = add_technical_indicators(df)
        df = add_time_features(df)
        
        # Select features based on config
        config = load_config()
        feature_list = config.get('analysis', {}).get('forecasting', {}).get('features', 
                                                                         ['close', 'volume', 'rsi_14', 'macd'])
        
        # Make sure all selected features exist
        feature_cols = [col for col in feature_list if col in df.columns]
        
        if not feature_cols:
            logger.error("No valid feature columns found")
            return None, None, None, None, None
        
        # Create target: future closing prices
        for i in range(1, horizon + 1):
            df[f'close_shift_{i}'] = df['close'].shift(-i)
        
        # Drop rows with NaN values
        df = df.dropna()
        
        if len(df) < lookback + horizon:
            logger.warning(f"Not enough data points ({len(df)}) for lookback {lookback} and horizon {horizon}")
            return None, None, None, None, None
        
        # Normalize data
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df[feature_cols]), columns=feature_cols)
        
        # Add target columns to scaled dataframe
        for i in range(1, horizon + 1):
            df_scaled[f'close_shift_{i}'] = df[f'close_shift_{i}']
        
        # Split into train and test sets (80/20)
        train_size = int(len(df_scaled) * 0.8)
        train_data = df_scaled[:train_size]
        test_data = df_scaled[train_size:]
        
        # Create sequences
        X_train, y_train = create_sequences(train_data, feature_cols, lookback, horizon)
        X_test, y_test = create_sequences(test_data, feature_cols, lookback, horizon)
        
        logger.info(f"Prepared data with shapes: X_train {X_train.shape}, y_train {y_train.shape}, "
                   f"X_test {X_test.shape}, y_test {y_test.shape}")
        
        return X_train, X_test, y_train, y_test, scaler
    
    except Exception as e:
        logger.error(f"Error preparing forecasting data: {str(e)}")
        return None, None, None, None, None

def create_sequences(df, feature_cols, lookback, horizon):
    """
    Create sequences for time series forecasting.
    
    Args:
        df (pd.DataFrame): DataFrame with features
        feature_cols (list): List of feature column names
        lookback (int): Number of historical periods to include
        horizon (int): Number of periods to forecast
    
    Returns:
        tuple: (X, y) features and targets
    """
    X = []
    y = []
    
    for i in range(len(df) - lookback - horizon + 1):
        # Extract sequence of features
        X.append(df[feature_cols].iloc[i:i+lookback].values)
        
        # Extract targets (future values)
        targets = []
        for j in range(1, horizon + 1):
            targets.append(df[f'close_shift_{j}'].iloc[i+lookback-1])
        
        y.append(targets)
    
    return np.array(X), np.array(y)

def train_lstm_model(X_train, y_train, X_test, y_test):
    """
    Train an LSTM model for forecasting.
    
    Args:
        X_train (numpy.ndarray): Training features
        y_train (numpy.ndarray): Training targets
        X_test (numpy.ndarray): Test features
        y_test (numpy.ndarray): Test targets
    
    Returns:
        tensorflow.keras.models.Model: Trained LSTM model
    """
    try:
        logger.info("Training LSTM model")
        
        # Get model parameters
        input_shape = (X_train.shape[1], X_train.shape[2])
        output_dim = y_train.shape[1]
        
        # Build LSTM model
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(output_dim)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        
        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Evaluate model
        test_loss = model.evaluate(X_test, y_test, verbose=0)
        logger.info(f"LSTM model test loss: {test_loss}")
        
        return model
    
    except Exception as e:
        logger.error(f"Error training LSTM model: {str(e)}")
        return None

def train_random_forest_model(X_train, y_train, X_test, y_test):
    """
    Train a Random Forest model for forecasting.
    
    Args:
        X_train (numpy.ndarray): Training features
        y_train (numpy.ndarray): Training targets
        X_test (numpy.ndarray): Test features
        y_test (numpy.ndarray): Test targets
    
    Returns:
        sklearn.ensemble.RandomForestRegressor: Trained Random Forest model
    """
    try:
        logger.info("Training Random Forest model")
        
        # Reshape data for Random Forest
        X_train_rf = X_train.reshape(X_train.shape[0], -1)
        X_test_rf = X_test.reshape(X_test.shape[0], -1)
        
        # Initialize and train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_rf, y_train)
        
        # Evaluate model
        test_score = model.score(X_test_rf, y_test)
        logger.info(f"Random Forest model R² score: {test_score}")
        
        return model
    
    except Exception as e:
        logger.error(f"Error training Random Forest model: {str(e)}")
        return None

def train_linear_regression_model(X_train, y_train, X_test, y_test):
    """
    Train a Linear Regression model for forecasting.
    
    Args:
        X_train (numpy.ndarray): Training features
        y_train (numpy.ndarray): Training targets
        X_test (numpy.ndarray): Test features
        y_test (numpy.ndarray): Test targets
    
    Returns:
        sklearn.linear_model.LinearRegression: Trained Linear Regression model
    """
    try:
        logger.info("Training Linear Regression model")
        
        # Reshape data for Linear Regression
        X_train_lr = X_train.reshape(X_train.shape[0], -1)
        X_test_lr = X_test.reshape(X_test.shape[0], -1)
        
        # Initialize and train model
        model = LinearRegression()
        model.fit(X_train_lr, y_train)
        
        # Evaluate model
        test_score = model.score(X_test_lr, y_test)
        logger.info(f"Linear Regression model R² score: {test_score}")
        
        return model
    
    except Exception as e:
        logger.error(f"Error training Linear Regression model: {str(e)}")
        return None

def save_model(model, model_type, symbol, interval):
    """
    Save a trained model to disk.
    
    Args:
        model: Trained model
        model_type (str): Type of model ('lstm', 'rf', 'lr')
        symbol (str): Trading pair symbol
        interval (str): Kline interval
    """
    try:
        # Create models directory if it doesn't exist
        models_dir = os.path.join(os.path.dirname(__file__), '../../models')
        os.makedirs(models_dir, exist_ok=True)
        
        # Create filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{symbol.lower()}_{interval}_{model_type}_{timestamp}"
        
        # Save model based on type
        if model_type == 'lstm':
            model_path = os.path.join(models_dir, f"{filename}.keras")
            model.save(model_path)
        else:
            model_path = os.path.join(models_dir, f"{filename}.joblib")
            joblib.dump(model, model_path)
        
        logger.info(f"Saved {model_type} model to {model_path}")
        
        # Save model info to database
        db_path = os.path.join(os.path.dirname(__file__), '../../instance/finance.db')
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                interval TEXT,
                model_type TEXT,
                path TEXT,
                created_at TEXT,
                is_active INTEGER DEFAULT 0
            )
        ''')
        
        # Insert model info
        cursor.execute('''
            INSERT INTO models (symbol, interval, model_type, path, created_at, is_active)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            symbol.lower(),
            interval,
            model_type,
            model_path,
            datetime.now().isoformat(),
            1  # Set as active
        ))
        
        # Deactivate previous models of the same type
        cursor.execute('''
            UPDATE models
            SET is_active = 0
            WHERE symbol = ? AND interval = ? AND model_type = ? AND id != ?
        ''', (
            symbol.lower(),
            interval,
            model_type,
            cursor.lastrowid
        ))
        
        conn.commit()
        conn.close()
    
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")

def load_saved_model(symbol, interval, model_type):
    """
    Load a saved model from disk.
    
    Args:
        symbol (str): Trading pair symbol
        interval (str): Kline interval
        model_type (str): Type of model ('lstm', 'rf', 'lr')
    
    Returns:
        Model: Loaded model
    """
    try:
        # Get model path from database
        db_path = os.path.join(os.path.dirname(__file__), '../../instance/finance.db')
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT path FROM models
            WHERE symbol = ? AND interval = ? AND model_type = ? AND is_active = 1
            ORDER BY created_at DESC
            LIMIT 1
        ''', (symbol.lower(), interval, model_type))
        
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            logger.warning(f"No saved {model_type} model found for {symbol} {interval}")
            return None
        
        model_path = result[0]
        
        # Load model based on type
        if model_type == 'lstm':
            model = load_model(model_path)
        else:
            model = joblib.load(model_path)
        
        logger.info(f"Loaded {model_type} model from {model_path}")
        return model
    
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None

def predict_future(symbol, interval, periods=5):
    """
    Make predictions for future periods.
    
    Args:
        symbol (str): Trading pair symbol
        interval (str): Kline interval
        periods (int): Number of periods to predict
    
    Returns:
        dict: Predictions from different models
    """
    try:
        logger.info(f"Making predictions for {symbol} {interval} ({periods} periods)")
        
        config = load_config()
        lookback = config.get('analysis', {}).get('forecasting', {}).get('lookback', 30)
        
        # Load market data
        df = load_market_data_from_db(symbol, interval)
        
        if df.empty:
            logger.warning("No market data available")
            return {}
        
        # Add features
        df = add_price_features(df)
        df = add_technical_indicators(df)
        df = add_time_features(df)
        
        # Select features based on config
        feature_list = config.get('analysis', {}).get('forecasting', {}).get('features', 
                                                                         ['close', 'volume', 'rsi_14', 'macd'])
        
        # Make sure all selected features exist
        feature_cols = [col for col in feature_list if col in df.columns]
        
        if not feature_cols:
            logger.error("No valid feature columns found")
            return {}
        
        # Normalize data
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df[feature_cols]), columns=feature_cols)
        
        # Get the most recent data for prediction
        recent_data = df_scaled.iloc[-lookback:][feature_cols].values
        recent_data = recent_data.reshape(1, lookback, len(feature_cols))
        
        # Get last close price for denormalization
        last_close = df['close'].iloc[-1]
        
        # Load models and make predictions
        predictions = {}
        
        # Try LSTM model
        lstm_model = load_saved_model(symbol, interval, 'lstm')
        if lstm_model:
            lstm_preds = lstm_model.predict(recent_data)[0]
            
            # Convert to prices
            lstm_prices = []
            for i, pred in enumerate(lstm_preds):
                # Calculate price from normalized prediction
                pred_price = last_close * (1 + pred)
                lstm_prices.append(pred_price)
            
            predictions['lstm'] = lstm_prices
        
        # Try Random Forest model
        rf_model = load_saved_model(symbol, interval, 'rf')
        if rf_model:
            rf_preds = rf_model.predict(recent_data.reshape(1, -1))[0]
            
            # Convert to prices
            rf_prices = []
            for i, pred in enumerate(rf_preds):
                # Calculate price from normalized prediction
                pred_price = last_close * (1 + pred)
                rf_prices.append(pred_price)
            
            predictions['rf'] = rf_prices
        
        # Try Linear Regression model
        lr_model = load_saved_model(symbol, interval, 'lr')
        if lr_model:
            lr_preds = lr_model.predict(recent_data.reshape(1, -1))[0]
            
            # Convert to prices
            lr_prices = []
            for i, pred in enumerate(lr_preds):
                # Calculate price from normalized prediction
                pred_price = last_close * (1 + pred)
                lr_prices.append(pred_price)
            
            predictions['lr'] = lr_prices
        
        # Generate timestamps for the predictions
        if df['timestamp'].iloc[-1]:
            last_timestamp = df['timestamp'].iloc[-1]
            timestamps = []
            
            # Map interval to timedelta
            interval_map = {
                '1m': timedelta(minutes=1),
                '5m': timedelta(minutes=5),
                '15m': timedelta(minutes=15),
                '30m': timedelta(minutes=30),
                '1h': timedelta(hours=1),
                '4h': timedelta(hours=4),
                '1d': timedelta(days=1),
                '1w': timedelta(weeks=1)
            }
            
            delta = interval_map.get(interval, timedelta(days=1))
            
            for i in range(1, periods + 1):
                timestamps.append((last_timestamp + delta * i).isoformat())
            
            predictions['timestamps'] = timestamps
        
        # Add current price for reference
        predictions['current_price'] = last_close
        predictions['current_timestamp'] = df['timestamp'].iloc[-1].isoformat() if df['timestamp'].iloc[-1] else None
        
        # Calculate ensemble prediction (average of all models)
        ensemble_prices = []
        
        for i in range(periods):
            model_preds = []
            for model_type in ['lstm', 'rf', 'lr']:
                if model_type in predictions and i < len(predictions[model_type]):
                    model_preds.append(predictions[model_type][i])
            
            if model_preds:
                ensemble_prices.append(sum(model_preds) / len(model_preds))
        
        if ensemble_prices:
            predictions['ensemble'] = ensemble_prices
        
        logger.info(f"Generated predictions for {symbol} {interval}")
        return predictions
    
    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        return {}

def update_forecasts():
    """Update forecasting models and predictions for all symbols and intervals."""
    try:
        # Get config
        config = load_config()
        
        # Get symbols and intervals from config
        symbols = config.get('data_collection', {}).get('symbols', ['BTCUSDT'])
        intervals = config.get('data_collection', {}).get('intervals', ['1h', '4h', '1d'])
        
        for symbol in symbols:
            for interval in intervals:
                logger.info(f"Updating forecasting models for {symbol} {interval}")
                
                # Prepare data
                X_train, X_test, y_train, y_test, scaler = prepare_data(symbol, interval)
                
                if X_train is None or y_train is None:
                    logger.warning(f"Could not prepare data for {symbol} {interval}")
                    continue
                
                # Train models
                lstm_model = train_lstm_model(X_train, y_train, X_test, y_test)
                if lstm_model:
                    save_model(lstm_model, 'lstm', symbol, interval)
                
                rf_model = train_random_forest_model(X_train, y_train, X_test, y_test)
                if rf_model:
                    save_model(rf_model, 'rf', symbol, interval)
                
                lr_model = train_linear_regression_model(X_train, y_train, X_test, y_test)
                if lr_model:
                    save_model(lr_model, 'lr', symbol, interval)
                
                # Generate and store predictions
                predictions = predict_future(symbol, interval)
                
                if predictions:
                    # Store predictions in database
                    db_path = os.path.join(os.path.dirname(__file__), '../../instance/finance.db')
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    
                    # Create table if it doesn't exist
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS predictions (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            symbol TEXT,
                            interval TEXT,
                            predictions TEXT,
                            created_at TEXT
                        )
                    ''')
                    
                    # Insert predictions
                    cursor.execute('''
                        INSERT INTO predictions (symbol, interval, predictions, created_at)
                        VALUES (?, ?, ?, ?)
                    ''', (
                        symbol.lower(),
                        interval,
                        json.dumps(predictions),
                        datetime.now().isoformat()
                    ))
                    
                    conn.commit()
                    conn.close()
                    
                    logger.info(f"Stored predictions for {symbol} {interval}")
    
    except Exception as e:
        logger.error(f"Error updating forecasts: {str(e)}")

def get_latest_predictions(symbol, interval):
    """
    Get the latest predictions for a symbol and interval.
    
    Args:
        symbol (str): Trading pair symbol
        interval (str): Kline interval
    
    Returns:
        dict: Latest predictions
    """
    try:
        # Get predictions from database
        db_path = os.path.join(os.path.dirname(__file__), '../../instance/finance.db')
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT predictions FROM predictions
            WHERE symbol = ? AND interval = ?
            ORDER BY created_at DESC
            LIMIT 1
        ''', (symbol.lower(), interval))
        
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            logger.warning(f"No predictions found for {symbol} {interval}")
            return {}
        
        predictions = json.loads(result[0])
        return predictions
    
    except Exception as e:
        logger.error(f"Error getting latest predictions: {str(e)}")
        return {}