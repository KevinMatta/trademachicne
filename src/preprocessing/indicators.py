"""
Module for calculating and updating technical indicators.
"""
import logging
import pandas as pd
import numpy as np
import os
import sqlite3
import talib
from datetime import datetime, timedelta
from .data_cleaning import load_market_data_from_db

logger = logging.getLogger(__name__)

def calculate_indicators(df):
    """
    Calculate technical indicators for a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
    
    Returns:
        pd.DataFrame: DataFrame with indicators added
    """
    try:
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
        
        # Calculate indicators
        
        # Moving Averages
        df_copy['sma_5'] = talib.SMA(df_copy['close'], timeperiod=5)
        df_copy['sma_10'] = talib.SMA(df_copy['close'], timeperiod=10)
        df_copy['sma_20'] = talib.SMA(df_copy['close'], timeperiod=20)
        df_copy['sma_50'] = talib.SMA(df_copy['close'], timeperiod=50)
        df_copy['sma_100'] = talib.SMA(df_copy['close'], timeperiod=100)
        df_copy['sma_200'] = talib.SMA(df_copy['close'], timeperiod=200)
        
        df_copy['ema_5'] = talib.EMA(df_copy['close'], timeperiod=5)
        df_copy['ema_10'] = talib.EMA(df_copy['close'], timeperiod=10)
        df_copy['ema_20'] = talib.EMA(df_copy['close'], timeperiod=20)
        df_copy['ema_50'] = talib.EMA(df_copy['close'], timeperiod=50)
        df_copy['ema_100'] = talib.EMA(df_copy['close'], timeperiod=100)
        df_copy['ema_200'] = talib.EMA(df_copy['close'], timeperiod=200)
        
        # Momentum Indicators
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
        
        # Volatility Indicators
        df_copy['bbands_upper'], df_copy['bbands_middle'], df_copy['bbands_lower'] = talib.BBANDS(
            df_copy['close'], 
            timeperiod=20, 
            nbdevup=2, 
            nbdevdn=2, 
            matype=0
        )
        
        df_copy['atr_14'] = talib.ATR(df_copy['high'], df_copy['low'], df_copy['close'], timeperiod=14)
        
        # Trend Indicators
        df_copy['adx_14'] = talib.ADX(df_copy['high'], df_copy['low'], df_copy['close'], timeperiod=14)
        
        # Volume Indicators
        df_copy['obv'] = talib.OBV(df_copy['close'], df_copy['volume'])
        df_copy['ad'] = talib.AD(df_copy['high'], df_copy['low'], df_copy['close'], df_copy['volume'])
        df_copy['adosc'] = talib.ADOSC(df_copy['high'], df_copy['low'], df_copy['close'], df_copy['volume'], fastperiod=3, slowperiod=10)
        
        # Oscillators
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
        
        df_copy['willr_14'] = talib.WILLR(df_copy['high'], df_copy['low'], df_copy['close'], timeperiod=14)
        
        # CCI (Commodity Channel Index)
        df_copy['cci_14'] = talib.CCI(df_copy['high'], df_copy['low'], df_copy['close'], timeperiod=14)
        
        # ROC (Rate of Change)
        df_copy['roc_10'] = talib.ROC(df_copy['close'], timeperiod=10)
        
        # Fill NaN values
        df_copy = df_copy.fillna(0)
        
        return df_copy
    
    except Exception as e:
        logger.error(f"Error calculating indicators: {str(e)}")
        return df

def store_indicators(symbol, interval, df):
    """
    Store calculated indicators in the database.
    
    Args:
        symbol (str): Trading pair symbol
        interval (str): Kline interval
        df (pd.DataFrame): DataFrame with indicators
    """
    try:
        if df.empty:
            logger.warning("Empty DataFrame provided for storing")
            return
        
        # Get database connection
        db_path = os.path.join(os.path.dirname(__file__), '../../instance/finance.db')
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create table if it doesn't exist
        table_name = f"indicators_{symbol.lower()}_{interval}"
        
        # Get column definitions
        columns = df.columns.tolist()
        columns.remove('timestamp') if 'timestamp' in columns else None
        
        # Create column definitions for SQL
        column_defs = ["timestamp TEXT PRIMARY KEY"]
        for col in columns:
            column_defs.append(f"{col} REAL")
        
        cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {table_name} (
                {', '.join(column_defs)}
            )
        ''')
        
        # Insert data
        for _, row in df.iterrows():
            # Skip if timestamp is missing
            if 'timestamp' not in row or pd.isna(row['timestamp']):
                continue
            
            timestamp = row['timestamp'].isoformat() if isinstance(row['timestamp'], datetime) else row['timestamp']
            
            # Create placeholders and values
            placeholders = ', '.join(['?'] * (len(columns) + 1))
            columns_sql = 'timestamp, ' + ', '.join(columns)
            
            values = [timestamp]
            for col in columns:
                values.append(float(row[col]) if col in row and not pd.isna(row[col]) else 0)
            
            # Insert row
            cursor.execute(f'''
                INSERT OR REPLACE INTO {table_name} ({columns_sql})
                VALUES ({placeholders})
            ''', values)
        
        conn.commit()
        logger.info(f"Stored indicators for {symbol} {interval} in database")
    
    except Exception as e:
        logger.error(f"Error storing indicators: {str(e)}")
        if 'conn' in locals():
            conn.rollback()
    finally:
        if 'conn' in locals():
            conn.close()

def update_indicators():
    """Update technical indicators for all symbols and intervals in the database."""
    try:
        # Get database connection to find available symbols and intervals
        db_path = os.path.join(os.path.dirname(__file__), '../../instance/finance.db')
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get list of klines tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'klines_%'")
        tables = cursor.fetchall()
        
        conn.close()
        
        for (table_name,) in tables:
            # Extract symbol and interval from table name
            parts = table_name.split('_')
            if len(parts) < 3:
                continue
            
            symbol = parts[1].upper()
            interval = parts[2]
            
            logger.info(f"Updating indicators for {symbol} {interval}")
            
            # Load market data
            df = load_market_data_from_db(symbol, interval)
            
            if df.empty:
                logger.warning(f"No market data available for {symbol} {interval}")
                continue
            
            # Calculate indicators
            df_indicators = calculate_indicators(df)
            
            # Store indicators
            store_indicators(symbol, interval, df_indicators)
            
            logger.info(f"Updated indicators for {symbol} {interval}")
    
    except Exception as e:
        logger.error(f"Error updating indicators: {str(e)}")

def get_indicators(symbol, interval, start_date=None, end_date=None):
    """
    Get indicators from the database.
    
    Args:
        symbol (str): Trading pair symbol
        interval (str): Kline interval
        start_date (str, optional): Start date in ISO format
        end_date (str, optional): End date in ISO format
    
    Returns:
        pd.DataFrame: DataFrame with indicators
    """
    try:
        # Get database connection
        db_path = os.path.join(os.path.dirname(__file__), '../../instance/finance.db')
        conn = sqlite3.connect(db_path)
        
        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.now().isoformat()
        
        if start_date is None:
            # Default to 30 days ago
            start_date = (datetime.now() - timedelta(days=30)).isoformat()
        
        # Query data
        table_name = f"indicators_{symbol.lower()}_{interval}"
        query = f"""
            SELECT * FROM {table_name}
            WHERE timestamp BETWEEN ? AND ?
            ORDER BY timestamp
        """
        
        df = pd.read_sql_query(query, conn, params=(start_date, end_date))
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        logger.info(f"Loaded {len(df)} rows of indicators for {symbol} {interval}")
        return df
    
    except Exception as e:
        logger.error(f"Error loading indicators from DB: {str(e)}")
        return pd.DataFrame()
    finally:
        if 'conn' in locals():
            conn.close()