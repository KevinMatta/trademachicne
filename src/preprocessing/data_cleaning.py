"""
Module for cleaning and preprocessing financial data.
"""
import logging
import pandas as pd
import numpy as np
import os
import sqlite3
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def clean_market_data(df):
    """
    Clean market data by handling missing values and outliers.
    
    Args:
        df (pd.DataFrame): DataFrame with market data
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    try:
        logger.info("Cleaning market data")
        
        if df.empty:
            logger.warning("Empty DataFrame provided for cleaning")
            return df
        
        # Make a copy to avoid modifying the original
        df_clean = df.copy()
        
        # Convert timestamp to datetime if it's not already
        if 'timestamp' in df_clean.columns and not pd.api.types.is_datetime64_any_dtype(df_clean['timestamp']):
            df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'])
        
        # Sort by timestamp
        if 'timestamp' in df_clean.columns:
            df_clean = df_clean.sort_values('timestamp')
        
        # Handle missing values
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        # Fill missing values in numeric columns with forward fill, then backward fill
        for col in numeric_cols:
            df_clean[col] = df_clean[col].fillna(method='ffill').fillna(method='bfill')
        
        # Check for remaining NaNs
        if df_clean.isna().any().any():
            logger.warning("NaN values remain after filling")
            
            # For any remaining NaNs, fill with column mean
            for col in numeric_cols:
                if df_clean[col].isna().any():
                    mean_val = df_clean[col].mean()
                    df_clean[col] = df_clean[col].fillna(mean_val)
        
        # Handle outliers using Interquartile Range (IQR) method
        for col in numeric_cols:
            q1 = df_clean[col].quantile(0.25)
            q3 = df_clean[col].quantile(0.75)
            iqr = q3 - q1
            
            lower_bound = q1 - 3 * iqr
            upper_bound = q3 + 3 * iqr
            
            # Identify outliers
            outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)]
            if not outliers.empty:
                logger.info(f"Found {len(outliers)} outliers in column {col}")
                
                # Replace outliers with nearest valid value
                df_clean.loc[df_clean[col] < lower_bound, col] = lower_bound
                df_clean.loc[df_clean[col] > upper_bound, col] = upper_bound
        
        logger.info("Market data cleaning completed")
        return df_clean
    
    except Exception as e:
        logger.error(f"Error cleaning market data: {str(e)}")
        return df

def resample_data(df, timeframe):
    """
    Resample time series data to a different timeframe.
    
    Args:
        df (pd.DataFrame): DataFrame with time series data
        timeframe (str): Target timeframe (e.g., '1H', '4H', '1D')
    
    Returns:
        pd.DataFrame: Resampled DataFrame
    """
    try:
        logger.info(f"Resampling data to {timeframe}")
        
        if df.empty:
            logger.warning("Empty DataFrame provided for resampling")
            return df
        
        # Make a copy to avoid modifying the original
        df_copy = df.copy()
        
        # Ensure timestamp column is datetime and set as index
        if 'timestamp' in df_copy.columns:
            if not pd.api.types.is_datetime64_any_dtype(df_copy['timestamp']):
                df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
            
            df_copy = df_copy.set_index('timestamp')
        
        # Dictionary for aggregation functions
        agg_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        # Filter the aggregation dictionary to include only columns that exist
        agg_dict = {k: v for k, v in agg_dict.items() if k in df_copy.columns}
        
        if not agg_dict:
            logger.warning("No columns available for resampling")
            return df
        
        # Resample the data
        df_resampled = df_copy.resample(timeframe).agg(agg_dict)
        
        # Reset index to restore timestamp as a column
        df_resampled = df_resampled.reset_index()
        
        logger.info(f"Resampled data from {len(df_copy)} to {len(df_resampled)} rows")
        return df_resampled
    
    except Exception as e:
        logger.error(f"Error resampling data: {str(e)}")
        return df

def normalize_data(df, columns=None, method='z-score'):
    """
    Normalize numeric columns in the DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame to normalize
        columns (list, optional): List of columns to normalize. If None, normalize all numeric columns.
        method (str): Normalization method ('z-score', 'min-max', 'robust')
    
    Returns:
        pd.DataFrame: Normalized DataFrame
    """
    try:
        logger.info(f"Normalizing data using {method} method")
        
        if df.empty:
            logger.warning("Empty DataFrame provided for normalization")
            return df
        
        # Make a copy to avoid modifying the original
        df_norm = df.copy()
        
        # If no columns specified, use all numeric columns
        if columns is None:
            columns = df_norm.select_dtypes(include=[np.number]).columns.tolist()
            # Exclude certain columns that shouldn't be normalized
            exclude_cols = ['timestamp', 'id', 'volume']
            columns = [col for col in columns if col not in exclude_cols]
        
        # Apply normalization based on the specified method
        if method == 'z-score':
            for col in columns:
                mean = df_norm[col].mean()
                std = df_norm[col].std()
                if std != 0:  # Avoid division by zero
                    df_norm[col] = (df_norm[col] - mean) / std
                else:
                    logger.warning(f"Column {col} has zero standard deviation, skipping normalization")
        
        elif method == 'min-max':
            for col in columns:
                min_val = df_norm[col].min()
                max_val = df_norm[col].max()
                if max_val > min_val:  # Avoid division by zero
                    df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
                else:
                    logger.warning(f"Column {col} has identical min and max values, skipping normalization")
        
        elif method == 'robust':
            for col in columns:
                median = df_norm[col].median()
                q1 = df_norm[col].quantile(0.25)
                q3 = df_norm[col].quantile(0.75)
                iqr = q3 - q1
                if iqr != 0:  # Avoid division by zero
                    df_norm[col] = (df_norm[col] - median) / iqr
                else:
                    logger.warning(f"Column {col} has zero IQR, skipping normalization")
        
        else:
            logger.warning(f"Unknown normalization method: {method}")
            return df
        
        logger.info(f"Normalized {len(columns)} columns")
        return df_norm
    
    except Exception as e:
        logger.error(f"Error normalizing data: {str(e)}")
        return df

def load_market_data_from_db(symbol, interval, start_date=None, end_date=None):
    """
    Load market data from the database.
    
    Args:
        symbol (str): Trading pair symbol (e.g., 'BTCUSDT')
        interval (str): Kline interval (e.g., '1m', '5m', '1h', '1d')
        start_date (str, optional): Start date in ISO format
        end_date (str, optional): End date in ISO format
    
    Returns:
        pd.DataFrame: Market data
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
        table_name = f"klines_{symbol.lower()}_{interval}"
        query = f"""
            SELECT * FROM {table_name}
            WHERE timestamp BETWEEN ? AND ?
            ORDER BY timestamp
        """
        
        df = pd.read_sql_query(query, conn, params=(start_date, end_date))
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        logger.info(f"Loaded {len(df)} rows of market data for {symbol} {interval}")
        return df
    
    except Exception as e:
        logger.error(f"Error loading market data from DB: {str(e)}")
        return pd.DataFrame()
    finally:
        if 'conn' in locals():
            conn.close()