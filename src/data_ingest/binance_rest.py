"""
Module for interacting with Binance REST API to fetch market data.
"""
import logging
import time
import requests
import pandas as pd
import os
import yaml
import json
from datetime import datetime, timedelta
import sqlite3

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
            'apis': {
                'binance': {
                    'api_key': '',
                    'api_secret': '',
                    'testnet': True
                }
            }
        }

def get_api_url():
    """Get Binance API URL based on configuration."""
    config = load_config()
    testnet = config.get('apis', {}).get('binance', {}).get('testnet', True)
    return "https://testnet.binance.vision/api" if testnet else "https://api.binance.com/api"

def get_headers():
    """Get headers for Binance API requests."""
    config = load_config()
    api_key = config.get('apis', {}).get('binance', {}).get('api_key', '')
    
    headers = {}
    if api_key:
        headers['X-MBX-APIKEY'] = api_key
    
    return headers

def get_historical_data(symbol, interval, limit=500):
    """
    Fetch historical klines (candlestick) data from Binance.
    
    Args:
        symbol (str): Trading pair symbol (e.g., 'BTCUSDT')
        interval (str): Kline interval (e.g., '1m', '5m', '1h', '1d')
        limit (int): Number of klines to retrieve (max 1000)
    
    Returns:
        list: List of klines data
    """
    base_url = get_api_url()
    endpoint = f"{base_url}/v3/klines"
    
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': min(limit, 1000)  # Binance limit is 1000
    }
    
    try:
        response = requests.get(endpoint, params=params, headers=get_headers())
        response.raise_for_status()
        
        data = response.json()
        
        # Convert to DataFrame and format
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert numeric columns
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 
                        'quote_asset_volume', 'taker_buy_base_asset_volume', 
                        'taker_buy_quote_asset_volume']
        
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
        
        # Convert timestamps to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        # Format for return
        result = df.to_dict('records')
        return result
    
    except Exception as e:
        logger.error(f"Error fetching historical data: {str(e)}")
        raise

def update_market_data(symbol, interval):
    """
    Update market data in the database.
    
    Args:
        symbol (str): Trading pair symbol (e.g., 'BTCUSDT')
        interval (str): Kline interval (e.g., '1m', '5m', '1h', '1d')
    """
    try:
        # Get database connection
        db_path = os.path.join(os.path.dirname(__file__), '../../instance/finance.db')
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create table if it doesn't exist
        table_name = f"klines_{symbol.lower()}_{interval}"
        cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {table_name} (
                timestamp TEXT PRIMARY KEY,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                close_time TEXT,
                quote_asset_volume REAL,
                number_of_trades INTEGER,
                taker_buy_base_asset_volume REAL,
                taker_buy_quote_asset_volume REAL
            )
        ''')
        
        # Get the latest timestamp in the database
        cursor.execute(f"SELECT MAX(timestamp) FROM {table_name}")
        latest_timestamp = cursor.fetchone()[0]
        
        # Fetch data from Binance
        limit = 1000  # Maximum allowed by Binance
        data = get_historical_data(symbol, interval, limit)
        
        # Filter out data we already have
        if latest_timestamp:
            latest_dt = datetime.fromisoformat(latest_timestamp.replace('Z', '+00:00'))
            new_data = [d for d in data if datetime.fromisoformat(str(d['timestamp']).replace('Z', '+00:00')) > latest_dt]
        else:
            new_data = data
        
        # Insert new data
        for kline in new_data:
            cursor.execute(f'''
                INSERT OR REPLACE INTO {table_name} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                kline['timestamp'].isoformat(),
                kline['open'],
                kline['high'],
                kline['low'],
                kline['close'],
                kline['volume'],
                kline['close_time'].isoformat(),
                kline['quote_asset_volume'],
                kline['number_of_trades'],
                kline['taker_buy_base_asset_volume'],
                kline['taker_buy_quote_asset_volume']
            ))
        
        conn.commit()
        logger.info(f"Updated {len(new_data)} new klines for {symbol} {interval}")
        
    except Exception as e:
        logger.error(f"Error updating market data: {str(e)}")
        if 'conn' in locals():
            conn.rollback()
    finally:
        if 'conn' in locals():
            conn.close()

def get_account_info():
    """
    Get account information from Binance.
    
    Returns:
        dict: Account information
    """
    config = load_config()
    api_key = config.get('apis', {}).get('binance', {}).get('api_key', '')
    api_secret = config.get('apis', {}).get('binance', {}).get('api_secret', '')
    
    if not api_key or not api_secret:
        logger.error("API key and secret required for account information")
        return {"error": "API key and secret required"}
    
    base_url = get_api_url()
    endpoint = f"{base_url}/v3/account"
    
    # Add timestamp and signature for authenticated request
    timestamp = int(time.time() * 1000)
    params = {
        'timestamp': timestamp
    }
    
    # Create signature
    import hmac
    import hashlib
    query_string = '&'.join([f"{key}={params[key]}" for key in params])
    signature = hmac.new(
        api_secret.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    params['signature'] = signature
    
    try:
        response = requests.get(
            endpoint, 
            params=params, 
            headers=get_headers()
        )
        response.raise_for_status()
        
        return response.json()
    except Exception as e:
        logger.error(f"Error fetching account info: {str(e)}")
        return {"error": str(e)}

def get_exchange_info():
    """
    Get exchange information from Binance.
    
    Returns:
        dict: Exchange information
    """
    base_url = get_api_url()
    endpoint = f"{base_url}/v3/exchangeInfo"
    
    try:
        response = requests.get(endpoint, headers=get_headers())
        response.raise_for_status()
        
        return response.json()
    except Exception as e:
        logger.error(f"Error fetching exchange info: {str(e)}")
        return {"error": str(e)}