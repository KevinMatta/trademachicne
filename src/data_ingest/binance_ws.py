"""
Module for interacting with Binance WebSocket API for real-time data.
"""
import json
import logging
import threading
import time
import traceback
import websocket
import yaml
import os
import sqlite3
from datetime import datetime

logger = logging.getLogger(__name__)

# Global variables
active_streams = {}
ws_base_url = "wss://stream.binance.com:9443/ws"
testnet_ws_base_url = "wss://testnet.binance.vision/ws"

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
                    'testnet': True
                }
            }
        }

def get_ws_url():
    """Get WebSocket URL based on configuration."""
    config = load_config()
    testnet = config.get('apis', {}).get('binance', {}).get('testnet', True)
    return testnet_ws_base_url if testnet else ws_base_url

class BinanceWebSocket:
    def __init__(self, stream_name, callback=None):
        """
        Initialize a Binance WebSocket connection.
        
        Args:
            stream_name (str): Name of the stream to subscribe to
            callback (function, optional): Callback function to process received data
        """
        self.stream_name = stream_name
        self.callback = callback
        self.ws = None
        self.running = False
        self.reconnect_count = 0
        self.max_reconnects = 5
        self.reconnect_delay = 5  # seconds
        
        # For storing received data
        self.db_path = os.path.join(os.path.dirname(__file__), '../../instance/finance.db')
        self.ensure_db_table()
    
    def ensure_db_table(self):
        """Ensure the database table exists for storing stream data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create a table specific to this stream
        table_name = f"stream_{self.stream_name}"
        cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {table_name} (
                timestamp TEXT PRIMARY KEY,
                data TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def on_message(self, ws, message):
        """Handle received WebSocket message."""
        try:
            data = json.loads(message)
            
            # Store data in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            timestamp = datetime.now().isoformat()
            table_name = f"stream_{self.stream_name}"
            
            cursor.execute(f'''
                INSERT INTO {table_name} (timestamp, data)
                VALUES (?, ?)
            ''', (timestamp, message))
            
            conn.commit()
            conn.close()
            
            # Call the callback if provided
            if self.callback:
                self.callback(data)
                
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {str(e)}")
            logger.error(traceback.format_exc())
    
    def on_error(self, ws, error):
        """Handle WebSocket error."""
        logger.error(f"WebSocket error: {str(error)}")
    
    def on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket connection close."""
        logger.info(f"WebSocket closed: {close_status_code} - {close_msg}")
        
        # Attempt to reconnect if still running
        if self.running and self.reconnect_count < self.max_reconnects:
            self.reconnect_count += 1
            logger.info(f"Attempting to reconnect (attempt {self.reconnect_count}/{self.max_reconnects})...")
            time.sleep(self.reconnect_delay)
            self.connect()
        elif self.reconnect_count >= self.max_reconnects:
            logger.warning(f"Max reconnection attempts reached for {self.stream_name}")
            self.running = False
    
    def on_open(self, ws):
        """Handle WebSocket connection open."""
        logger.info(f"WebSocket connection established for {self.stream_name}")
        self.reconnect_count = 0  # Reset reconnect counter on successful connection
    
    def connect(self):
        """Establish WebSocket connection."""
        ws_url = f"{get_ws_url()}/{self.stream_name}"
        
        self.ws = websocket.WebSocketApp(
            ws_url,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
            on_open=self.on_open
        )
        
        self.running = True
        
        # Start WebSocket connection in a new thread
        wst = threading.Thread(target=self.ws.run_forever)
        wst.daemon = True
        wst.start()
        
        # Add to active streams
        active_streams[self.stream_name] = self
        
        return self
    
    def disconnect(self):
        """Close WebSocket connection."""
        self.running = False
        if self.ws:
            self.ws.close()
        
        # Remove from active streams
        if self.stream_name in active_streams:
            del active_streams[self.stream_name]

def subscribe_kline(symbol, interval, callback=None):
    """
    Subscribe to kline/candlestick data for a symbol and interval.
    
    Args:
        symbol (str): Trading pair symbol (e.g., 'btcusdt')
        interval (str): Kline interval (e.g., '1m', '5m', '1h', '1d')
        callback (function, optional): Callback function to process received data
    
    Returns:
        BinanceWebSocket: WebSocket connection object
    """
    stream_name = f"{symbol.lower()}@kline_{interval}"
    return BinanceWebSocket(stream_name, callback).connect()

def subscribe_ticker(symbol, callback=None):
    """
    Subscribe to 24hr ticker updates for a symbol.
    
    Args:
        symbol (str): Trading pair symbol (e.g., 'btcusdt')
        callback (function, optional): Callback function to process received data
    
    Returns:
        BinanceWebSocket: WebSocket connection object
    """
    stream_name = f"{symbol.lower()}@ticker"
    return BinanceWebSocket(stream_name, callback).connect()

def subscribe_trade(symbol, callback=None):
    """
    Subscribe to trade updates for a symbol.
    
    Args:
        symbol (str): Trading pair symbol (e.g., 'btcusdt')
        callback (function, optional): Callback function to process received data
    
    Returns:
        BinanceWebSocket: WebSocket connection object
    """
    stream_name = f"{symbol.lower()}@trade"
    return BinanceWebSocket(stream_name, callback).connect()

def subscribe_depth(symbol, update_speed="100ms", callback=None):
    """
    Subscribe to order book updates for a symbol.
    
    Args:
        symbol (str): Trading pair symbol (e.g., 'btcusdt')
        update_speed (str): Update speed ('100ms', '1000ms')
        callback (function, optional): Callback function to process received data
    
    Returns:
        BinanceWebSocket: WebSocket connection object
    """
    stream_name = f"{symbol.lower()}@depth@{update_speed}"
    return BinanceWebSocket(stream_name, callback).connect()

def close_all_connections():
    """Close all active WebSocket connections."""
    for stream_name, ws in list(active_streams.items()):
        logger.info(f"Closing WebSocket connection for {stream_name}")
        ws.disconnect()