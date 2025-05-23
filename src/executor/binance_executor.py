"""
Module for executing trades on Binance exchange.
"""
import logging
import os
import yaml
import json
import time
import hmac
import hashlib
import requests
import sqlite3
from datetime import datetime

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
            },
            'trading': {
                'enabled': False,
                'risk_per_trade': 0.01,
                'max_open_trades': 3,
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.04
            }
        }

def get_api_url():
    """Get Binance API URL based on configuration."""
    config = load_config()
    testnet = config.get('apis', {}).get('binance', {}).get('testnet', True)
    return "https://testnet.binance.vision/api" if testnet else "https://api.binance.com/api"

def get_headers(api_key):
    """Get headers for Binance API requests."""
    headers = {
        'X-MBX-APIKEY': api_key,
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    return headers

def get_signature(query_string, api_secret):
    """
    Generate signature for Binance API requests.
    
    Args:
        query_string (str): Query string to sign
        api_secret (str): API secret key
    
    Returns:
        str: HMAC SHA256 signature
    """
    signature = hmac.new(
        api_secret.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    return signature

def get_account_info(api_key, api_secret):
    """
    Get account information from Binance.
    
    Args:
        api_key (str): API key
        api_secret (str): API secret key
    
    Returns:
        dict: Account information
    """
    try:
        base_url = get_api_url()
        endpoint = f"{base_url}/v3/account"
        
        # Add timestamp for signature
        timestamp = int(time.time() * 1000)
        params = {
            'timestamp': timestamp
        }
        
        # Create query string
        query_string = '&'.join([f"{key}={params[key]}" for key in params])
        
        # Add signature
        signature = get_signature(query_string, api_secret)
        query_string += f"&signature={signature}"
        
        # Make request
        response = requests.get(
            f"{endpoint}?{query_string}",
            headers=get_headers(api_key)
        )
        
        if response.status_code != 200:
            logger.error(f"Error getting account info: {response.text}")
            return None
        
        return response.json()
    
    except Exception as e:
        logger.error(f"Error getting account info: {str(e)}")
        return None

def create_order(symbol, side, quantity, api_key, api_secret, order_type='MARKET', price=None, stop_price=None):
    """
    Create an order on Binance.
    
    Args:
        symbol (str): Trading pair symbol
        side (str): Order side (BUY or SELL)
        quantity (float): Order quantity
        api_key (str): API key
        api_secret (str): API secret key
        order_type (str): Order type (MARKET, LIMIT, STOP_LOSS, etc.)
        price (float, optional): Order price (required for LIMIT orders)
        stop_price (float, optional): Stop price (required for STOP_LOSS orders)
    
    Returns:
        dict: Order response
    """
    try:
        base_url = get_api_url()
        endpoint = f"{base_url}/v3/order"
        
        # Prepare parameters
        params = {
            'symbol': symbol.upper(),
            'side': side.upper(),
            'type': order_type.upper(),
            'quantity': quantity,
            'timestamp': int(time.time() * 1000)
        }
        
        # Add price for LIMIT orders
        if order_type.upper() == 'LIMIT' and price is not None:
            params['price'] = price
            params['timeInForce'] = 'GTC'  # Good Till Cancel
        
        # Add stop price for STOP_LOSS orders
        if order_type.upper() in ['STOP_LOSS', 'STOP_LOSS_LIMIT'] and stop_price is not None:
            params['stopPrice'] = stop_price
        
        # Create query string
        query_string = '&'.join([f"{key}={params[key]}" for key in params])
        
        # Add signature
        signature = get_signature(query_string, api_secret)
        query_string += f"&signature={signature}"
        
        # Make request
        response = requests.post(
            endpoint,
            headers=get_headers(api_key),
            data=query_string
        )
        
        if response.status_code != 200:
            logger.error(f"Error creating order: {response.text}")
            return None
        
        return response.json()
    
    except Exception as e:
        logger.error(f"Error creating order: {str(e)}")
        return None

def get_symbol_price(symbol):
    """
    Get current price for a symbol.
    
    Args:
        symbol (str): Trading pair symbol
    
    Returns:
        float: Current price
    """
    try:
        base_url = get_api_url()
        endpoint = f"{base_url}/v3/ticker/price"
        
        params = {
            'symbol': symbol.upper()
        }
        
        response = requests.get(endpoint, params=params)
        
        if response.status_code != 200:
            logger.error(f"Error getting symbol price: {response.text}")
            return None
        
        data = response.json()
        return float(data['price'])
    
    except Exception as e:
        logger.error(f"Error getting symbol price: {str(e)}")
        return None

def calculate_quantity(symbol, amount, side):
    """
    Calculate order quantity based on amount.
    
    Args:
        symbol (str): Trading pair symbol
        amount (float): Amount to trade
        side (str): Order side (BUY or SELL)
    
    Returns:
        float: Order quantity
    """
    try:
        # Get current price
        price = get_symbol_price(symbol)
        
        if price is None:
            logger.error(f"Could not get price for {symbol}")
            return None
        
        # Calculate quantity
        quantity = amount / price
        
        # Round quantity based on symbol precision
        # In a real implementation, you would get the precision from exchange info
        # For now, we'll just round to 5 decimal places
        quantity = round(quantity, 5)
        
        return quantity
    
    except Exception as e:
        logger.error(f"Error calculating quantity: {str(e)}")
        return None

def execute_signal(signal, user_api_key, user_api_secret):
    """
    Execute a trading signal.
    
    Args:
        signal (dict): Trading signal
        user_api_key (str): User's API key
        user_api_secret (str): User's API secret key
    
    Returns:
        dict: Execution result
    """
    try:
        logger.info(f"Executing signal for {signal['symbol']}")
        
        config = load_config()
        trading_config = config.get('trading', {})
        
        # Get user account balance
        account = get_account_info(user_api_key, user_api_secret)
        
        if not account or 'balances' not in account:
            logger.error("Could not get account info")
            return {
                'success': False,
                'error': 'Could not get account info'
            }
        
        # Calculate trade amount
        risk_per_trade = trading_config.get('risk_per_trade', 0.01)  # 1% of balance
        
        # Find the base currency balance (e.g., USDT for BTCUSDT)
        # For simplicity, we'll assume it's always USDT
        base_currency = 'USDT'
        
        # Find base currency balance
        base_balance = 0
        for balance in account['balances']:
            if balance['asset'] == base_currency:
                base_balance = float(balance['free'])
                break
        
        # Calculate trade amount
        trade_amount = base_balance * risk_per_trade
        
        # Calculate quantity
        quantity = calculate_quantity(signal['symbol'], trade_amount, signal['side'])
        
        if not quantity:
            logger.error("Could not calculate quantity")
            return {
                'success': False,
                'error': 'Could not calculate quantity'
            }
        
        # Execute order
        order = create_order(
            symbol=signal['symbol'],
            side=signal['side'],
            quantity=quantity,
            api_key=user_api_key,
            api_secret=user_api_secret,
            order_type=signal.get('entry_type', 'MARKET')
        )
        
        if not order:
            logger.error("Order execution failed")
            return {
                'success': False,
                'error': 'Order execution failed'
            }
        
        # Get entry price
        entry_price = float(order.get('price', 0))
        if entry_price == 0 and 'fills' in order:
            # For market orders, calculate average price from fills
            total_qty = 0
            total_cost = 0
            for fill in order['fills']:
                qty = float(fill['qty'])
                price = float(fill['price'])
                total_qty += qty
                total_cost += qty * price
            
            if total_qty > 0:
                entry_price = total_cost / total_qty
        
        # Calculate stop loss and take profit levels
        stop_loss_pct = trading_config.get('stop_loss_pct', 0.02)  # 2%
        take_profit_pct = trading_config.get('take_profit_pct', 0.04)  # 4%
        
        stop_price = None
        take_profit = None
        
        if entry_price > 0:
            if signal['side'] == 'BUY':
                stop_price = entry_price * (1 - stop_loss_pct)
                take_profit = entry_price * (1 + take_profit_pct)
            else:  # SELL
                stop_price = entry_price * (1 + stop_loss_pct)
                take_profit = entry_price * (1 - take_profit_pct)
        
        # Store trade in database
        db_path = os.path.join(os.path.dirname(__file__), '../../instance/finance.db')
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO trade
            (symbol, entry_price, quantity, side, status, entry_time, strategy_id, user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            signal['symbol'],
            entry_price,
            quantity,
            signal['side'],
            'OPEN',
            datetime.now().isoformat(),
            signal.get('strategy_id'),
            signal.get('user_id')
        ))
        
        trade_id = cursor.lastrowid
        
        # Create entry in trade_details
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trade_details (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id INTEGER,
                stop_price REAL,
                take_profit REAL,
                order_data TEXT,
                signal_data TEXT,
                FOREIGN KEY (trade_id) REFERENCES trade (id)
            )
        ''')
        
        cursor.execute('''
            INSERT INTO trade_details
            (trade_id, stop_price, take_profit, order_data, signal_data)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            trade_id,
            stop_price,
            take_profit,
            json.dumps(order),
            json.dumps(signal.get('signal_data', {}))
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Trade executed successfully: {signal['side']} {quantity} {signal['symbol']} at {entry_price}")
        
        return {
            'success': True,
            'trade_id': trade_id,
            'symbol': signal['symbol'],
            'side': signal['side'],
            'quantity': quantity,
            'entry_price': entry_price,
            'stop_price': stop_price,
            'take_profit': take_profit
        }
    
    except Exception as e:
        logger.error(f"Error executing signal: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

def execute_signals(signals):
    """
    Execute multiple trading signals.
    
    Args:
        signals (list): List of trading signals
    
    Returns:
        list: Execution results
    """
    try:
        logger.info(f"Executing {len(signals)} trading signals")
        
        results = []
        
        for signal in signals:
            # Get user API keys
            user_id = signal.get('user_id')
            
            if not user_id:
                logger.warning("Signal missing user_id, skipping")
                continue
            
            # Get user API keys from database
            db_path = os.path.join(os.path.dirname(__file__), '../../instance/finance.db')
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT api_key, api_secret FROM user
                WHERE id = ?
            ''', (user_id,))
            
            user = cursor.fetchone()
            conn.close()
            
            if not user or not user[0] or not user[1]:
                logger.warning(f"User {user_id} has no API keys configured, skipping")
                continue
            
            api_key, api_secret = user
            
            # Execute signal
            result = execute_signal(signal, api_key, api_secret)
            results.append(result)
        
        logger.info(f"Executed {len(results)} signals")
        return results
    
    except Exception as e:
        logger.error(f"Error executing signals: {str(e)}")
        return []

def close_trade(trade_id, user_id):
    """
    Close an open trade.
    
    Args:
        trade_id (int): Trade ID
        user_id (int): User ID
    
    Returns:
        dict: Close result
    """
    try:
        logger.info(f"Closing trade {trade_id}")
        
        # Get trade from database
        db_path = os.path.join(os.path.dirname(__file__), '../../instance/finance.db')
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT symbol, quantity, side, entry_price FROM trade
            WHERE id = ? AND user_id = ? AND status = 'OPEN'
        ''', (trade_id, user_id))
        
        trade = cursor.fetchone()
        
        if not trade:
            logger.warning(f"Trade {trade_id} not found or not open")
            return {
                'success': False,
                'error': 'Trade not found or not open'
            }
        
        symbol, quantity, side, entry_price = trade
        
        # Get user API keys
        cursor.execute('''
            SELECT api_key, api_secret FROM user
            WHERE id = ?
        ''', (user_id,))
        
        user = cursor.fetchone()
        
        if not user or not user[0] or not user[1]:
            logger.warning(f"User {user_id} has no API keys configured")
            return {
                'success': False,
                'error': 'User API keys not configured'
            }
        
        api_key, api_secret = user
        
        # Close position
        close_side = 'SELL' if side == 'BUY' else 'BUY'
        
        order = create_order(
            symbol=symbol,
            side=close_side,
            quantity=quantity,
            api_key=api_key,
            api_secret=api_secret,
            order_type='MARKET'
        )
        
        if not order:
            logger.error("Close order execution failed")
            return {
                'success': False,
                'error': 'Close order execution failed'
            }
        
        # Get exit price
        exit_price = float(order.get('price', 0))
        if exit_price == 0 and 'fills' in order:
            # For market orders, calculate average price from fills
            total_qty = 0
            total_cost = 0
            for fill in order['fills']:
                qty = float(fill['qty'])
                price = float(fill['price'])
                total_qty += qty
                total_cost += qty * price
            
            if total_qty > 0:
                exit_price = total_cost / total_qty
        
        # Calculate PnL
        if side == 'BUY':
            pnl = (exit_price - entry_price) * quantity
        else:  # SELL
            pnl = (entry_price - exit_price) * quantity
        
        # Update trade in database
        cursor.execute('''
            UPDATE trade
            SET status = 'CLOSED', exit_price = ?, exit_time = ?, pnl = ?
            WHERE id = ?
        ''', (
            exit_price,
            datetime.now().isoformat(),
            pnl,
            trade_id
        ))
        
        # Update trade details
        cursor.execute('''
            UPDATE trade_details
            SET close_order_data = ?
            WHERE trade_id = ?
        ''', (
            json.dumps(order),
            trade_id
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Trade {trade_id} closed: {close_side} {quantity} {symbol} at {exit_price}, PnL: {pnl}")
        
        return {
            'success': True,
            'trade_id': trade_id,
            'symbol': symbol,
            'side': close_side,
            'quantity': quantity,
            'exit_price': exit_price,
            'pnl': pnl
        }
    
    except Exception as e:
        logger.error(f"Error closing trade: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

def check_stop_loss_take_profit():
    """Check open trades for stop loss and take profit conditions."""
    try:
        logger.info("Checking stop loss and take profit conditions")
        
        # Get open trades from database
        db_path = os.path.join(os.path.dirname(__file__), '../../instance/finance.db')
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT t.id, t.symbol, t.quantity, t.side, t.entry_price, t.user_id,
                   td.stop_price, td.take_profit
            FROM trade t
            JOIN trade_details td ON t.id = td.trade_id
            WHERE t.status = 'OPEN'
        ''')
        
        trades = cursor.fetchall()
        conn.close()
        
        for trade in trades:
            trade_id, symbol, quantity, side, entry_price, user_id, stop_price, take_profit = trade
            
            # Get current price
            current_price = get_symbol_price(symbol)
            
            if current_price is None:
                logger.warning(f"Could not get price for {symbol}")
                continue
            
            # Check stop loss and take profit conditions
            if side == 'BUY':
                # Long position
                if stop_price and current_price <= stop_price:
                    logger.info(f"Stop loss triggered for trade {trade_id}: {current_price} <= {stop_price}")
                    close_trade(trade_id, user_id)
                elif take_profit and current_price >= take_profit:
                    logger.info(f"Take profit triggered for trade {trade_id}: {current_price} >= {take_profit}")
                    close_trade(trade_id, user_id)
            else:
                # Short position
                if stop_price and current_price >= stop_price:
                    logger.info(f"Stop loss triggered for trade {trade_id}: {current_price} >= {stop_price}")
                    close_trade(trade_id, user_id)
                elif take_profit and current_price <= take_profit:
                    logger.info(f"Take profit triggered for trade {trade_id}: {current_price} <= {take_profit}")
                    close_trade(trade_id, user_id)
    
    except Exception as e:
        logger.error(f"Error checking stop loss and take profit: {str(e)}")