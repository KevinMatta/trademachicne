"""
Module for ICT (Inner Circle Trader) market structure analysis.
"""
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sqlite3

from ..preprocessing.data_cleaning import load_market_data_from_db

logger = logging.getLogger(__name__)

def identify_swings(df, threshold=0.002, swing_periods=3):
    """
    Identify swing points in the market structure.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLC data
        threshold (float): Minimum price change to consider as a swing (in % of price)
        swing_periods (int): Number of periods to look back and forward
    
    Returns:
        pd.DataFrame: DataFrame with swing points identified
    """
    try:
        logger.info("Identifying swing points")
        
        if df.empty:
            logger.warning("Empty DataFrame provided")
            return df
        
        # Make a copy to avoid modifying the original
        df_copy = df.copy()
        
        # Check if required columns exist
        required_cols = ['high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df_copy.columns]
        
        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")
            return df_copy
        
        # Initialize swing columns
        df_copy['swing_high'] = False
        df_copy['swing_low'] = False
        
        # Identify swing highs
        for i in range(swing_periods, len(df_copy) - swing_periods):
            # Check if the current high is higher than surrounding highs
            is_swing_high = True
            current_high = df_copy['high'].iloc[i]
            
            # Look back
            for j in range(1, swing_periods + 1):
                if df_copy['high'].iloc[i - j] >= current_high:
                    is_swing_high = False
                    break
            
            # Look forward
            if is_swing_high:
                for j in range(1, swing_periods + 1):
                    if df_copy['high'].iloc[i + j] >= current_high:
                        is_swing_high = False
                        break
            
            # Check threshold
            if is_swing_high:
                # Calculate minimum price change (as % of price)
                min_change = current_high * threshold
                
                # Check if any surrounding high is within threshold
                for j in range(1, swing_periods + 1):
                    if current_high - df_copy['high'].iloc[i - j] < min_change:
                        is_swing_high = False
                        break
                
                for j in range(1, swing_periods + 1):
                    if current_high - df_copy['high'].iloc[i + j] < min_change:
                        is_swing_high = False
                        break
            
            df_copy.loc[df_copy.index[i], 'swing_high'] = is_swing_high
        
        # Identify swing lows
        for i in range(swing_periods, len(df_copy) - swing_periods):
            # Check if the current low is lower than surrounding lows
            is_swing_low = True
            current_low = df_copy['low'].iloc[i]
            
            # Look back
            for j in range(1, swing_periods + 1):
                if df_copy['low'].iloc[i - j] <= current_low:
                    is_swing_low = False
                    break
            
            # Look forward
            if is_swing_low:
                for j in range(1, swing_periods + 1):
                    if df_copy['low'].iloc[i + j] <= current_low:
                        is_swing_low = False
                        break
            
            # Check threshold
            if is_swing_low:
                # Calculate minimum price change (as % of price)
                min_change = current_low * threshold
                
                # Check if any surrounding low is within threshold
                for j in range(1, swing_periods + 1):
                    if df_copy['low'].iloc[i - j] - current_low < min_change:
                        is_swing_low = False
                        break
                
                for j in range(1, swing_periods + 1):
                    if df_copy['low'].iloc[i + j] - current_low < min_change:
                        is_swing_low = False
                        break
            
            df_copy.loc[df_copy.index[i], 'swing_low'] = is_swing_low
        
        logger.info(f"Identified {df_copy['swing_high'].sum()} swing highs and {df_copy['swing_low'].sum()} swing lows")
        return df_copy
    
    except Exception as e:
        logger.error(f"Error identifying swings: {str(e)}")
        return df

def identify_fair_value_gaps(df):
    """
    Identify Fair Value Gaps (FVGs) in the market structure.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLC data
    
    Returns:
        pd.DataFrame: DataFrame with FVGs identified
    """
    try:
        logger.info("Identifying Fair Value Gaps")
        
        if df.empty:
            logger.warning("Empty DataFrame provided")
            return df
        
        # Make a copy to avoid modifying the original
        df_copy = df.copy()
        
        # Check if required columns exist
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df_copy.columns]
        
        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")
            return df_copy
        
        # Initialize FVG columns
        df_copy['bullish_fvg'] = False
        df_copy['bearish_fvg'] = False
        df_copy['bullish_fvg_top'] = None
        df_copy['bullish_fvg_bottom'] = None
        df_copy['bearish_fvg_top'] = None
        df_copy['bearish_fvg_bottom'] = None
        
        # Identify Bullish FVGs (Low1 > High3)
        for i in range(2, len(df_copy)):
            if df_copy['low'].iloc[i-2] > df_copy['high'].iloc[i]:
                df_copy.loc[df_copy.index[i-1], 'bullish_fvg'] = True
                df_copy.loc[df_copy.index[i-1], 'bullish_fvg_top'] = df_copy['low'].iloc[i-2]
                df_copy.loc[df_copy.index[i-1], 'bullish_fvg_bottom'] = df_copy['high'].iloc[i]
        
        # Identify Bearish FVGs (High1 < Low3)
        for i in range(2, len(df_copy)):
            if df_copy['high'].iloc[i-2] < df_copy['low'].iloc[i]:
                df_copy.loc[df_copy.index[i-1], 'bearish_fvg'] = True
                df_copy.loc[df_copy.index[i-1], 'bearish_fvg_top'] = df_copy['low'].iloc[i]
                df_copy.loc[df_copy.index[i-1], 'bearish_fvg_bottom'] = df_copy['high'].iloc[i-2]
        
        logger.info(f"Identified {df_copy['bullish_fvg'].sum()} bullish FVGs and {df_copy['bearish_fvg'].sum()} bearish FVGs")
        return df_copy
    
    except Exception as e:
        logger.error(f"Error identifying Fair Value Gaps: {str(e)}")
        return df

def identify_order_blocks(df):
    """
    Identify Order Blocks in the market structure.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLC data
    
    Returns:
        pd.DataFrame: DataFrame with Order Blocks identified
    """
    try:
        logger.info("Identifying Order Blocks")
        
        if df.empty:
            logger.warning("Empty DataFrame provided")
            return df
        
        # Make a copy to avoid modifying the original
        df_copy = df.copy()
        
        # Check if required columns exist
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df_copy.columns]
        
        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")
            return df_copy
        
        # First identify swing points
        df_copy = identify_swings(df_copy)
        
        # Initialize Order Block columns
        df_copy['bullish_ob'] = False
        df_copy['bearish_ob'] = False
        df_copy['bullish_ob_top'] = None
        df_copy['bullish_ob_bottom'] = None
        df_copy['bearish_ob_top'] = None
        df_copy['bearish_ob_bottom'] = None
        
        # Find bullish order blocks (before swing lows)
        swing_lows = df_copy[df_copy['swing_low']].index.tolist()
        
        for low_idx in swing_lows:
            # Find the index of this swing low
            i = df_copy.index.get_loc(low_idx)
            
            # Look back for a bearish candle with a close below open
            for j in range(i-1, max(0, i-5), -1):
                if df_copy['close'].iloc[j] < df_copy['open'].iloc[j]:
                    # This is a bearish candle, mark it as a bullish order block
                    df_copy.loc[df_copy.index[j], 'bullish_ob'] = True
                    df_copy.loc[df_copy.index[j], 'bullish_ob_top'] = df_copy['high'].iloc[j]
                    df_copy.loc[df_copy.index[j], 'bullish_ob_bottom'] = df_copy['low'].iloc[j]
                    break
        
        # Find bearish order blocks (before swing highs)
        swing_highs = df_copy[df_copy['swing_high']].index.tolist()
        
        for high_idx in swing_highs:
            # Find the index of this swing high
            i = df_copy.index.get_loc(high_idx)
            
            # Look back for a bullish candle with a close above open
            for j in range(i-1, max(0, i-5), -1):
                if df_copy['close'].iloc[j] > df_copy['open'].iloc[j]:
                    # This is a bullish candle, mark it as a bearish order block
                    df_copy.loc[df_copy.index[j], 'bearish_ob'] = True
                    df_copy.loc[df_copy.index[j], 'bearish_ob_top'] = df_copy['high'].iloc[j]
                    df_copy.loc[df_copy.index[j], 'bearish_ob_bottom'] = df_copy['low'].iloc[j]
                    break
        
        logger.info(f"Identified {df_copy['bullish_ob'].sum()} bullish OBs and {df_copy['bearish_ob'].sum()} bearish OBs")
        return df_copy
    
    except Exception as e:
        logger.error(f"Error identifying Order Blocks: {str(e)}")
        return df

def identify_liquidity_levels(df):
    """
    Identify liquidity levels in the market structure.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLC data
    
    Returns:
        pd.DataFrame: DataFrame with liquidity levels identified
    """
    try:
        logger.info("Identifying liquidity levels")
        
        if df.empty:
            logger.warning("Empty DataFrame provided")
            return df
        
        # Make a copy to avoid modifying the original
        df_copy = df.copy()
        
        # Check if required columns exist
        required_cols = ['high', 'low']
        missing_cols = [col for col in required_cols if col not in df_copy.columns]
        
        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")
            return df_copy
        
        # First identify swing points
        df_copy = identify_swings(df_copy)
        
        # Initialize liquidity level columns
        df_copy['buy_liquidity'] = False
        df_copy['sell_liquidity'] = False
        df_copy['buy_liquidity_level'] = None
        df_copy['sell_liquidity_level'] = None
        
        # Find buy liquidity below swing lows
        swing_lows = df_copy[df_copy['swing_low']].index.tolist()
        
        for low_idx in swing_lows:
            i = df_copy.index.get_loc(low_idx)
            low_price = df_copy['low'].iloc[i]
            
            # Mark liquidity level
            df_copy.loc[df_copy.index[i], 'buy_liquidity'] = True
            df_copy.loc[df_copy.index[i], 'buy_liquidity_level'] = low_price
        
        # Find sell liquidity above swing highs
        swing_highs = df_copy[df_copy['swing_high']].index.tolist()
        
        for high_idx in swing_highs:
            i = df_copy.index.get_loc(high_idx)
            high_price = df_copy['high'].iloc[i]
            
            # Mark liquidity level
            df_copy.loc[df_copy.index[i], 'sell_liquidity'] = True
            df_copy.loc[df_copy.index[i], 'sell_liquidity_level'] = high_price
        
        logger.info(f"Identified {df_copy['buy_liquidity'].sum()} buy liquidity levels and {df_copy['sell_liquidity'].sum()} sell liquidity levels")
        return df_copy
    
    except Exception as e:
        logger.error(f"Error identifying liquidity levels: {str(e)}")
        return df

def identify_breaker_blocks(df):
    """
    Identify breaker blocks in the market structure.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLC data
    
    Returns:
        pd.DataFrame: DataFrame with breaker blocks identified
    """
    try:
        logger.info("Identifying breaker blocks")
        
        if df.empty:
            logger.warning("Empty DataFrame provided")
            return df
        
        # Make a copy to avoid modifying the original
        df_copy = df.copy()
        
        # First identify order blocks
        df_copy = identify_order_blocks(df_copy)
        
        # Initialize breaker block columns
        df_copy['bullish_breaker'] = False
        df_copy['bearish_breaker'] = False
        
        # Find bullish breaker blocks (bullish OB that price revisits from above)
        bullish_obs = df_copy[df_copy['bullish_ob']].index.tolist()
        
        for ob_idx in bullish_obs:
            i = df_copy.index.get_loc(ob_idx)
            ob_top = df_copy['bullish_ob_top'].iloc[i]
            ob_bottom = df_copy['bullish_ob_bottom'].iloc[i]
            
            # Look forward to see if price revisits this OB from above
            revisited = False
            for j in range(i+1, len(df_copy)):
                if df_copy['low'].iloc[j] <= ob_top and not revisited:
                    # Price has revisited the OB
                    revisited = True
                
                if revisited and df_copy['close'].iloc[j] > ob_top:
                    # Price has broken back above the OB after revisiting
                    df_copy.loc[df_copy.index[i], 'bullish_breaker'] = True
                    break
        
        # Find bearish breaker blocks (bearish OB that price revisits from below)
        bearish_obs = df_copy[df_copy['bearish_ob']].index.tolist()
        
        for ob_idx in bearish_obs:
            i = df_copy.index.get_loc(ob_idx)
            ob_top = df_copy['bearish_ob_top'].iloc[i]
            ob_bottom = df_copy['bearish_ob_bottom'].iloc[i]
            
            # Look forward to see if price revisits this OB from below
            revisited = False
            for j in range(i+1, len(df_copy)):
                if df_copy['high'].iloc[j] >= ob_bottom and not revisited:
                    # Price has revisited the OB
                    revisited = True
                
                if revisited and df_copy['close'].iloc[j] < ob_bottom:
                    # Price has broken back below the OB after revisiting
                    df_copy.loc[df_copy.index[i], 'bearish_breaker'] = True
                    break
        
        logger.info(f"Identified {df_copy['bullish_breaker'].sum()} bullish breakers and {df_copy['bearish_breaker'].sum()} bearish breakers")
        return df_copy
    
    except Exception as e:
        logger.error(f"Error identifying breaker blocks: {str(e)}")
        return df

def analyze_market_structure(symbol, interval):
    """
    Perform a comprehensive ICT market structure analysis.
    
    Args:
        symbol (str): Trading pair symbol
        interval (str): Kline interval
    
    Returns:
        dict: Market structure analysis
    """
    try:
        logger.info(f"Analyzing market structure for {symbol} {interval}")
        
        # Load market data
        df = load_market_data_from_db(symbol, interval)
        
        if df.empty:
            logger.warning(f"No market data available for {symbol} {interval}")
            return {}
        
        # Identify all ICT elements
        df = identify_swings(df)
        df = identify_fair_value_gaps(df)
        df = identify_order_blocks(df)
        df = identify_liquidity_levels(df)
        df = identify_breaker_blocks(df)
        
        # Get current price
        current_price = df['close'].iloc[-1]
        
        # Collect active levels
        active_levels = {
            'current_price': current_price,
            'swing_highs': [],
            'swing_lows': [],
            'bullish_fvgs': [],
            'bearish_fvgs': [],
            'bullish_obs': [],
            'bearish_obs': [],
            'buy_liquidity': [],
            'sell_liquidity': [],
            'bullish_breakers': [],
            'bearish_breakers': []
        }
        
        # Only include recent levels (last 100 candles)
        recent_df = df.iloc[-100:]
        
        # Extract swing points
        for idx, row in recent_df[recent_df['swing_high']].iterrows():
            active_levels['swing_highs'].append({
                'timestamp': row['timestamp'].isoformat() if isinstance(row['timestamp'], datetime) else row['timestamp'],
                'price': row['high']
            })
        
        for idx, row in recent_df[recent_df['swing_low']].iterrows():
            active_levels['swing_lows'].append({
                'timestamp': row['timestamp'].isoformat() if isinstance(row['timestamp'], datetime) else row['timestamp'],
                'price': row['low']
            })
        
        # Extract FVGs
        for idx, row in recent_df[recent_df['bullish_fvg']].iterrows():
            active_levels['bullish_fvgs'].append({
                'timestamp': row['timestamp'].isoformat() if isinstance(row['timestamp'], datetime) else row['timestamp'],
                'top': row['bullish_fvg_top'],
                'bottom': row['bullish_fvg_bottom']
            })
        
        for idx, row in recent_df[recent_df['bearish_fvg']].iterrows():
            active_levels['bearish_fvgs'].append({
                'timestamp': row['timestamp'].isoformat() if isinstance(row['timestamp'], datetime) else row['timestamp'],
                'top': row['bearish_fvg_top'],
                'bottom': row['bearish_fvg_bottom']
            })
        
        # Extract Order Blocks
        for idx, row in recent_df[recent_df['bullish_ob']].iterrows():
            active_levels['bullish_obs'].append({
                'timestamp': row['timestamp'].isoformat() if isinstance(row['timestamp'], datetime) else row['timestamp'],
                'top': row['bullish_ob_top'],
                'bottom': row['bullish_ob_bottom']
            })
        
        for idx, row in recent_df[recent_df['bearish_ob']].iterrows():
            active_levels['bearish_obs'].append({
                'timestamp': row['timestamp'].isoformat() if isinstance(row['timestamp'], datetime) else row['timestamp'],
                'top': row['bearish_ob_top'],
                'bottom': row['bearish_ob_bottom']
            })
        
        # Extract Liquidity Levels
        for idx, row in recent_df[recent_df['buy_liquidity']].iterrows():
            active_levels['buy_liquidity'].append({
                'timestamp': row['timestamp'].isoformat() if isinstance(row['timestamp'], datetime) else row['timestamp'],
                'price': row['buy_liquidity_level']
            })
        
        for idx, row in recent_df[recent_df['sell_liquidity']].iterrows():
            active_levels['sell_liquidity'].append({
                'timestamp': row['timestamp'].isoformat() if isinstance(row['timestamp'], datetime) else row['timestamp'],
                'price': row['sell_liquidity_level']
            })
        
        # Extract Breaker Blocks
        for idx, row in recent_df[recent_df['bullish_breaker']].iterrows():
            active_levels['bullish_breakers'].append({
                'timestamp': row['timestamp'].isoformat() if isinstance(row['timestamp'], datetime) else row['timestamp'],
                'top': row['bullish_ob_top'],
                'bottom': row['bullish_ob_bottom']
            })
        
        for idx, row in recent_df[recent_df['bearish_breaker']].iterrows():
            active_levels['bearish_breakers'].append({
                'timestamp': row['timestamp'].isoformat() if isinstance(row['timestamp'], datetime) else row['timestamp'],
                'top': row['bearish_ob_top'],
                'bottom': row['bearish_ob_bottom']
            })
        
        # Sort levels by price
        active_levels['swing_highs'] = sorted(active_levels['swing_highs'], key=lambda x: x['price'], reverse=True)
        active_levels['swing_lows'] = sorted(active_levels['swing_lows'], key=lambda x: x['price'])
        active_levels['buy_liquidity'] = sorted(active_levels['buy_liquidity'], key=lambda x: x['price'])
        active_levels['sell_liquidity'] = sorted(active_levels['sell_liquidity'], key=lambda x: x['price'], reverse=True)
        
        # Identify nearby levels for trading decisions
        nearby_resistance = []
        nearby_support = []
        
        # Add swing highs above current price as resistance
        for level in active_levels['swing_highs']:
            if level['price'] > current_price:
                nearby_resistance.append({
                    'type': 'swing_high',
                    'price': level['price'],
                    'distance': (level['price'] / current_price - 1) * 100  # distance in percent
                })
        
        # Add swing lows below current price as support
        for level in active_levels['swing_lows']:
            if level['price'] < current_price:
                nearby_support.append({
                    'type': 'swing_low',
                    'price': level['price'],
                    'distance': (1 - level['price'] / current_price) * 100  # distance in percent
                })
        
        # Add FVGs as support/resistance
        for fvg in active_levels['bullish_fvgs']:
            if fvg['bottom'] < current_price < fvg['top']:
                # Current price is inside the FVG
                nearby_support.append({
                    'type': 'bullish_fvg_bottom',
                    'price': fvg['bottom'],
                    'distance': (1 - fvg['bottom'] / current_price) * 100
                })
                nearby_resistance.append({
                    'type': 'bullish_fvg_top',
                    'price': fvg['top'],
                    'distance': (fvg['top'] / current_price - 1) * 100
                })
            elif fvg['bottom'] > current_price:
                nearby_resistance.append({
                    'type': 'bullish_fvg_bottom',
                    'price': fvg['bottom'],
                    'distance': (fvg['bottom'] / current_price - 1) * 100
                })
            elif fvg['top'] < current_price:
                nearby_support.append({
                    'type': 'bullish_fvg_top',
                    'price': fvg['top'],
                    'distance': (1 - fvg['top'] / current_price) * 100
                })
        
        for fvg in active_levels['bearish_fvgs']:
            if fvg['bottom'] < current_price < fvg['top']:
                # Current price is inside the FVG
                nearby_support.append({
                    'type': 'bearish_fvg_bottom',
                    'price': fvg['bottom'],
                    'distance': (1 - fvg['bottom'] / current_price) * 100
                })
                nearby_resistance.append({
                    'type': 'bearish_fvg_top',
                    'price': fvg['top'],
                    'distance': (fvg['top'] / current_price - 1) * 100
                })
            elif fvg['bottom'] > current_price:
                nearby_resistance.append({
                    'type': 'bearish_fvg_bottom',
                    'price': fvg['bottom'],
                    'distance': (fvg['bottom'] / current_price - 1) * 100
                })
            elif fvg['top'] < current_price:
                nearby_support.append({
                    'type': 'bearish_fvg_top',
                    'price': fvg['top'],
                    'distance': (1 - fvg['top'] / current_price) * 100
                })
        
        # Sort by distance
        nearby_resistance = sorted(nearby_resistance, key=lambda x: x['distance'])
        nearby_support = sorted(nearby_support, key=lambda x: x['distance'])
        
        # Get most recent structures
        recent_structures = {
            'swing_high': None,
            'swing_low': None,
            'bullish_fvg': None,
            'bearish_fvg': None,
            'bullish_ob': None,
            'bearish_ob': None
        }
        
        # Check for structures in the last 10 candles
        last_10 = df.iloc[-10:]
        
        if last_10[last_10['swing_high']].any():
            idx = last_10[last_10['swing_high']].index[-1]
            recent_structures['swing_high'] = {
                'timestamp': df.loc[idx, 'timestamp'].isoformat() if isinstance(df.loc[idx, 'timestamp'], datetime) else df.loc[idx, 'timestamp'],
                'price': df.loc[idx, 'high']
            }
        
        if last_10[last_10['swing_low']].any():
            idx = last_10[last_10['swing_low']].index[-1]
            recent_structures['swing_low'] = {
                'timestamp': df.loc[idx, 'timestamp'].isoformat() if isinstance(df.loc[idx, 'timestamp'], datetime) else df.loc[idx, 'timestamp'],
                'price': df.loc[idx, 'low']
            }
        
        if last_10[last_10['bullish_fvg']].any():
            idx = last_10[last_10['bullish_fvg']].index[-1]
            recent_structures['bullish_fvg'] = {
                'timestamp': df.loc[idx, 'timestamp'].isoformat() if isinstance(df.loc[idx, 'timestamp'], datetime) else df.loc[idx, 'timestamp'],
                'top': df.loc[idx, 'bullish_fvg_top'],
                'bottom': df.loc[idx, 'bullish_fvg_bottom']
            }
        
        if last_10[last_10['bearish_fvg']].any():
            idx = last_10[last_10['bearish_fvg']].index[-1]
            recent_structures['bearish_fvg'] = {
                'timestamp': df.loc[idx, 'timestamp'].isoformat() if isinstance(df.loc[idx, 'timestamp'], datetime) else df.loc[idx, 'timestamp'],
                'top': df.loc[idx, 'bearish_fvg_top'],
                'bottom': df.loc[idx, 'bearish_fvg_bottom']
            }
        
        if last_10[last_10['bullish_ob']].any():
            idx = last_10[last_10['bullish_ob']].index[-1]
            recent_structures['bullish_ob'] = {
                'timestamp': df.loc[idx, 'timestamp'].isoformat() if isinstance(df.loc[idx, 'timestamp'], datetime) else df.loc[idx, 'timestamp'],
                'top': df.loc[idx, 'bullish_ob_top'],
                'bottom': df.loc[idx, 'bullish_ob_bottom']
            }
        
        if last_10[last_10['bearish_ob']].any():
            idx = last_10[last_10['bearish_ob']].index[-1]
            recent_structures['bearish_ob'] = {
                'timestamp': df.loc[idx, 'timestamp'].isoformat() if isinstance(df.loc[idx, 'timestamp'], datetime) else df.loc[idx, 'timestamp'],
                'top': df.loc[idx, 'bearish_ob_top'],
                'bottom': df.loc[idx, 'bearish_ob_bottom']
            }
        
        # Prepare final analysis
        analysis = {
            'symbol': symbol,
            'interval': interval,
            'current_price': current_price,
            'timestamp': datetime.now().isoformat(),
            'active_levels': active_levels,
            'nearby_resistance': nearby_resistance[:5],  # Top 5 closest resistance levels
            'nearby_support': nearby_support[:5],  # Top 5 closest support levels
            'recent_structures': recent_structures
        }
        
        # Store analysis in database
        db_path = os.path.join(os.path.dirname(__file__), '../../instance/finance.db')
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_structure (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                interval TEXT,
                analysis TEXT,
                created_at TEXT
            )
        ''')
        
        # Insert analysis
        cursor.execute('''
            INSERT INTO market_structure (symbol, interval, analysis, created_at)
            VALUES (?, ?, ?, ?)
        ''', (
            symbol.lower(),
            interval,
            json.dumps(analysis),
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Completed market structure analysis for {symbol} {interval}")
        return analysis
    
    except Exception as e:
        logger.error(f"Error analyzing market structure: {str(e)}")
        return {}

def get_latest_analysis(symbol, interval):
    """
    Get the latest market structure analysis.
    
    Args:
        symbol (str): Trading pair symbol
        interval (str): Kline interval
    
    Returns:
        dict: Market structure analysis
    """
    try:
        # Get analysis from database
        db_path = os.path.join(os.path.dirname(__file__), '../../instance/finance.db')
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT analysis FROM market_structure
            WHERE symbol = ? AND interval = ?
            ORDER BY created_at DESC
            LIMIT 1
        ''', (symbol.lower(), interval))
        
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            logger.warning(f"No analysis found for {symbol} {interval}")
            return {}
        
        analysis = json.loads(result[0])
        return analysis
    
    except Exception as e:
        logger.error(f"Error getting latest analysis: {str(e)}")
        return {}

def update_market_structure_analysis():
    """Update market structure analysis for all symbols and intervals."""
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
            
            logger.info(f"Updating market structure analysis for {symbol} {interval}")
            
            # Analyze market structure
            analyze_market_structure(symbol, interval)
            
            logger.info(f"Updated market structure analysis for {symbol} {interval}")
    
    except Exception as e:
        logger.error(f"Error updating market structure analysis: {str(e)}")