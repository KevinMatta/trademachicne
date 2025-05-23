"""
Module for trading decision engine that evaluates signals from various sources.
"""
import logging
import os
import yaml
import json
import sqlite3
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from ..analysis.forecasting_model import get_latest_predictions
from ..analysis.nlp_sentiment import get_sentiment_summary
from ..analysis.ict_rules import get_latest_analysis
from ..preprocessing.indicators import get_indicators

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
            'trading': {
                'enabled': False,
                'risk_per_trade': 0.01,
                'max_open_trades': 3,
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.04
            }
        }

def evaluate_technical_signals(symbol, interval):
    """
    Evaluate technical indicators for trading signals.
    
    Args:
        symbol (str): Trading pair symbol
        interval (str): Kline interval
    
    Returns:
        dict: Technical signals
    """
    try:
        logger.info(f"Evaluating technical signals for {symbol} {interval}")
        
        # Get indicators
        df = get_indicators(symbol, interval)
        
        if df.empty:
            logger.warning(f"No indicator data available for {symbol} {interval}")
            return {
                'strength': 0,
                'direction': 'neutral',
                'signals': []
            }
        
        # Get the most recent row
        latest = df.iloc[-1]
        
        # Initialize signals
        signals = []
        bullish_count = 0
        bearish_count = 0
        
        # Check Moving Average signals
        if 'sma_20' in latest and 'sma_50' in latest:
            if latest['sma_20'] > latest['sma_50']:
                signals.append({
                    'name': 'MA Cross',
                    'description': 'SMA 20 above SMA 50',
                    'direction': 'bullish',
                    'strength': 1
                })
                bullish_count += 1
            elif latest['sma_20'] < latest['sma_50']:
                signals.append({
                    'name': 'MA Cross',
                    'description': 'SMA 20 below SMA 50',
                    'direction': 'bearish',
                    'strength': 1
                })
                bearish_count += 1
        
        # Check RSI signals
        if 'rsi_14' in latest:
            if latest['rsi_14'] > 70:
                signals.append({
                    'name': 'RSI',
                    'description': 'RSI above 70 (overbought)',
                    'direction': 'bearish',
                    'strength': 2
                })
                bearish_count += 2
            elif latest['rsi_14'] < 30:
                signals.append({
                    'name': 'RSI',
                    'description': 'RSI below 30 (oversold)',
                    'direction': 'bullish',
                    'strength': 2
                })
                bullish_count += 2
        
        # Check MACD signals
        if 'macd' in latest and 'macd_signal' in latest:
            if latest['macd'] > latest['macd_signal']:
                signals.append({
                    'name': 'MACD',
                    'description': 'MACD above signal line',
                    'direction': 'bullish',
                    'strength': 1.5
                })
                bullish_count += 1.5
            elif latest['macd'] < latest['macd_signal']:
                signals.append({
                    'name': 'MACD',
                    'description': 'MACD below signal line',
                    'direction': 'bearish',
                    'strength': 1.5
                })
                bearish_count += 1.5
        
        # Check Bollinger Bands signals
        if 'bbands_upper' in latest and 'bbands_lower' in latest and 'close' in latest:
            if latest['close'] > latest['bbands_upper']:
                signals.append({
                    'name': 'Bollinger Bands',
                    'description': 'Price above upper band',
                    'direction': 'bearish',
                    'strength': 1.5
                })
                bearish_count += 1.5
            elif latest['close'] < latest['bbands_lower']:
                signals.append({
                    'name': 'Bollinger Bands',
                    'description': 'Price below lower band',
                    'direction': 'bullish',
                    'strength': 1.5
                })
                bullish_count += 1.5
        
        # Determine overall direction and strength
        if bullish_count > bearish_count:
            direction = 'bullish'
            strength = min(bullish_count - bearish_count, 10) / 10  # Scale to 0-1
        elif bearish_count > bullish_count:
            direction = 'bearish'
            strength = min(bearish_count - bullish_count, 10) / 10  # Scale to 0-1
        else:
            direction = 'neutral'
            strength = 0
        
        result = {
            'strength': strength,
            'direction': direction,
            'signals': signals
        }
        
        logger.info(f"Technical signals for {symbol} {interval}: {direction} (strength: {strength:.2f})")
        return result
    
    except Exception as e:
        logger.error(f"Error evaluating technical signals: {str(e)}")
        return {
            'strength': 0,
            'direction': 'neutral',
            'signals': []
        }

def evaluate_sentiment_signals():
    """
    Evaluate sentiment analysis for trading signals.
    
    Returns:
        dict: Sentiment signals
    """
    try:
        logger.info("Evaluating sentiment signals")
        
        # Get sentiment summary
        sentiment = get_sentiment_summary(days=3)
        
        if not sentiment or 'overall' not in sentiment:
            logger.warning("No sentiment data available")
            return {
                'strength': 0,
                'direction': 'neutral',
                'score': 0
            }
        
        # Get overall sentiment score
        score = sentiment['overall'].get('score', 0)
        
        # Determine direction and strength
        if score > 0.2:
            direction = 'bullish'
            strength = min(score * 2, 1)  # Scale to 0-1
        elif score < -0.2:
            direction = 'bearish'
            strength = min(abs(score) * 2, 1)  # Scale to 0-1
        else:
            direction = 'neutral'
            strength = 0
        
        result = {
            'strength': strength,
            'direction': direction,
            'score': score
        }
        
        logger.info(f"Sentiment signals: {direction} (strength: {strength:.2f}, score: {score:.2f})")
        return result
    
    except Exception as e:
        logger.error(f"Error evaluating sentiment signals: {str(e)}")
        return {
            'strength': 0,
            'direction': 'neutral',
            'score': 0
        }

def evaluate_price_prediction(symbol, interval):
    """
    Evaluate price prediction models for trading signals.
    
    Args:
        symbol (str): Trading pair symbol
        interval (str): Kline interval
    
    Returns:
        dict: Price prediction signals
    """
    try:
        logger.info(f"Evaluating price prediction for {symbol} {interval}")
        
        # Get latest predictions
        predictions = get_latest_predictions(symbol, interval)
        
        if not predictions or 'ensemble' not in predictions:
            logger.warning(f"No prediction data available for {symbol} {interval}")
            return {
                'strength': 0,
                'direction': 'neutral',
                'predicted_change': 0
            }
        
        # Get current price and ensemble prediction
        current_price = predictions.get('current_price', 0)
        if current_price == 0:
            logger.warning("Invalid current price")
            return {
                'strength': 0,
                'direction': 'neutral',
                'predicted_change': 0
            }
        
        # Get the last prediction in the ensemble
        ensemble_prediction = predictions['ensemble'][-1]
        
        # Calculate predicted change
        predicted_change = (ensemble_prediction / current_price - 1) * 100  # as percentage
        
        # Determine direction and strength
        if predicted_change > 1:  # More than 1% increase
            direction = 'bullish'
            strength = min(predicted_change / 5, 1)  # Scale to 0-1, max at 5% change
        elif predicted_change < -1:  # More than 1% decrease
            direction = 'bearish'
            strength = min(abs(predicted_change) / 5, 1)  # Scale to 0-1, max at 5% change
        else:
            direction = 'neutral'
            strength = 0
        
        result = {
            'strength': strength,
            'direction': direction,
            'predicted_change': predicted_change
        }
        
        logger.info(f"Price prediction for {symbol} {interval}: {direction} (strength: {strength:.2f}, change: {predicted_change:.2f}%)")
        return result
    
    except Exception as e:
        logger.error(f"Error evaluating price prediction: {str(e)}")
        return {
            'strength': 0,
            'direction': 'neutral',
            'predicted_change': 0
        }

def evaluate_market_structure(symbol, interval):
    """
    Evaluate market structure for trading signals.
    
    Args:
        symbol (str): Trading pair symbol
        interval (str): Kline interval
    
    Returns:
        dict: Market structure signals
    """
    try:
        logger.info(f"Evaluating market structure for {symbol} {interval}")
        
        # Get latest market structure analysis
        analysis = get_latest_analysis(symbol, interval)
        
        if not analysis:
            logger.warning(f"No market structure analysis available for {symbol} {interval}")
            return {
                'strength': 0,
                'direction': 'neutral',
                'signals': []
            }
        
        # Extract key elements
        current_price = analysis.get('current_price', 0)
        if current_price == 0:
            logger.warning("Invalid current price")
            return {
                'strength': 0,
                'direction': 'neutral',
                'signals': []
            }
        
        nearby_support = analysis.get('nearby_support', [])
        nearby_resistance = analysis.get('nearby_resistance', [])
        recent_structures = analysis.get('recent_structures', {})
        
        # Initialize signals
        signals = []
        bullish_count = 0
        bearish_count = 0
        
        # Check distance to nearest support and resistance
        if nearby_support and nearby_resistance:
            nearest_support = nearby_support[0]
            nearest_resistance = nearby_resistance[0]
            
            support_distance = nearest_support['distance']
            resistance_distance = nearest_resistance['distance']
            
            # If price is much closer to support than resistance, bullish signal
            if support_distance < resistance_distance * 0.5:
                signals.append({
                    'name': 'Support Proximity',
                    'description': f"Price near support ({support_distance:.2f}% away)",
                    'direction': 'bullish',
                    'strength': 2
                })
                bullish_count += 2
            
            # If price is much closer to resistance than support, bearish signal
            elif resistance_distance < support_distance * 0.5:
                signals.append({
                    'name': 'Resistance Proximity',
                    'description': f"Price near resistance ({resistance_distance:.2f}% away)",
                    'direction': 'bearish',
                    'strength': 2
                })
                bearish_count += 2
        
        # Check recent structures
        if recent_structures.get('bullish_fvg'):
            signals.append({
                'name': 'Bullish FVG',
                'description': "Recent bullish fair value gap",
                'direction': 'bullish',
                'strength': 1.5
            })
            bullish_count += 1.5
        
        if recent_structures.get('bearish_fvg'):
            signals.append({
                'name': 'Bearish FVG',
                'description': "Recent bearish fair value gap",
                'direction': 'bearish',
                'strength': 1.5
            })
            bearish_count += 1.5
        
        if recent_structures.get('bullish_ob'):
            signals.append({
                'name': 'Bullish OB',
                'description': "Recent bullish order block",
                'direction': 'bullish',
                'strength': 1.5
            })
            bullish_count += 1.5
        
        if recent_structures.get('bearish_ob'):
            signals.append({
                'name': 'Bearish OB',
                'description': "Recent bearish order block",
                'direction': 'bearish',
                'strength': 1.5
            })
            bearish_count += 1.5
        
        # Determine overall direction and strength
        if bullish_count > bearish_count:
            direction = 'bullish'
            strength = min(bullish_count - bearish_count, 10) / 10  # Scale to 0-1
        elif bearish_count > bullish_count:
            direction = 'bearish'
            strength = min(bearish_count - bullish_count, 10) / 10  # Scale to 0-1
        else:
            direction = 'neutral'
            strength = 0
        
        result = {
            'strength': strength,
            'direction': direction,
            'signals': signals
        }
        
        logger.info(f"Market structure signals for {symbol} {interval}: {direction} (strength: {strength:.2f})")
        return result
    
    except Exception as e:
        logger.error(f"Error evaluating market structure: {str(e)}")
        return {
            'strength': 0,
            'direction': 'neutral',
            'signals': []
        }

def evaluate_signal(symbol, interval):
    """
    Evaluate all signals for a trading decision.
    
    Args:
        symbol (str): Trading pair symbol
        interval (str): Kline interval
    
    Returns:
        dict: Combined trading signal
    """
    try:
        logger.info(f"Evaluating combined signal for {symbol} {interval}")
        
        # Get signals from different sources
        technical = evaluate_technical_signals(symbol, interval)
        sentiment = evaluate_sentiment_signals()
        prediction = evaluate_price_prediction(symbol, interval)
        structure = evaluate_market_structure(symbol, interval)
        
        # Weights for different signal types
        weights = {
            'technical': 0.4,
            'sentiment': 0.1,
            'prediction': 0.2,
            'structure': 0.3
        }
        
        # Calculate weighted strengths
        bullish_strength = 0
        bearish_strength = 0
        
        if technical['direction'] == 'bullish':
            bullish_strength += technical['strength'] * weights['technical']
        elif technical['direction'] == 'bearish':
            bearish_strength += technical['strength'] * weights['technical']
        
        if sentiment['direction'] == 'bullish':
            bullish_strength += sentiment['strength'] * weights['sentiment']
        elif sentiment['direction'] == 'bearish':
            bearish_strength += sentiment['strength'] * weights['sentiment']
        
        if prediction['direction'] == 'bullish':
            bullish_strength += prediction['strength'] * weights['prediction']
        elif prediction['direction'] == 'bearish':
            bearish_strength += prediction['strength'] * weights['prediction']
        
        if structure['direction'] == 'bullish':
            bullish_strength += structure['strength'] * weights['structure']
        elif structure['direction'] == 'bearish':
            bearish_strength += structure['strength'] * weights['structure']
        
        # Determine overall direction and strength
        if bullish_strength > bearish_strength:
            direction = 'bullish'
            strength = bullish_strength - bearish_strength
        elif bearish_strength > bullish_strength:
            direction = 'bearish'
            strength = bearish_strength - bullish_strength
        else:
            direction = 'neutral'
            strength = 0
        
        # Determine signal quality
        if strength > 0.5:
            quality = 'strong'
        elif strength > 0.2:
            quality = 'moderate'
        else:
            quality = 'weak'
        
        # Determine action
        config = load_config()
        trading_enabled = config.get('trading', {}).get('enabled', False)
        
        if trading_enabled and quality in ['strong', 'moderate']:
            action = 'buy' if direction == 'bullish' else 'sell' if direction == 'bearish' else 'hold'
        else:
            action = 'hold'
        
        # Prepare result
        result = {
            'symbol': symbol,
            'interval': interval,
            'timestamp': datetime.now().isoformat(),
            'direction': direction,
            'strength': strength,
            'quality': quality,
            'action': action,
            'components': {
                'technical': technical,
                'sentiment': sentiment,
                'prediction': prediction,
                'structure': structure
            }
        }
        
        # Store result in database
        db_path = os.path.join(os.path.dirname(__file__), '../../instance/finance.db')
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trading_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                interval TEXT,
                direction TEXT,
                strength REAL,
                quality TEXT,
                action TEXT,
                data TEXT,
                created_at TEXT
            )
        ''')
        
        # Insert signal
        cursor.execute('''
            INSERT INTO trading_signals
            (symbol, interval, direction, strength, quality, action, data, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            symbol.lower(),
            interval,
            direction,
            strength,
            quality,
            action,
            json.dumps(result),
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Combined signal for {symbol} {interval}: {direction} (strength: {strength:.2f}, quality: {quality}, action: {action})")
        return result
    
    except Exception as e:
        logger.error(f"Error evaluating combined signal: {str(e)}")
        return {
            'symbol': symbol,
            'interval': interval,
            'timestamp': datetime.now().isoformat(),
            'direction': 'neutral',
            'strength': 0,
            'quality': 'weak',
            'action': 'hold',
            'components': {}
        }

def check_strategy_conditions(strategy):
    """
    Check if a strategy's conditions are met.
    
    Args:
        strategy (dict): Strategy configuration
    
    Returns:
        tuple: (bool, dict) - Whether conditions are met and signal details
    """
    try:
        logger.info(f"Checking conditions for strategy: {strategy.get('name', 'Unknown')}")
        
        symbol = strategy.get('symbol')
        interval = strategy.get('interval')
        
        if not symbol or not interval:
            logger.warning("Strategy missing symbol or interval")
            return False, {}
        
        # Get combined signal
        signal = evaluate_signal(symbol, interval)
        
        # Check strategy conditions
        conditions = strategy.get('conditions', {})
        
        # Direction condition
        direction_condition = conditions.get('direction')
        if direction_condition and signal['direction'] != direction_condition:
            return False, signal
        
        # Strength condition
        min_strength = conditions.get('min_strength', 0)
        if signal['strength'] < min_strength:
            return False, signal
        
        # Quality condition
        quality_condition = conditions.get('quality')
        if quality_condition and signal['quality'] != quality_condition:
            return False, signal
        
        # Check for specific component conditions
        components = conditions.get('components', {})
        
        for component, requirements in components.items():
            if component not in signal['components']:
                return False, signal
            
            comp_direction = requirements.get('direction')
            if comp_direction and signal['components'][component]['direction'] != comp_direction:
                return False, signal
            
            comp_min_strength = requirements.get('min_strength', 0)
            if signal['components'][component]['strength'] < comp_min_strength:
                return False, signal
        
        # All conditions met
        return True, signal
    
    except Exception as e:
        logger.error(f"Error checking strategy conditions: {str(e)}")
        return False, {}

def get_active_strategies():
    """
    Get all active trading strategies from the database.
    
    Returns:
        list: Active strategies
    """
    try:
        # Get database connection
        db_path = os.path.join(os.path.dirname(__file__), '../../instance/finance.db')
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Query active strategies
        cursor.execute('''
            SELECT id, name, config, user_id FROM strategy
            WHERE is_active = 1
        ''')
        
        strategies = []
        for row in cursor.fetchall():
            strategy_id, name, config_json, user_id = row
            
            try:
                config = json.loads(config_json)
                config['id'] = strategy_id
                config['name'] = name
                config['user_id'] = user_id
                strategies.append(config)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON in strategy {strategy_id}")
        
        conn.close()
        
        logger.info(f"Found {len(strategies)} active strategies")
        return strategies
    
    except Exception as e:
        logger.error(f"Error getting active strategies: {str(e)}")
        return []

def check_all_strategies():
    """
    Check all active strategies for trading signals.
    
    Returns:
        list: Trading signals to execute
    """
    try:
        logger.info("Checking all active strategies")
        
        # Get active strategies
        strategies = get_active_strategies()
        
        # Check each strategy
        signals = []
        
        for strategy in strategies:
            conditions_met, signal = check_strategy_conditions(strategy)
            
            if conditions_met:
                logger.info(f"Strategy '{strategy.get('name')}' conditions met")
                
                # Create trading signal
                trade_signal = {
                    'strategy_id': strategy.get('id'),
                    'user_id': strategy.get('user_id'),
                    'symbol': strategy.get('symbol'),
                    'side': 'BUY' if signal['direction'] == 'bullish' else 'SELL',
                    'entry_type': 'MARKET',
                    'quantity': None,  # Will be calculated by executor
                    'timestamp': datetime.now().isoformat(),
                    'signal_data': signal
                }
                
                signals.append(trade_signal)
            else:
                logger.info(f"Strategy '{strategy.get('name')}' conditions not met")
        
        logger.info(f"Generated {len(signals)} trading signals")
        return signals
    
    except Exception as e:
        logger.error(f"Error checking all strategies: {str(e)}")
        return []

def get_open_trades(user_id=None):
    """
    Get all open trades.
    
    Args:
        user_id (int, optional): Filter by user ID
    
    Returns:
        list: Open trades
    """
    try:
        # Get database connection
        db_path = os.path.join(os.path.dirname(__file__), '../../instance/finance.db')
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        cursor = conn.cursor()
        
        # Query open trades
        if user_id:
            cursor.execute('''
                SELECT * FROM trade
                WHERE status = 'OPEN' AND user_id = ?
                ORDER BY entry_time DESC
            ''', (user_id,))
        else:
            cursor.execute('''
                SELECT * FROM trade
                WHERE status = 'OPEN'
                ORDER BY entry_time DESC
            ''')
        
        trades = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        
        logger.info(f"Found {len(trades)} open trades")
        return trades
    
    except Exception as e:
        logger.error(f"Error getting open trades: {str(e)}")
        return []

def get_trade_history(user_id=None, limit=100):
    """
    Get trade history.
    
    Args:
        user_id (int, optional): Filter by user ID
        limit (int): Maximum number of trades to return
    
    Returns:
        list: Trade history
    """
    try:
        # Get database connection
        db_path = os.path.join(os.path.dirname(__file__), '../../instance/finance.db')
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        cursor = conn.cursor()
        
        # Query trades
        if user_id:
            cursor.execute('''
                SELECT * FROM trade
                WHERE user_id = ?
                ORDER BY entry_time DESC
                LIMIT ?
            ''', (user_id, limit))
        else:
            cursor.execute('''
                SELECT * FROM trade
                ORDER BY entry_time DESC
                LIMIT ?
            ''', (limit,))
        
        trades = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        
        logger.info(f"Found {len(trades)} trades")
        return trades
    
    except Exception as e:
        logger.error(f"Error getting trade history: {str(e)}")
        return []

def get_performance_metrics(user_id=None):
    """
    Calculate performance metrics from trade history.
    
    Args:
        user_id (int, optional): Filter by user ID
    
    Returns:
        dict: Performance metrics
    """
    try:
        # Get trade history
        trades = get_trade_history(user_id=user_id, limit=1000)
        
        # Filter closed trades
        closed_trades = [t for t in trades if t['status'] == 'CLOSED' and t['pnl'] is not None]
        
        if not closed_trades:
            logger.warning("No closed trades found for performance calculation")
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_pnl': 0,
                'max_profit': 0,
                'max_loss': 0,
                'profit_factor': 0
            }
        
        # Calculate metrics
        total_trades = len(closed_trades)
        winning_trades = sum(1 for t in closed_trades if t['pnl'] > 0)
        losing_trades = sum(1 for t in closed_trades if t['pnl'] < 0)
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_pnl = sum(t['pnl'] for t in closed_trades)
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
        
        max_profit = max((t['pnl'] for t in closed_trades), default=0)
        max_loss = min((t['pnl'] for t in closed_trades), default=0)
        
        total_profit = sum(t['pnl'] for t in closed_trades if t['pnl'] > 0)
        total_loss = abs(sum(t['pnl'] for t in closed_trades if t['pnl'] < 0))
        
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        metrics = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'profit_factor': profit_factor
        }
        
        logger.info(f"Calculated performance metrics: {metrics}")
        return metrics
    
    except Exception as e:
        logger.error(f"Error calculating performance metrics: {str(e)}")
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'avg_pnl': 0,
            'max_profit': 0,
            'max_loss': 0,
            'profit_factor': 0
        }