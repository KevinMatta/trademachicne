"""
Module for monitoring and tracking system metrics.
"""
import logging
import os
import json
import sqlite3
from datetime import datetime, timedelta
import pandas as pd

from ..decision_engine.engine import get_performance_metrics

logger = logging.getLogger(__name__)

def update_metrics():
    """Update and store system metrics."""
    try:
        logger.info("Updating system metrics")
        
        # Get database connection
        db_path = os.path.join(os.path.dirname(__file__), '../../instance/finance.db')
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create metrics table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_type TEXT,
                data TEXT,
                created_at TEXT
            )
        ''')
        
        # Update performance metrics
        update_performance_metrics(conn, cursor)
        
        # Update data collection metrics
        update_data_collection_metrics(conn, cursor)
        
        # Update system status metrics
        update_system_status_metrics(conn, cursor)
        
        conn.close()
        logger.info("System metrics updated")
    
    except Exception as e:
        logger.error(f"Error updating metrics: {str(e)}")

def update_performance_metrics(conn, cursor):
    """
    Update and store trading performance metrics.
    
    Args:
        conn: Database connection
        cursor: Database cursor
    """
    try:
        # Get overall performance metrics
        metrics = get_performance_metrics()
        
        # Store metrics
        cursor.execute('''
            INSERT INTO system_metrics (metric_type, data, created_at)
            VALUES (?, ?, ?)
        ''', (
            'performance',
            json.dumps(metrics),
            datetime.now().isoformat()
        ))
        
        # Get per-user metrics
        cursor.execute('SELECT id FROM user')
        users = cursor.fetchall()
        
        for (user_id,) in users:
            user_metrics = get_performance_metrics(user_id=user_id)
            
            cursor.execute('''
                INSERT INTO system_metrics (metric_type, data, created_at)
                VALUES (?, ?, ?)
            ''', (
                f'performance_user_{user_id}',
                json.dumps(user_metrics),
                datetime.now().isoformat()
            ))
        
        conn.commit()
        logger.info("Performance metrics updated")
    
    except Exception as e:
        logger.error(f"Error updating performance metrics: {str(e)}")

def update_data_collection_metrics(conn, cursor):
    """
    Update and store data collection metrics.
    
    Args:
        conn: Database connection
        cursor: Database cursor
    """
    try:
        metrics = {
            'market_data': {},
            'news': {},
            'social': {}
        }
        
        # Get market data metrics
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'klines_%'")
        klines_tables = cursor.fetchall()
        
        for (table_name,) in klines_tables:
            # Extract symbol and interval from table name
            parts = table_name.split('_')
            if len(parts) < 3:
                continue
            
            symbol = parts[1].upper()
            interval = parts[2]
            
            # Count rows
            cursor.execute(f'SELECT COUNT(*) FROM {table_name}')
            count = cursor.fetchone()[0]
            
            # Get oldest and newest timestamps
            cursor.execute(f'SELECT MIN(timestamp), MAX(timestamp) FROM {table_name}')
            oldest, newest = cursor.fetchone()
            
            # Store metrics
            if symbol not in metrics['market_data']:
                metrics['market_data'][symbol] = {}
            
            metrics['market_data'][symbol][interval] = {
                'count': count,
                'oldest': oldest,
                'newest': newest
            }
        
        # Get news metrics
        cursor.execute('SELECT COUNT(*) FROM news_articles')
        news_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM news_articles WHERE processed_at IS NOT NULL')
        processed_count = cursor.fetchone()[0]
        
        metrics['news'] = {
            'total': news_count,
            'processed': processed_count,
            'unprocessed': news_count - processed_count
        }
        
        # Get social metrics
        cursor.execute('SELECT COUNT(*) FROM tweets')
        tweets_count = cursor.fetchone()[0] if cursor.rowcount > 0 else 0
        
        cursor.execute('SELECT COUNT(*) FROM tweets WHERE processed_at IS NOT NULL')
        processed_tweets = cursor.fetchone()[0] if cursor.rowcount > 0 else 0
        
        metrics['social'] = {
            'total': tweets_count,
            'processed': processed_tweets,
            'unprocessed': tweets_count - processed_tweets
        }
        
        # Store metrics
        cursor.execute('''
            INSERT INTO system_metrics (metric_type, data, created_at)
            VALUES (?, ?, ?)
        ''', (
            'data_collection',
            json.dumps(metrics),
            datetime.now().isoformat()
        ))
        
        conn.commit()
        logger.info("Data collection metrics updated")
    
    except Exception as e:
        logger.error(f"Error updating data collection metrics: {str(e)}")

def update_system_status_metrics(conn, cursor):
    """
    Update and store system status metrics.
    
    Args:
        conn: Database connection
        cursor: Database cursor
    """
    try:
        metrics = {
            'users': {},
            'strategies': {},
            'trades': {},
            'predictions': {},
            'system': {}
        }
        
        # Get user metrics
        cursor.execute('SELECT COUNT(*) FROM user')
        user_count = cursor.fetchone()[0]
        
        metrics['users'] = {
            'total': user_count
        }
        
        # Get strategy metrics
        cursor.execute('SELECT COUNT(*) FROM strategy')
        strategy_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM strategy WHERE is_active = 1')
        active_strategy_count = cursor.fetchone()[0]
        
        metrics['strategies'] = {
            'total': strategy_count,
            'active': active_strategy_count,
            'inactive': strategy_count - active_strategy_count
        }
        
        # Get trade metrics
        cursor.execute('SELECT COUNT(*) FROM trade')
        trade_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM trade WHERE status = "OPEN"')
        open_trade_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM trade WHERE status = "CLOSED"')
        closed_trade_count = cursor.fetchone()[0]
        
        metrics['trades'] = {
            'total': trade_count,
            'open': open_trade_count,
            'closed': closed_trade_count
        }
        
        # Get prediction metrics
        cursor.execute('SELECT COUNT(*) FROM predictions')
        prediction_count = cursor.fetchone()[0] if cursor.rowcount > 0 else 0
        
        # Get last 24 hours predictions
        yesterday = (datetime.now() - timedelta(days=1)).isoformat()
        cursor.execute('SELECT COUNT(*) FROM predictions WHERE created_at > ?', (yesterday,))
        recent_prediction_count = cursor.fetchone()[0] if cursor.rowcount > 0 else 0
        
        metrics['predictions'] = {
            'total': prediction_count,
            'last_24h': recent_prediction_count
        }
        
        # System metrics
        metrics['system'] = {
            'uptime': get_uptime(),
            'version': '0.1.0',
            'last_updated': datetime.now().isoformat()
        }
        
        # Store metrics
        cursor.execute('''
            INSERT INTO system_metrics (metric_type, data, created_at)
            VALUES (?, ?, ?)
        ''', (
            'system_status',
            json.dumps(metrics),
            datetime.now().isoformat()
        ))
        
        conn.commit()
        logger.info("System status metrics updated")
    
    except Exception as e:
        logger.error(f"Error updating system status metrics: {str(e)}")

def get_uptime():
    """
    Get system uptime.
    
    Returns:
        str: Uptime in human-readable format
    """
    try:
        # Read uptime file
        with open('/proc/uptime', 'r') as f:
            uptime_seconds = float(f.readline().split()[0])
        
        # Convert to human-readable format
        uptime = str(timedelta(seconds=uptime_seconds))
        return uptime
    
    except Exception as e:
        logger.error(f"Error getting uptime: {str(e)}")
        return "Unknown"

def get_latest_metrics(metric_type):
    """
    Get the latest metrics of a specific type.
    
    Args:
        metric_type (str): Type of metrics to retrieve
    
    Returns:
        dict: Latest metrics
    """
    try:
        # Get database connection
        db_path = os.path.join(os.path.dirname(__file__), '../../instance/finance.db')
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT data FROM system_metrics
            WHERE metric_type = ?
            ORDER BY created_at DESC
            LIMIT 1
        ''', (metric_type,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return json.loads(result[0])
        else:
            logger.warning(f"No metrics found for type {metric_type}")
            return {}
    
    except Exception as e:
        logger.error(f"Error getting latest metrics: {str(e)}")
        return {}

def get_metrics_history(metric_type, days=7):
    """
    Get historical metrics of a specific type.
    
    Args:
        metric_type (str): Type of metrics to retrieve
        days (int): Number of days of history to retrieve
    
    Returns:
        list: Historical metrics
    """
    try:
        # Get database connection
        db_path = os.path.join(os.path.dirname(__file__), '../../instance/finance.db')
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Calculate date range
        start_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        cursor.execute('''
            SELECT data, created_at FROM system_metrics
            WHERE metric_type = ? AND created_at > ?
            ORDER BY created_at
        ''', (metric_type, start_date))
        
        results = cursor.fetchall()
        conn.close()
        
        if results:
            history = []
            for data, timestamp in results:
                history.append({
                    'timestamp': timestamp,
                    'data': json.loads(data)
                })
            return history
        else:
            logger.warning(f"No historical metrics found for type {metric_type}")
            return []
    
    except Exception as e:
        logger.error(f"Error getting metrics history: {str(e)}")
        return []

def generate_performance_report(user_id=None):
    """
    Generate a comprehensive performance report.
    
    Args:
        user_id (int, optional): User ID for user-specific report
    
    Returns:
        dict: Performance report
    """
    try:
        logger.info(f"Generating performance report for user {user_id if user_id else 'all'}")
        
        # Get performance metrics
        metrics = get_performance_metrics(user_id=user_id)
        
        # Get historical metrics
        metric_type = f"performance_user_{user_id}" if user_id else "performance"
        history = get_metrics_history(metric_type, days=30)
        
        # Extract performance trends
        win_rates = []
        pnls = []
        timestamps = []
        
        for entry in history:
            data = entry['data']
            timestamps.append(entry['timestamp'])
            win_rates.append(data.get('win_rate', 0))
            pnls.append(data.get('total_pnl', 0))
        
        # Create report
        report = {
            'current_metrics': metrics,
            'history': {
                'timestamps': timestamps,
                'win_rates': win_rates,
                'pnls': pnls
            },
            'generated_at': datetime.now().isoformat()
        }
        
        # Calculate additional insights
        if len(pnls) > 1:
            # Calculate daily PnL change
            daily_changes = []
            for i in range(1, len(pnls)):
                daily_changes.append(pnls[i] - pnls[i-1])
            
            report['insights'] = {
                'avg_daily_pnl': sum(daily_changes) / len(daily_changes) if daily_changes else 0,
                'pnl_trend': 'increasing' if pnls[-1] > pnls[0] else 'decreasing',
                'win_rate_trend': 'improving' if win_rates[-1] > win_rates[0] else 'declining'
            }
        
        logger.info("Performance report generated")
        return report
    
    except Exception as e:
        logger.error(f"Error generating performance report: {str(e)}")
        return {}