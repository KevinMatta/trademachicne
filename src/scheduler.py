"""
Scheduler module for running periodic tasks.
"""
import threading
import time
import logging
import yaml
import os
import schedule
from datetime import datetime

logger = logging.getLogger(__name__)

# Flag to control scheduler running state
scheduler_running = False

def load_config():
    """Load configuration from config.yaml file."""
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    else:
        logger.warning("Config file not found. Using default configuration.")
        return {
            'scheduler': {
                'data_collection_interval': 300,
                'analysis_interval': 900,
                'trading_interval': 60,
                'monitoring_interval': 300
            }
        }

def collect_market_data():
    """Collect market data from configured sources."""
    from src.data_ingest.binance_rest import update_market_data
    
    logger.info("Running scheduled market data collection")
    try:
        config = load_config()
        symbols = config.get('data_collection', {}).get('symbols', ['BTCUSDT'])
        intervals = config.get('data_collection', {}).get('intervals', ['1h'])
        
        for symbol in symbols:
            for interval in intervals:
                update_market_data(symbol, interval)
                
        logger.info("Market data collection completed")
    except Exception as e:
        logger.error(f"Error in market data collection: {str(e)}")

def collect_news_data():
    """Collect news data from configured sources."""
    from src.news_scraper.bloomberg_scraper import scrape_bloomberg
    from src.news_scraper.reuters_scraper import scrape_reuters
    
    logger.info("Running scheduled news data collection")
    try:
        config = load_config()
        keywords = config.get('analysis', {}).get('sentiment', {}).get('keywords', ['bitcoin'])
        
        # Scrape news from Bloomberg
        bloomberg_articles = scrape_bloomberg(keywords)
        logger.info(f"Collected {len(bloomberg_articles)} articles from Bloomberg")
        
        # Scrape news from Reuters
        reuters_articles = scrape_reuters(keywords)
        logger.info(f"Collected {len(reuters_articles)} articles from Reuters")
        
        logger.info("News data collection completed")
    except Exception as e:
        logger.error(f"Error in news data collection: {str(e)}")

def run_analysis():
    """Run analysis on collected data."""
    from src.analysis.forecasting_model import update_forecasts
    from src.analysis.nlp_sentiment import analyze_news_sentiment
    
    logger.info("Running scheduled analysis")
    try:
        # Update technical indicators
        from src.preprocessing.indicators import update_indicators
        update_indicators()
        
        # Update forecasts
        update_forecasts()
        
        # Analyze news sentiment
        analyze_news_sentiment()
        
        logger.info("Analysis completed")
    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}")

def check_trading_signals():
    """Check for trading signals and execute trades if enabled."""
    from src.decision_engine.engine import check_all_strategies
    
    logger.info("Running scheduled trading signal check")
    try:
        config = load_config()
        trading_enabled = config.get('trading', {}).get('enabled', False)
        
        if trading_enabled:
            signals = check_all_strategies()
            if signals:
                from src.executor.binance_executor import execute_signals
                execute_signals(signals)
        else:
            logger.info("Trading is disabled. Skipping execution.")
        
        logger.info("Trading signal check completed")
    except Exception as e:
        logger.error(f"Error in trading signal check: {str(e)}")

def run_monitoring():
    """Run monitoring and reporting tasks."""
    from src.monitoring.metrics import update_metrics
    
    logger.info("Running scheduled monitoring")
    try:
        update_metrics()
        logger.info("Monitoring completed")
    except Exception as e:
        logger.error(f"Error in monitoring: {str(e)}")

def schedule_jobs():
    """Schedule all periodic jobs."""
    config = load_config()
    
    # Schedule data collection
    data_interval = config.get('scheduler', {}).get('data_collection_interval', 300)
    schedule.every(data_interval).seconds.do(collect_market_data)
    
    # Schedule news collection (every 3 hours by default)
    schedule.every(3).hours.do(collect_news_data)
    
    # Schedule analysis
    analysis_interval = config.get('scheduler', {}).get('analysis_interval', 900)
    schedule.every(analysis_interval).seconds.do(run_analysis)
    
    # Schedule trading signal check
    trading_interval = config.get('scheduler', {}).get('trading_interval', 60)
    schedule.every(trading_interval).seconds.do(check_trading_signals)
    
    # Schedule monitoring
    monitoring_interval = config.get('scheduler', {}).get('monitoring_interval', 300)
    schedule.every(monitoring_interval).seconds.do(run_monitoring)
    
    # Schedule daily report at midnight
    schedule.every().day.at("00:00").do(lambda: logger.info("Daily report would be generated here"))

def scheduler_loop():
    """Main scheduler loop that runs continuously."""
    global scheduler_running
    
    # Schedule all jobs
    schedule_jobs()
    
    logger.info("Scheduler started")
    
    # Run the scheduler loop
    while scheduler_running:
        schedule.run_pending()
        time.sleep(1)
    
    logger.info("Scheduler stopped")

def start_scheduler():
    """Start the scheduler in a separate thread."""
    global scheduler_running
    
    if scheduler_running:
        logger.warning("Scheduler is already running")
        return
    
    scheduler_running = True
    scheduler_thread = threading.Thread(target=scheduler_loop)
    scheduler_thread.daemon = True
    scheduler_thread.start()
    
    return scheduler_thread

def stop_scheduler():
    """Stop the scheduler."""
    global scheduler_running
    
    if not scheduler_running:
        logger.warning("Scheduler is not running")
        return
    
    logger.info("Stopping scheduler")
    scheduler_running = False