#!/usr/bin/env python3
"""
Main entry point for the financial analysis system.
"""
import os
import sys
import logging
from datetime import datetime
import threading
import time
import argparse

# Add project directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(
            os.path.dirname(__file__), 
            f'../logs/app_{datetime.now().strftime("%Y%m%d")}.log'
        ))
    ]
)
logger = logging.getLogger(__name__)

# Create logs directory if it doesn't exist
os.makedirs(os.path.join(os.path.dirname(__file__), '../logs'), exist_ok=True)

from src.app import app
from src.scheduler import start_scheduler, stop_scheduler

def parse_args():
    parser = argparse.ArgumentParser(description='Financial Analysis System')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the server on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--no-scheduler', action='store_true', help='Do not start the scheduler')
    return parser.parse_args()

def main():
    args = parse_args()
    
    logger.info("Starting Financial Analysis System")
    
    # Start scheduler in a separate thread if not disabled
    if not args.no_scheduler:
        scheduler_thread = threading.Thread(target=start_scheduler)
        scheduler_thread.daemon = True
        scheduler_thread.start()
        logger.info("Scheduler started")
    
    try:
        # Start Flask application
        logger.info(f"Starting Flask application on {args.host}:{args.port}")
        app.run(host=args.host, port=args.port, debug=args.debug)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        stop_scheduler()
    except Exception as e:
        logger.error(f"Error running application: {str(e)}")
        stop_scheduler()
        sys.exit(1)

if __name__ == "__main__":
    main()