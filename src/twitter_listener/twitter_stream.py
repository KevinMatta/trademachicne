"""
Module for listening to Twitter streams for financial signals.
"""
import logging
import os
import yaml
import json
import requests
import time
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
                'twitter': {
                    'bearer_token': '',
                    'api_key': '',
                    'api_secret': ''
                }
            },
            'analysis': {
                'sentiment': {
                    'keywords': ['bitcoin', 'crypto', 'blockchain']
                }
            }
        }

def get_twitter_headers():
    """Get headers for Twitter API requests."""
    config = load_config()
    bearer_token = config.get('apis', {}).get('twitter', {}).get('bearer_token', '')
    
    if not bearer_token:
        logger.warning("Twitter bearer token not configured")
        return {}
    
    return {
        'Authorization': f'Bearer {bearer_token}',
        'Content-Type': 'application/json'
    }

def create_twitter_rule(keywords):
    """
    Create a Twitter stream rule for the specified keywords.
    
    Args:
        keywords (list): List of keywords to track
    
    Returns:
        dict: Response from Twitter API
    """
    headers = get_twitter_headers()
    if not headers:
        return {'error': 'Twitter API not configured'}
    
    # Format keywords into a rule
    rule_value = ' OR '.join([f'"{kw}"' for kw in keywords])
    
    rule = {
        'add': [
            {'value': rule_value, 'tag': 'financial-keywords'}
        ]
    }
    
    response = requests.post(
        'https://api.twitter.com/2/tweets/search/stream/rules',
        headers=headers,
        json=rule
    )
    
    if response.status_code != 201:
        logger.error(f"Failed to create Twitter rule: {response.text}")
        return {'error': response.text}
    
    return response.json()

def delete_all_rules():
    """
    Delete all existing Twitter stream rules.
    
    Returns:
        dict: Response from Twitter API
    """
    headers = get_twitter_headers()
    if not headers:
        return {'error': 'Twitter API not configured'}
    
    # Get existing rules
    response = requests.get(
        'https://api.twitter.com/2/tweets/search/stream/rules',
        headers=headers
    )
    
    if response.status_code != 200:
        logger.error(f"Failed to get Twitter rules: {response.text}")
        return {'error': response.text}
    
    rules = response.json()
    
    if not rules.get('data'):
        return {'message': 'No rules to delete'}
    
    # Format rules for deletion
    ids = [rule['id'] for rule in rules['data']]
    payload = {'delete': {'ids': ids}}
    
    # Delete rules
    response = requests.post(
        'https://api.twitter.com/2/tweets/search/stream/rules',
        headers=headers,
        json=payload
    )
    
    if response.status_code != 200:
        logger.error(f"Failed to delete Twitter rules: {response.text}")
        return {'error': response.text}
    
    return response.json()

def store_tweet(tweet):
    """
    Store a tweet in the database.
    
    Args:
        tweet (dict): Tweet data
    """
    try:
        # Get database connection
        db_path = os.path.join(os.path.dirname(__file__), '../../instance/finance.db')
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tweets (
                id TEXT PRIMARY KEY,
                text TEXT,
                author_id TEXT,
                created_at TEXT,
                raw_data TEXT,
                sentiment TEXT,
                sentiment_score REAL,
                processed_at TEXT,
                fetched_at TEXT
            )
        ''')
        
        # Extract tweet data
        tweet_id = tweet.get('data', {}).get('id')
        text = tweet.get('data', {}).get('text', '')
        author_id = tweet.get('data', {}).get('author_id', '')
        created_at = tweet.get('data', {}).get('created_at', datetime.now().isoformat())
        
        # Insert tweet
        cursor.execute('''
            INSERT OR IGNORE INTO tweets 
            (id, text, author_id, created_at, raw_data, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            tweet_id,
            text,
            author_id,
            created_at,
            json.dumps(tweet),
            datetime.now().isoformat()
        ))
        
        conn.commit()
        
    except Exception as e:
        logger.error(f"Error storing tweet: {str(e)}")
        if 'conn' in locals():
            conn.rollback()
    finally:
        if 'conn' in locals():
            conn.close()

def stream_tweets(duration=300):
    """
    Stream tweets for a specified duration.
    
    Args:
        duration (int): Duration to stream in seconds
    
    Returns:
        int: Number of tweets collected
    """
    headers = get_twitter_headers()
    if not headers:
        logger.error("Twitter API not configured")
        return 0
    
    logger.info(f"Starting Twitter stream for {duration} seconds")
    
    # Parameters for the stream
    params = {
        'tweet.fields': 'created_at,author_id,public_metrics,entities',
        'expansions': 'author_id',
        'user.fields': 'name,username,description,public_metrics'
    }
    
    tweet_count = 0
    start_time = time.time()
    
    try:
        # Start the stream
        response = requests.get(
            'https://api.twitter.com/2/tweets/search/stream',
            headers=headers,
            params=params,
            stream=True
        )
        
        if response.status_code != 200:
            logger.error(f"Failed to start Twitter stream: {response.text}")
            return 0
        
        # Process tweets
        for line in response.iter_lines():
            if line:
                try:
                    tweet = json.loads(line)
                    store_tweet(tweet)
                    tweet_count += 1
                    logger.debug(f"Collected tweet: {tweet.get('data', {}).get('text', '')[:50]}...")
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse tweet: {line}")
            
            # Check duration
            if time.time() - start_time > duration:
                break
    
    except Exception as e:
        logger.error(f"Error in Twitter stream: {str(e)}")
    
    logger.info(f"Twitter stream completed. Collected {tweet_count} tweets.")
    return tweet_count

def get_recent_tweets(keywords, count=100):
    """
    Get recent tweets for the specified keywords using the recent search API.
    
    Args:
        keywords (list): List of keywords to search for
        count (int): Number of tweets to retrieve
    
    Returns:
        list: List of tweets
    """
    headers = get_twitter_headers()
    if not headers:
        logger.error("Twitter API not configured")
        return []
    
    logger.info(f"Searching for recent tweets with keywords: {keywords}")
    
    # Format keywords into a query
    query = ' OR '.join([f'"{kw}"' for kw in keywords])
    
    # Parameters for the search
    params = {
        'query': query,
        'max_results': min(count, 100),  # Max allowed is 100
        'tweet.fields': 'created_at,author_id,public_metrics,entities',
        'expansions': 'author_id',
        'user.fields': 'name,username,description,public_metrics'
    }
    
    try:
        response = requests.get(
            'https://api.twitter.com/2/tweets/search/recent',
            headers=headers,
            params=params
        )
        
        if response.status_code != 200:
            logger.error(f"Failed to search Twitter: {response.text}")
            return []
        
        data = response.json()
        tweets = data.get('data', [])
        
        # Store tweets
        for tweet_data in tweets:
            tweet = {
                'data': tweet_data,
                'includes': data.get('includes', {})
            }
            store_tweet(tweet)
        
        logger.info(f"Retrieved and stored {len(tweets)} tweets")
        return tweets
    
    except Exception as e:
        logger.error(f"Error searching Twitter: {str(e)}")
        return []

def update_twitter_data():
    """Update Twitter data in the database."""
    try:
        config = load_config()
        keywords = config.get('analysis', {}).get('sentiment', {}).get('keywords', ['bitcoin'])
        
        # Get recent tweets
        tweets = get_recent_tweets(keywords)
        
        logger.info(f"Updated Twitter data with {len(tweets)} tweets")
    except Exception as e:
        logger.error(f"Error updating Twitter data: {str(e)}")