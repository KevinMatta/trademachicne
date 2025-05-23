"""
Module for NLP-based sentiment analysis of financial news and social media.
"""
import logging
import os
import yaml
import json
import sqlite3
import pandas as pd
import requests
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
                'openai': {
                    'api_key': ''
                }
            },
            'analysis': {
                'sentiment': {
                    'model': 'text-davinci-003'
                }
            }
        }

def analyze_sentiment_openai(text):
    """
    Analyze sentiment of text using OpenAI API.
    
    Args:
        text (str): Text to analyze
    
    Returns:
        str: Sentiment ("positive", "negative", or "neutral")
    """
    config = load_config()
    api_key = config.get('apis', {}).get('openai', {}).get('api_key', '')
    model = config.get('analysis', {}).get('sentiment', {}).get('model', 'text-davinci-003')
    
    if not api_key:
        logger.warning("OpenAI API key not configured")
        return "neutral"  # Default fallback
    
    try:
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        # Prepare prompt for OpenAI
        prompt = f"""Analyze the sentiment of the following financial text and classify it as positive, negative, or neutral. 
        Respond with just one word: positive, negative, or neutral.
        
        Text: "{text}"
        
        Sentiment:"""
        
        payload = {
            'model': model,
            'prompt': prompt,
            'max_tokens': 10,
            'temperature': 0.0
        }
        
        response = requests.post(
            'https://api.openai.com/v1/completions',
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            logger.error(f"OpenAI API error: {response.text}")
            return "neutral"  # Default fallback
        
        result = response.json()
        sentiment_text = result['choices'][0]['text'].strip().lower()
        
        # Normalize the response
        if 'positive' in sentiment_text:
            return 'positive'
        elif 'negative' in sentiment_text:
            return 'negative'
        else:
            return 'neutral'
    
    except Exception as e:
        logger.error(f"Error analyzing sentiment with OpenAI: {str(e)}")
        return "neutral"  # Default fallback

def analyze_news_sentiment():
    """Analyze sentiment of unprocessed news articles in the database."""
    try:
        # Get database connection
        db_path = os.path.join(os.path.dirname(__file__), '../../instance/finance.db')
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get unprocessed articles
        cursor.execute('''
            SELECT id, title, summary
            FROM news_articles
            WHERE processed_at IS NULL
            LIMIT 50  # Process in batches to avoid rate limits
        ''')
        
        articles = cursor.fetchall()
        
        if not articles:
            logger.info("No unprocessed news articles found")
            return
        
        logger.info(f"Analyzing sentiment for {len(articles)} news articles")
        
        for article_id, title, summary in articles:
            # Combine title and summary for analysis
            text = f"{title}. {summary}"
            
            # Analyze sentiment
            sentiment = analyze_sentiment_openai(text)
            
            # Determine sentiment score
            sentiment_score = 1.0 if sentiment == 'positive' else (-1.0 if sentiment == 'negative' else 0.0)
            
            # Update database
            cursor.execute('''
                UPDATE news_articles
                SET sentiment = ?, sentiment_score = ?, processed_at = ?
                WHERE id = ?
            ''', (sentiment, sentiment_score, datetime.now().isoformat(), article_id))
        
        conn.commit()
        logger.info(f"Analyzed sentiment for {len(articles)} news articles")
    
    except Exception as e:
        logger.error(f"Error analyzing news sentiment: {str(e)}")
        if 'conn' in locals():
            conn.rollback()
    finally:
        if 'conn' in locals():
            conn.close()

def analyze_tweets_sentiment():
    """Analyze sentiment of unprocessed tweets in the database."""
    try:
        # Get database connection
        db_path = os.path.join(os.path.dirname(__file__), '../../instance/finance.db')
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get unprocessed tweets
        cursor.execute('''
            SELECT id, text
            FROM tweets
            WHERE processed_at IS NULL
            LIMIT 50  # Process in batches to avoid rate limits
        ''')
        
        tweets = cursor.fetchall()
        
        if not tweets:
            logger.info("No unprocessed tweets found")
            return
        
        logger.info(f"Analyzing sentiment for {len(tweets)} tweets")
        
        for tweet_id, text in tweets:
            # Analyze sentiment
            sentiment = analyze_sentiment_openai(text)
            
            # Determine sentiment score
            sentiment_score = 1.0 if sentiment == 'positive' else (-1.0 if sentiment == 'negative' else 0.0)
            
            # Update database
            cursor.execute('''
                UPDATE tweets
                SET sentiment = ?, sentiment_score = ?, processed_at = ?
                WHERE id = ?
            ''', (sentiment, sentiment_score, datetime.now().isoformat(), tweet_id))
        
        conn.commit()
        logger.info(f"Analyzed sentiment for {len(tweets)} tweets")
    
    except Exception as e:
        logger.error(f"Error analyzing tweets sentiment: {str(e)}")
        if 'conn' in locals():
            conn.rollback()
    finally:
        if 'conn' in locals():
            conn.close()

def get_sentiment_summary(days=7):
    """
    Get a summary of sentiment analysis for the past days.
    
    Args:
        days (int): Number of days to include
    
    Returns:
        dict: Sentiment summary
    """
    try:
        # Get database connection
        db_path = os.path.join(os.path.dirname(__file__), '../../instance/finance.db')
        conn = sqlite3.connect(db_path)
        
        # Query news sentiment
        news_query = f'''
            SELECT date(substr(fetched_at, 1, 10)) as day, sentiment, COUNT(*) as count
            FROM news_articles
            WHERE processed_at IS NOT NULL
            AND fetched_at >= date('now', '-{days} days')
            GROUP BY day, sentiment
            ORDER BY day, sentiment
        '''
        
        news_df = pd.read_sql_query(news_query, conn)
        
        # Query tweet sentiment
        tweet_query = f'''
            SELECT date(substr(fetched_at, 1, 10)) as day, sentiment, COUNT(*) as count
            FROM tweets
            WHERE processed_at IS NOT NULL
            AND fetched_at >= date('now', '-{days} days')
            GROUP BY day, sentiment
            ORDER BY day, sentiment
        '''
        
        tweet_df = pd.read_sql_query(tweet_query, conn)
        
        # Prepare summary
        summary = {
            'news': {},
            'tweets': {},
            'overall': {
                'positive': 0,
                'negative': 0,
                'neutral': 0
            }
        }
        
        # Process news sentiment
        for _, row in news_df.iterrows():
            day = row['day']
            sentiment = row['sentiment']
            count = row['count']
            
            if day not in summary['news']:
                summary['news'][day] = {
                    'positive': 0,
                    'negative': 0,
                    'neutral': 0
                }
            
            summary['news'][day][sentiment] = count
            summary['overall'][sentiment] += count
        
        # Process tweet sentiment
        for _, row in tweet_df.iterrows():
            day = row['day']
            sentiment = row['sentiment']
            count = row['count']
            
            if day not in summary['tweets']:
                summary['tweets'][day] = {
                    'positive': 0,
                    'negative': 0,
                    'neutral': 0
                }
            
            summary['tweets'][day][sentiment] = count
            summary['overall'][sentiment] += count
        
        # Calculate sentiment scores
        for source in ['news', 'tweets']:
            for day in summary[source]:
                total = sum(summary[source][day].values())
                if total > 0:
                    summary[source][day]['score'] = (
                        summary[source][day]['positive'] - summary[source][day]['negative']
                    ) / total
                else:
                    summary[source][day]['score'] = 0
        
        # Calculate overall sentiment score
        total_overall = sum(summary['overall'].values())
        if total_overall > 0:
            summary['overall']['score'] = (
                summary['overall']['positive'] - summary['overall']['negative']
            ) / total_overall
        else:
            summary['overall']['score'] = 0
        
        return summary
    
    except Exception as e:
        logger.error(f"Error getting sentiment summary: {str(e)}")
        return {
            'news': {},
            'tweets': {},
            'overall': {
                'positive': 0,
                'negative': 0,
                'neutral': 0,
                'score': 0
            }
        }
    finally:
        if 'conn' in locals():
            conn.close()