"""
Module for scraping financial news from Reuters.
"""
import logging
import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3
import os
from datetime import datetime
import time
import random

logger = logging.getLogger(__name__)

def scrape_reuters(keywords=None, max_pages=3):
    """
    Scrape Reuters for financial news.
    
    Args:
        keywords (list): List of keywords to filter articles by
        max_pages (int): Maximum number of pages to scrape
    
    Returns:
        list: List of article dictionaries
    """
    if keywords is None:
        keywords = ["bitcoin", "crypto", "blockchain"]
    
    logger.info(f"Scraping Reuters for keywords: {keywords}")
    
    articles = []
    base_url = "https://www.reuters.com"
    
    # User agent rotation to avoid blocking
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0'
    ]
    
    try:
        for page in range(1, max_pages + 1):
            # Add delay to avoid rate limiting
            time.sleep(random.uniform(2, 5))
            
            # Rotate user agents
            headers = {
                'User-Agent': random.choice(user_agents),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Cache-Control': 'max-age=0'
            }
            
            # Reuters search URL
            search_term = "+".join(keywords)
            url = f"{base_url}/site-search/?query={search_term}&offset={10 * (page - 1)}"
            
            response = requests.get(url, headers=headers)
            
            if response.status_code != 200:
                logger.warning(f"Failed to fetch Reuters page {page}: Status code {response.status_code}")
                continue
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract articles from search results
            article_elements = soup.select('.search-result-indiv')
            
            if not article_elements:
                logger.info(f"No articles found on page {page}")
                break
            
            for article in article_elements:
                try:
                    # Extract title
                    title_element = article.select_one('h3')
                    if not title_element:
                        continue
                    
                    title = title_element.text.strip()
                    
                    # Extract link
                    link_element = article.select_one('a')
                    if not link_element or not link_element.has_attr('href'):
                        continue
                    
                    link = link_element['href']
                    if not link.startswith('http'):
                        link = base_url + link
                    
                    # Extract date
                    date_element = article.select_one('time')
                    if date_element and date_element.has_attr('datetime'):
                        date = date_element['datetime']
                    else:
                        date_element = article.select_one('.search-result-timestamp')
                        date = date_element.text.strip() if date_element else datetime.now().isoformat()
                    
                    # Extract summary
                    summary_element = article.select_one('p')
                    summary = summary_element.text.strip() if summary_element else ""
                    
                    # Create article object
                    article_obj = {
                        'title': title,
                        'url': link,
                        'date': date,
                        'summary': summary,
                        'source': 'Reuters',
                        'keywords': keywords,
                        'fetched_at': datetime.now().isoformat()
                    }
                    
                    articles.append(article_obj)
                
                except Exception as e:
                    logger.error(f"Error parsing Reuters article: {str(e)}")
                    continue
            
            logger.info(f"Scraped {len(articles)} articles from Reuters (page {page})")
        
        return articles
    
    except Exception as e:
        logger.error(f"Error scraping Reuters: {str(e)}")
        return articles

def store_articles(articles):
    """
    Store scraped articles in the database.
    
    Args:
        articles (list): List of article dictionaries
    """
    if not articles:
        logger.warning("No articles to store")
        return
    
    try:
        # Get database connection
        db_path = os.path.join(os.path.dirname(__file__), '../../instance/finance.db')
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS news_articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                url TEXT UNIQUE,
                date TEXT,
                summary TEXT,
                source TEXT,
                keywords TEXT,
                sentiment TEXT,
                sentiment_score REAL,
                fetched_at TEXT,
                processed_at TEXT
            )
        ''')
        
        # Insert articles
        for article in articles:
            cursor.execute('''
                INSERT OR IGNORE INTO news_articles 
                (title, url, date, summary, source, keywords, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                article['title'],
                article['url'],
                article['date'],
                article['summary'],
                article['source'],
                ','.join(article['keywords']),
                article['fetched_at']
            ))
        
        conn.commit()
        logger.info(f"Stored {cursor.rowcount} new articles in the database")
        
    except Exception as e:
        logger.error(f"Error storing articles: {str(e)}")
        if 'conn' in locals():
            conn.rollback()
    finally:
        if 'conn' in locals():
            conn.close()

def get_article_content(url):
    """
    Get the full content of an article.
    
    Args:
        url (str): URL of the article
    
    Returns:
        str: Article content
    """
    try:
        # User agent rotation to avoid blocking
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0'
        ]
        
        headers = {
            'User-Agent': random.choice(user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0'
        }
        
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            logger.warning(f"Failed to fetch article content: Status code {response.status_code}")
            return ""
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract article content
        article_body = soup.select_one('article')
        if not article_body:
            article_body = soup.select_one('.ArticleBody__content')
        
        if not article_body:
            return ""
        
        # Get all paragraphs
        paragraphs = article_body.select('p')
        content = ' '.join([p.text.strip() for p in paragraphs])
        
        return content
    
    except Exception as e:
        logger.error(f"Error fetching article content: {str(e)}")
        return ""

def update_reuters_news():
    """Update Reuters news articles in the database."""
    try:
        articles = scrape_reuters()
        if articles:
            store_articles(articles)
    except Exception as e:
        logger.error(f"Error updating Reuters news: {str(e)}")