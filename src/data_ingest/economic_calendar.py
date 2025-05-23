"""
Module for fetching economic calendar data from various sources.
"""
import logging
import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3
import os
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def fetch_investing_calendar(days=7):
    """
    Fetch economic calendar data from Investing.com.
    
    Args:
        days (int): Number of days to fetch data for
    
    Returns:
        pd.DataFrame: Economic calendar events
    """
    try:
        logger.info(f"Fetching economic calendar data for the next {days} days")
        
        # Calculate date range
        today = datetime.now()
        end_date = today + timedelta(days=days)
        
        # Format dates for URL
        today_str = today.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        # Investing.com uses a specific format for their API
        url = f"https://www.investing.com/economic-calendar/Service/getCalendarFilteredData"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'X-Requested-With': 'XMLHttpRequest',
            'Referer': 'https://www.investing.com/economic-calendar/'
        }
        
        payload = {
            'country[]': '25,32,6,37,72,22,17,39,14,10,35,43,56,36,110,11,26,12,4,5',  # Major economies
            'dateFrom': today_str,
            'dateTo': end_date_str,
            'timeZone': '55',  # GMT-4
            'timeFilter': 'timeRemain',
            'currentTab': 'custom',
            'limit_from': 0
        }
        
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()
        
        # Parse HTML response
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'id': 'economicCalendarData'})
        
        if not table:
            logger.warning("No economic calendar data found")
            return pd.DataFrame()
        
        # Parse table data
        rows = table.find_all('tr', {'class': 'js-event-item'})
        
        events = []
        for row in rows:
            try:
                time_cell = row.find('td', {'class': 'time'})
                time = time_cell.text.strip() if time_cell else ''
                
                country_cell = row.find('td', {'class': 'flagCur'})
                country = country_cell.find('span').get('title') if country_cell and country_cell.find('span') else ''
                
                event_cell = row.find('td', {'class': 'event'})
                event = event_cell.text.strip() if event_cell else ''
                
                impact_cell = row.find('td', {'class': 'sentiment'})
                impact = len(impact_cell.find_all('i', {'class': 'grayFullBullishIcon'})) if impact_cell else 0
                
                actual_cell = row.find('td', {'class': 'act'})
                actual = actual_cell.text.strip() if actual_cell else ''
                
                forecast_cell = row.find('td', {'class': 'fore'})
                forecast = forecast_cell.text.strip() if forecast_cell else ''
                
                previous_cell = row.find('td', {'class': 'prev'})
                previous = previous_cell.text.strip() if previous_cell else ''
                
                # Extract date from row attributes
                date_str = row.get('data-event-datetime', '')
                date = datetime.strptime(date_str, '%Y/%m/%d %H:%M:%S') if date_str else None
                
                events.append({
                    'date': date,
                    'time': time,
                    'country': country,
                    'event': event,
                    'impact': impact,
                    'actual': actual,
                    'forecast': forecast,
                    'previous': previous
                })
            except Exception as e:
                logger.error(f"Error parsing calendar row: {str(e)}")
                continue
        
        # Create DataFrame
        df = pd.DataFrame(events)
        logger.info(f"Fetched {len(df)} economic calendar events")
        
        return df
    
    except Exception as e:
        logger.error(f"Error fetching economic calendar: {str(e)}")
        return pd.DataFrame()

def store_calendar_events(events_df):
    """
    Store economic calendar events in the database.
    
    Args:
        events_df (pd.DataFrame): Economic calendar events
    """
    if events_df.empty:
        logger.warning("No events to store")
        return
    
    try:
        # Get database connection
        db_path = os.path.join(os.path.dirname(__file__), '../../instance/finance.db')
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS economic_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                time TEXT,
                country TEXT,
                event TEXT,
                impact INTEGER,
                actual TEXT,
                forecast TEXT,
                previous TEXT,
                created_at TEXT
            )
        ''')
        
        # Insert events
        for _, event in events_df.iterrows():
            cursor.execute('''
                INSERT INTO economic_events 
                (date, time, country, event, impact, actual, forecast, previous, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                event['date'].isoformat() if pd.notna(event['date']) else None,
                event['time'],
                event['country'],
                event['event'],
                event['impact'],
                event['actual'],
                event['forecast'],
                event['previous'],
                datetime.now().isoformat()
            ))
        
        conn.commit()
        logger.info(f"Stored {len(events_df)} economic events in the database")
        
    except Exception as e:
        logger.error(f"Error storing economic events: {str(e)}")
        if 'conn' in locals():
            conn.rollback()
    finally:
        if 'conn' in locals():
            conn.close()

def update_economic_calendar():
    """Update economic calendar data in the database."""
    try:
        events_df = fetch_investing_calendar()
        if not events_df.empty:
            store_calendar_events(events_df)
    except Exception as e:
        logger.error(f"Error updating economic calendar: {str(e)}")

def get_upcoming_events(days=7, min_impact=2):
    """
    Get upcoming high-impact economic events.
    
    Args:
        days (int): Number of days to look ahead
        min_impact (int): Minimum impact level (1-3)
    
    Returns:
        list: Upcoming high-impact events
    """
    try:
        # Get database connection
        db_path = os.path.join(os.path.dirname(__file__), '../../instance/finance.db')
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Set date range
        today = datetime.now().date()
        end_date = (today + timedelta(days=days)).isoformat()
        today = today.isoformat()
        
        # Query events
        cursor.execute('''
            SELECT date, time, country, event, impact, forecast, previous
            FROM economic_events
            WHERE date BETWEEN ? AND ?
            AND impact >= ?
            ORDER BY date, time
        ''', (today, end_date, min_impact))
        
        events = []
        for row in cursor.fetchall():
            events.append({
                'date': row[0],
                'time': row[1],
                'country': row[2],
                'event': row[3],
                'impact': row[4],
                'forecast': row[5],
                'previous': row[6]
            })
        
        return events
        
    except Exception as e:
        logger.error(f"Error getting upcoming events: {str(e)}")
        return []
    finally:
        if 'conn' in locals():
            conn.close()