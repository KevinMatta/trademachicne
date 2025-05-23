"""
Test module for data ingestion components.
"""
import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from datetime import datetime

# Add project directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data_ingest.binance_rest import get_historical_data, get_exchange_info
from src.data_ingest.economic_calendar import fetch_investing_calendar
from src.news_scraper.bloomberg_scraper import scrape_bloomberg

class TestBinanceRest(unittest.TestCase):
    @patch('src.data_ingest.binance_rest.requests.get')
    @patch('src.data_ingest.binance_rest.get_api_url')
    @patch('src.data_ingest.binance_rest.get_headers')
    def test_get_historical_data(self, mock_get_headers, mock_get_api_url, mock_get):
        # Mock API URL
        mock_get_api_url.return_value = 'https://testnet.binance.vision/api'
        
        # Mock headers
        mock_get_headers.return_value = {'X-MBX-APIKEY': 'test_key'}
        
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            [1625097600000, "35000.0", "35500.0", "34800.0", "35200.0", "100.0", 
             1625097900000, "3520000.0", 100, "50.0", "1760000.0", "0"],
            [1625097900000, "35200.0", "35700.0", "35100.0", "35600.0", "120.0", 
             1625098200000, "4272000.0", 120, "60.0", "2136000.0", "0"]
        ]
        mock_get.return_value = mock_response
        
        # Call the function
        data = get_historical_data('BTCUSDT', '5m', 2)
        
        # Assertions
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]['open'], 35000.0)
        self.assertEqual(data[0]['high'], 35500.0)
        self.assertEqual(data[1]['close'], 35600.0)
    
    @patch('src.data_ingest.binance_rest.requests.get')
    @patch('src.data_ingest.binance_rest.get_api_url')
    @patch('src.data_ingest.binance_rest.get_headers')
    def test_get_exchange_info(self, mock_get_headers, mock_get_api_url, mock_get):
        # Mock API URL
        mock_get_api_url.return_value = 'https://testnet.binance.vision/api'
        
        # Mock headers
        mock_get_headers.return_value = {'X-MBX-APIKEY': 'test_key'}
        
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'timezone': 'UTC',
            'serverTime': 1625097600000,
            'symbols': [
                {
                    'symbol': 'BTCUSDT',
                    'status': 'TRADING',
                    'baseAsset': 'BTC',
                    'quoteAsset': 'USDT'
                },
                {
                    'symbol': 'ETHUSDT',
                    'status': 'TRADING',
                    'baseAsset': 'ETH',
                    'quoteAsset': 'USDT'
                }
            ]
        }
        mock_get.return_value = mock_response
        
        # Call the function
        info = get_exchange_info()
        
        # Assertions
        self.assertEqual(len(info['symbols']), 2)
        self.assertEqual(info['symbols'][0]['symbol'], 'BTCUSDT')
        self.assertEqual(info['symbols'][1]['baseAsset'], 'ETH')

class TestEconomicCalendar(unittest.TestCase):
    @patch('src.data_ingest.economic_calendar.requests.post')
    @patch('src.data_ingest.economic_calendar.BeautifulSoup')
    def test_fetch_investing_calendar(self, mock_bs, mock_post):
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = '<html><body><table id="economicCalendarData"><tr class="js-event-item" data-event-datetime="2023/01/01 12:00:00"><td class="time">12:00</td><td class="flagCur"><span title="US"></span></td><td class="event">GDP</td><td class="sentiment"><i class="grayFullBullishIcon"></i><i class="grayFullBullishIcon"></i><i class="grayFullBullishIcon"></i></td><td class="act">3.5%</td><td class="fore">3.2%</td><td class="prev">3.0%</td></tr></table></body></html>'
        mock_post.return_value = mock_response
        
        # Mock BeautifulSoup
        mock_soup = MagicMock()
        mock_table = MagicMock()
        mock_rows = [MagicMock()]
        
        mock_time = MagicMock()
        mock_time.text = '12:00'
        
        mock_country = MagicMock()
        mock_country_span = MagicMock()
        mock_country_span.get.return_value = 'US'
        mock_country.find.return_value = mock_country_span
        
        mock_event = MagicMock()
        mock_event.text = 'GDP'
        
        mock_impact = MagicMock()
        mock_impact.find_all.return_value = [MagicMock(), MagicMock(), MagicMock()]
        
        mock_actual = MagicMock()
        mock_actual.text = '3.5%'
        
        mock_forecast = MagicMock()
        mock_forecast.text = '3.2%'
        
        mock_previous = MagicMock()
        mock_previous.text = '3.0%'
        
        mock_rows[0].find.side_effect = lambda *args, **kwargs: {
            ('td', {'class': 'time'}): mock_time,
            ('td', {'class': 'flagCur'}): mock_country,
            ('td', {'class': 'event'}): mock_event,
            ('td', {'class': 'sentiment'}): mock_impact,
            ('td', {'class': 'act'}): mock_actual,
            ('td', {'class': 'fore'}): mock_forecast,
            ('td', {'class': 'prev'}): mock_previous
        }.get((args[0], args[1]), None)
        
        mock_rows[0].get.return_value = '2023/01/01 12:00:00'
        
        mock_table.find_all.return_value = mock_rows
        mock_soup.find.return_value = mock_table
        mock_bs.return_value = mock_soup
        
        # Call the function
        calendar = fetch_investing_calendar(days=1)
        
        # Assertions
        self.assertEqual(len(calendar), 1)
        self.assertEqual(calendar.iloc[0]['event'], 'GDP')
        self.assertEqual(calendar.iloc[0]['country'], 'US')
        self.assertEqual(calendar.iloc[0]['impact'], 3)

class TestNewsScraper(unittest.TestCase):
    @patch('src.news_scraper.bloomberg_scraper.requests.get')
    @patch('src.news_scraper.bloomberg_scraper.BeautifulSoup')
    def test_scrape_bloomberg(self, mock_bs, mock_get):
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = '<html><body><article><h1>Bitcoin Surges</h1><a href="/news/bitcoin-surges">Link</a><time datetime="2023-01-01T12:00:00Z"></time><p>Bitcoin price reaches new highs.</p></article></body></html>'
        mock_get.return_value = mock_response
        
        # Mock BeautifulSoup
        mock_soup = MagicMock()
        mock_articles = [MagicMock()]
        
        mock_title = MagicMock()
        mock_title.text = 'Bitcoin Surges'
        
        mock_link = MagicMock()
        mock_link.__getitem__.side_effect = lambda key: '/news/bitcoin-surges' if key == 'href' else None
        mock_link.has_attr.return_value = True
        
        mock_date = MagicMock()
        mock_date.__getitem__.side_effect = lambda key: '2023-01-01T12:00:00Z' if key == 'datetime' else None
        mock_date.has_attr.return_value = True
        
        mock_summary = MagicMock()
        mock_summary.text = 'Bitcoin price reaches new highs.'
        
        mock_articles[0].select_one.side_effect = lambda selector: {
            'h1': mock_title,
            'a': mock_link,
            'time': mock_date,
            'p': mock_summary
        }.get(selector, None)
        
        mock_soup.select.return_value = mock_articles
        mock_bs.return_value = mock_soup
        
        # Call the function
        articles = scrape_bloomberg(['bitcoin'], max_pages=1)
        
        # Assertions
        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0]['title'], 'Bitcoin Surges')
        self.assertEqual(articles[0]['summary'], 'Bitcoin price reaches new highs.')
        self.assertEqual(articles[0]['source'], 'Bloomberg')

if __name__ == '__main__':
    unittest.main()