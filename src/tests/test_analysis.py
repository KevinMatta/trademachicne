"""
Test module for analysis components.
"""
import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from datetime import datetime

# Add project directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.analysis.nlp_sentiment import analyze_sentiment_openai
from src.analysis.forecasting_model import prepare_data, train_lstm_model
from src.analysis.ict_rules import identify_swings, identify_fair_value_gaps

class TestSentimentAnalysis(unittest.TestCase):
    @patch('src.analysis.nlp_sentiment.requests.post')
    @patch('src.analysis.nlp_sentiment.load_config')
    def test_sentiment_analysis_positive(self, mock_load_config, mock_post):
        # Mock configuration
        mock_load_config.return_value = {
            'apis': {
                'openai': {
                    'api_key': 'test_key'
                }
            },
            'analysis': {
                'sentiment': {
                    'model': 'test-model'
                }
            }
        }
        
        # Mock API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [
                {
                    'text': 'positive'
                }
            ]
        }
        mock_post.return_value = mock_response
        
        text = "Bitcoin price is soaring after positive news."
        sentiment = analyze_sentiment_openai(text)
        self.assertEqual(sentiment, 'positive')
    
    @patch('src.analysis.nlp_sentiment.requests.post')
    @patch('src.analysis.nlp_sentiment.load_config')
    def test_sentiment_analysis_negative(self, mock_load_config, mock_post):
        # Mock configuration
        mock_load_config.return_value = {
            'apis': {
                'openai': {
                    'api_key': 'test_key'
                }
            },
            'analysis': {
                'sentiment': {
                    'model': 'test-model'
                }
            }
        }
        
        # Mock API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [
                {
                    'text': 'negative'
                }
            ]
        }
        mock_post.return_value = mock_response
        
        text = "Major hack reported on a crypto exchange, prices are crashing."
        sentiment = analyze_sentiment_openai(text)
        self.assertEqual(sentiment, 'negative')

class TestForecastingModel(unittest.TestCase):
    @patch('src.analysis.forecasting_model.load_market_data_from_db')
    def test_prepare_data(self, mock_load_data):
        # Create mock data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'timestamp': dates,
            'open': range(100, 200),
            'high': range(110, 210),
            'low': range(90, 190),
            'close': range(105, 205),
            'volume': range(1000, 1100)
        })
        mock_load_data.return_value = df
        
        # Test with patch for talib functions
        with patch('talib.SMA', return_value=pd.Series(range(100))), \
             patch('talib.EMA', return_value=pd.Series(range(100))), \
             patch('talib.RSI', return_value=pd.Series(range(100))), \
             patch('talib.MACD', return_value=(pd.Series(range(100)), pd.Series(range(100)), pd.Series(range(100)))), \
             patch('talib.BBANDS', return_value=(pd.Series(range(100)), pd.Series(range(100)), pd.Series(range(100)))), \
             patch('talib.ATR', return_value=pd.Series(range(100))), \
             patch('talib.ADX', return_value=pd.Series(range(100))), \
             patch('talib.OBV', return_value=pd.Series(range(100))), \
             patch('talib.AD', return_value=pd.Series(range(100))), \
             patch('talib.ADOSC', return_value=pd.Series(range(100))), \
             patch('talib.STOCH', return_value=(pd.Series(range(100)), pd.Series(range(100)))):
            
            # Call the function
            X_train, X_test, y_train, y_test, scaler = prepare_data('BTCUSDT', '1d', lookback=10, horizon=2)
            
            # Assertions
            self.assertIsNotNone(X_train)
            self.assertIsNotNone(y_train)
            self.assertTrue(len(X_train) > 0)
            self.assertTrue(len(y_train) > 0)
            self.assertEqual(X_train.shape[1], 10)  # lookback
            self.assertEqual(y_train.shape[1], 2)   # horizon

class TestICTRules(unittest.TestCase):
    def test_identify_swings(self):
        # Create test data
        dates = pd.date_range(start='2023-01-01', periods=20, freq='D')
        
        # Create a pattern with obvious swing points
        highs = [100, 105, 110, 105, 100, 95, 90, 95, 100, 105, 110, 115, 120, 115, 110, 105, 100, 95, 90, 85]
        lows = [90, 95, 100, 95, 90, 85, 80, 85, 90, 95, 100, 105, 110, 105, 100, 95, 90, 85, 80, 75]
        
        df = pd.DataFrame({
            'timestamp': dates,
            'high': highs,
            'low': lows,
            'close': [(h+l)/2 for h, l in zip(highs, lows)]
        })
        
        # Identify swings
        result = identify_swings(df, threshold=0.001, swing_periods=2)
        
        # Verify that swings were identified
        self.assertTrue('swing_high' in result.columns)
        self.assertTrue('swing_low' in result.columns)
        self.assertTrue(result['swing_high'].any())
        self.assertTrue(result['swing_low'].any())
    
    def test_identify_fair_value_gaps(self):
        # Create test data
        dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
        
        # Create a pattern with a bullish gap (Low1 > High3)
        # 100-110, 105-115, 90-100 (gap between candle 1 and 3)
        opens = [100, 105, 90, 95, 100, 105, 110, 105, 100, 95]
        highs = [110, 115, 100, 105, 110, 115, 120, 115, 110, 105]
        lows = [100, 105, 90, 95, 100, 105, 110, 105, 100, 95]
        closes = [105, 110, 95, 100, 105, 110, 115, 110, 105, 100]
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes
        })
        
        # Identify FVGs
        result = identify_fair_value_gaps(df)
        
        # Verify that FVGs were identified
        self.assertTrue('bullish_fvg' in result.columns)
        self.assertTrue('bearish_fvg' in result.columns)
        # In our test data, there should be a bullish FVG
        self.assertTrue(result['bullish_fvg'].any())

if __name__ == '__main__':
    unittest.main()