"""
Test module for decision engine components.
"""
import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import json
from datetime import datetime

# Add project directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.decision_engine.engine import evaluate_technical_signals, evaluate_sentiment_signals, evaluate_signal
from src.executor.binance_executor import calculate_quantity, create_order

class TestDecisionEngine(unittest.TestCase):
    @patch('src.decision_engine.engine.get_indicators')
    def test_evaluate_technical_signals(self, mock_get_indicators):
        # Create mock indicators data
        dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
        df = pd.DataFrame({
            'timestamp': dates,
            'close': [100, 102, 105, 103, 101, 98, 95, 97, 100, 103],
            'sma_20': [100, 101, 102, 103, 103, 102, 101, 100, 100, 101],
            'sma_50': [95, 96, 97, 98, 99, 100, 100, 100, 100, 100],
            'rsi_14': [60, 65, 70, 75, 72, 65, 55, 45, 50, 55],
            'macd': [1, 1.2, 1.5, 1.3, 1.0, 0.5, 0.2, 0.1, 0.3, 0.5],
            'macd_signal': [0.8, 0.9, 1.0, 1.1, 1.2, 1.0, 0.8, 0.5, 0.3, 0.2],
            'bbands_upper': [110, 112, 115, 113, 111, 108, 105, 107, 110, 113],
            'bbands_lower': [90, 92, 95, 93, 91, 88, 85, 87, 90, 93]
        })
        mock_get_indicators.return_value = df
        
        # Call the function
        signals = evaluate_technical_signals('BTCUSDT', '1d')
        
        # Assertions
        self.assertIn('strength', signals)
        self.assertIn('direction', signals)
        self.assertIn('signals', signals)
        self.assertTrue(isinstance(signals['signals'], list))
        
        # Check we get different signals depending on conditions
        self.assertEqual(signals['direction'], 'bullish')  # Because SMA 20 > SMA 50
        
        # Test with bearish conditions
        df['sma_20'] = [90, 89, 88, 87, 86, 85, 84, 83, 82, 81]  # SMA 20 < SMA 50
        df['rsi_14'] = [80, 82, 85, 83, 80, 75, 72, 70, 75, 78]  # RSI overbought
        mock_get_indicators.return_value = df
        
        signals = evaluate_technical_signals('BTCUSDT', '1d')
        self.assertEqual(signals['direction'], 'bearish')
    
    @patch('src.decision_engine.engine.get_sentiment_summary')
    def test_evaluate_sentiment_signals(self, mock_get_sentiment_summary):
        # Test bullish sentiment
        mock_get_sentiment_summary.return_value = {
            'overall': {
                'positive': 15,
                'negative': 5,
                'neutral': 10,
                'score': 0.3  # Bullish
            }
        }
        
        signals = evaluate_sentiment_signals()
        self.assertEqual(signals['direction'], 'bullish')
        self.assertTrue(signals['strength'] > 0)
        
        # Test bearish sentiment
        mock_get_sentiment_summary.return_value = {
            'overall': {
                'positive': 5,
                'negative': 15,
                'neutral': 10,
                'score': -0.3  # Bearish
            }
        }
        
        signals = evaluate_sentiment_signals()
        self.assertEqual(signals['direction'], 'bearish')
        self.assertTrue(signals['strength'] > 0)
        
        # Test neutral sentiment
        mock_get_sentiment_summary.return_value = {
            'overall': {
                'positive': 10,
                'negative': 10,
                'neutral': 10,
                'score': 0  # Neutral
            }
        }
        
        signals = evaluate_sentiment_signals()
        self.assertEqual(signals['direction'], 'neutral')
        self.assertEqual(signals['strength'], 0)
    
    @patch('src.decision_engine.engine.evaluate_technical_signals')
    @patch('src.decision_engine.engine.evaluate_sentiment_signals')
    @patch('src.decision_engine.engine.evaluate_price_prediction')
    @patch('src.decision_engine.engine.evaluate_market_structure')
    @patch('src.decision_engine.engine.sqlite3.connect')
    def test_evaluate_signal(self, mock_connect, mock_structure, mock_prediction, mock_sentiment, mock_technical):
        # Mock individual signals
        mock_technical.return_value = {
            'direction': 'bullish',
            'strength': 0.7,
            'signals': [{'name': 'MA Cross', 'direction': 'bullish', 'strength': 1}]
        }
        
        mock_sentiment.return_value = {
            'direction': 'bullish',
            'strength': 0.5,
            'score': 0.25
        }
        
        mock_prediction.return_value = {
            'direction': 'bullish',
            'strength': 0.6,
            'predicted_change': 2.5
        }
        
        mock_structure.return_value = {
            'direction': 'bullish',
            'strength': 0.8,
            'signals': [{'name': 'Support Proximity', 'direction': 'bullish', 'strength': 2}]
        }
        
        # Mock database connection
        mock_cursor = MagicMock()
        mock_connection = MagicMock()
        mock_connection.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_connection
        
        # Call the function
        signal = evaluate_signal('BTCUSDT', '1d')
        
        # Assertions
        self.assertEqual(signal['direction'], 'bullish')
        self.assertTrue(signal['strength'] > 0)
        self.assertEqual(signal['symbol'], 'BTCUSDT')
        self.assertEqual(signal['interval'], '1d')
        self.assertIn('components', signal)
        
        # Test with mixed signals
        mock_technical.return_value['direction'] = 'bearish'
        mock_technical.return_value['strength'] = 0.9
        mock_sentiment.return_value['direction'] = 'bearish'
        mock_sentiment.return_value['strength'] = 0.8
        
        signal = evaluate_signal('BTCUSDT', '1d')
        
        # With strong bearish technical and sentiment but bullish prediction and structure,
        # the overall signal could go either way depending on weights
        self.assertIn(signal['direction'], ['bullish', 'bearish'])

class TestBinanceExecutor(unittest.TestCase):
    @patch('src.executor.binance_executor.get_symbol_price')
    def test_calculate_quantity(self, mock_get_symbol_price):
        # Mock price
        mock_get_symbol_price.return_value = 50000  # BTC price
        
        # Calculate quantity for $10,000 worth of BTC
        quantity = calculate_quantity('BTCUSDT', 10000, 'BUY')
        
        # Assertions
        self.assertEqual(quantity, 0.2)  # 10000 / 50000 = 0.2
        
        # Test with different price
        mock_get_symbol_price.return_value = 2000  # ETH price
        quantity = calculate_quantity('ETHUSDT', 5000, 'BUY')
        self.assertEqual(quantity, 2.5)  # 5000 / 2000 = 2.5
    
    @patch('src.executor.binance_executor.requests.post')
    @patch('src.executor.binance_executor.get_api_url')
    @patch('src.executor.binance_executor.get_headers')
    @patch('src.executor.binance_executor.get_signature')
    def test_create_order(self, mock_get_signature, mock_get_headers, mock_get_api_url, mock_post):
        # Mock API URL
        mock_get_api_url.return_value = 'https://testnet.binance.vision/api'
        
        # Mock headers
        mock_get_headers.return_value = {'X-MBX-APIKEY': 'test_key'}
        
        # Mock signature
        mock_get_signature.return_value = 'test_signature'
        
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'symbol': 'BTCUSDT',
            'orderId': 12345,
            'price': '50000.00',
            'origQty': '0.2',
            'side': 'BUY',
            'type': 'MARKET',
            'status': 'FILLED',
            'fills': [
                {
                    'price': '50000.00',
                    'qty': '0.2',
                    'commission': '0.00002',
                    'commissionAsset': 'BTC'
                }
            ]
        }
        mock_post.return_value = mock_response
        
        # Call the function
        order = create_order('BTCUSDT', 'BUY', 0.2, 'test_key', 'test_secret')
        
        # Assertions
        self.assertEqual(order['symbol'], 'BTCUSDT')
        self.assertEqual(order['orderId'], 12345)
        self.assertEqual(order['origQty'], '0.2')
        self.assertEqual(order['side'], 'BUY')
        self.assertEqual(order['status'], 'FILLED')

if __name__ == '__main__':
    unittest.main()