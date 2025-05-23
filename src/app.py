from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import os
import json
import logging
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user, login_required
from werkzeug.security import generate_password_hash, check_password_hash
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, static_folder='../dist', static_url_path='/')
CORS(app)

# Load configuration
def load_config():
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    else:
        logger.warning("Config file not found. Using default configuration.")
        return {
            'app': {
                'secret_key': os.environ.get('SECRET_KEY', 'dev-key-change-in-production'),
                'database_uri': os.environ.get('DATABASE_URL', 'sqlite:///finance.db'),
                'upload_folder': os.environ.get('UPLOAD_FOLDER', '/tmp/uploads'),
                'max_content_length': 16 * 1024 * 1024  # 16MB
            },
            'apis': {
                'openai_api_key': os.environ.get('OPENAI_API_KEY', ''),
                'binance': {
                    'api_key': os.environ.get('BINANCE_API_KEY', ''),
                    'api_secret': os.environ.get('BINANCE_API_SECRET', '')
                },
                'twitter': {
                    'api_key': os.environ.get('TWITTER_API_KEY', ''),
                    'api_secret': os.environ.get('TWITTER_API_SECRET', '')
                }
            }
        }

config = load_config()

# App configuration
app.config['SECRET_KEY'] = config['app']['secret_key']
app.config['SQLALCHEMY_DATABASE_URI'] = config['app']['database_uri']
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = config['app']['upload_folder']
app.config['MAX_CONTENT_LENGTH'] = config['app']['max_content_length']

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Database setup
db = SQLAlchemy(app)

# Login manager setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Import modules after app initialization to avoid circular imports
from src.data_ingest.binance_rest import get_historical_data
from src.analysis.nlp_sentiment import analyze_sentiment_openai
from src.decision_engine.engine import evaluate_signal
from src.preprocessing.indicators import calculate_indicators

# Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    api_key = db.Column(db.String(128))
    api_secret = db.Column(db.String(128))
    reports = db.relationship('Report', backref='owner', lazy='dynamic')
    strategies = db.relationship('Strategy', backref='owner', lazy='dynamic')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Report(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    description = db.Column(db.String(500))
    data = db.Column(db.Text, nullable=False)  # JSON data
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))

class Strategy(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.String(500))
    config = db.Column(db.Text, nullable=False)  # JSON configuration
    is_active = db.Column(db.Boolean, default=False)
    last_run = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    trades = db.relationship('Trade', backref='strategy', lazy='dynamic')

class Trade(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(20), nullable=False)
    entry_price = db.Column(db.Float, nullable=False)
    exit_price = db.Column(db.Float)
    quantity = db.Column(db.Float, nullable=False)
    side = db.Column(db.String(10), nullable=False)  # BUY or SELL
    status = db.Column(db.String(20), default='OPEN')  # OPEN, CLOSED, CANCELLED
    pnl = db.Column(db.Float)
    entry_time = db.Column(db.DateTime, default=datetime.utcnow)
    exit_time = db.Column(db.DateTime)
    strategy_id = db.Column(db.Integer, db.ForeignKey('strategy.id'))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))

with app.app_context():
    db.create_all()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Routes
@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/api/health')
def health_check():
    return jsonify({"status": "ok", "timestamp": datetime.utcnow().isoformat()})

# User authentication routes
@app.route('/api/auth/register', methods=['POST'])
def register():
    data = request.get_json()
    
    if User.query.filter_by(username=data.get('username')).first():
        return jsonify({"error": "Username already exists"}), 400
    
    if User.query.filter_by(email=data.get('email')).first():
        return jsonify({"error": "Email already exists"}), 400
    
    user = User(
        username=data.get('username'),
        email=data.get('email')
    )
    user.set_password(data.get('password'))
    
    db.session.add(user)
    db.session.commit()
    
    return jsonify({"message": "User registered successfully"}), 201

@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.get_json()
    user = User.query.filter_by(username=data.get('username')).first()
    
    if user and user.check_password(data.get('password')):
        login_user(user)
        return jsonify({
            "message": "Login successful",
            "user": {
                "id": user.id,
                "username": user.username,
                "email": user.email
            }
        })
    
    return jsonify({"error": "Invalid username or password"}), 401

@app.route('/api/auth/logout')
@login_required
def logout():
    logout_user()
    return jsonify({"message": "Logged out successfully"})

# Data analysis routes
@app.route('/api/market/historical', methods=['GET'])
@login_required
def get_market_data():
    symbol = request.args.get('symbol', 'BTCUSDT')
    interval = request.args.get('interval', '1d')
    limit = request.args.get('limit', 100, type=int)
    
    try:
        data = get_historical_data(symbol, interval, limit)
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error fetching market data: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/analysis/sentiment', methods=['POST'])
@login_required
def analyze_text_sentiment():
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    try:
        sentiment = analyze_sentiment_openai(text)
        return jsonify({"text": text, "sentiment": sentiment})
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/strategies', methods=['GET'])
@login_required
def get_strategies():
    strategies = Strategy.query.filter_by(user_id=current_user.id).all()
    return jsonify([{
        "id": s.id,
        "name": s.name,
        "description": s.description,
        "is_active": s.is_active,
        "created_at": s.created_at.isoformat()
    } for s in strategies])

@app.route('/api/strategies', methods=['POST'])
@login_required
def create_strategy():
    data = request.get_json()
    
    strategy = Strategy(
        name=data.get('name'),
        description=data.get('description'),
        config=json.dumps(data.get('config', {})),
        user_id=current_user.id
    )
    
    db.session.add(strategy)
    db.session.commit()
    
    return jsonify({
        "id": strategy.id,
        "name": strategy.name,
        "message": "Strategy created successfully"
    }), 201

@app.route('/api/trades', methods=['GET'])
@login_required
def get_trades():
    trades = Trade.query.filter_by(user_id=current_user.id).all()
    return jsonify([{
        "id": t.id,
        "symbol": t.symbol,
        "entry_price": t.entry_price,
        "exit_price": t.exit_price,
        "quantity": t.quantity,
        "side": t.side,
        "status": t.status,
        "pnl": t.pnl,
        "entry_time": t.entry_time.isoformat(),
        "exit_time": t.exit_time.isoformat() if t.exit_time else None
    } for t in trades])

# Error handlers
@app.errorhandler(404)
def not_found(e):
    if request.path.startswith('/api/'):
        return jsonify({"error": "Not found"}), 404
    return app.send_static_file('index.html')

@app.errorhandler(500)
def server_error(e):
    logger.error(f"Server error: {str(e)}")
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))