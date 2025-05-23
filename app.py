from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from werkzeug.utils import secure_filename
import tempfile
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user, login_required

app = Flask(__name__, static_folder='dist', static_url_path='/')
CORS(app)

# Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'default-dev-key')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///finance.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Database setup
db = SQLAlchemy(app)

# Login manager setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    reports = db.relationship('Report', backref='owner', lazy='dynamic')

class Report(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    description = db.Column(db.String(500))
    data = db.Column(db.Text, nullable=False)  # JSON data
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))

with app.app_context():
    db.create_all()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Financial analysis functions
def calculate_basic_metrics(df):
    """Calculate basic financial metrics from dataframe"""
    try:
        # Assuming df has columns like 'revenue', 'expenses', etc.
        metrics = {}
        
        # Basic calculations - modify according to your data structure
        if 'revenue' in df.columns and 'expenses' in df.columns:
            df['profit'] = df['revenue'] - df['expenses']
            df['profit_margin'] = (df['profit'] / df['revenue']) * 100
            
            metrics['total_revenue'] = float(df['revenue'].sum())
            metrics['total_expenses'] = float(df['expenses'].sum())
            metrics['total_profit'] = float(df['profit'].sum())
            metrics['avg_profit_margin'] = float(df['profit_margin'].mean())
            
            # Growth calculations if temporal data exists
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')
                
                # Calculate period-over-period growth
                df['revenue_growth'] = df['revenue'].pct_change() * 100
                metrics['avg_revenue_growth'] = float(df['revenue_growth'].mean())
        
        return metrics
    except Exception as e:
        print(f"Error in calculate_basic_metrics: {e}")
        return {"error": str(e)}

# Routes
@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        # Determine file type and read
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(filepath)
        else:
            return jsonify({"error": "Unsupported file format. Please upload CSV or Excel files."}), 400
        
        # Process the data
        data_preview = df.head(5).to_dict(orient='records')
        column_types = {col: str(dtype) for col, dtype in df.dtypes.items()}
        
        # Calculate metrics if possible
        metrics = calculate_basic_metrics(df)
        
        return jsonify({
            "success": True,
            "preview": data_preview,
            "columns": list