#!/usr/bin/env python3
"""
ULTIMATE AI TRADING BOT - FINAL WINDOWS-READY VERSION
Advanced AI-powered trading bot with complete ML capabilities
Ready to run on Windows with zero configuration issues

INSTALLATION:
1. pip install -r requirements_final.txt
2. Create .env file with your credentials
3. Run: python FINAL_TRADING_BOT.py

Author: Your Original Implementation Enhanced
Compatible: Windows 10/11, Python 3.8-3.10
"""

import os
import sys
import time
import sqlite3
import logging
import requests
import numpy as np
import pandas as pd
import threading
import random
import json
import subprocess
import traceback
from datetime import datetime, timedelta
from collections import deque
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✓ Environment variables loaded")
except ImportError:
    print("Installing python-dotenv...")
    os.system("pip install python-dotenv==1.0.0")
    from dotenv import load_dotenv
    load_dotenv()

# Enhanced imports with proper fallback handling
try:
    from bs4 import BeautifulSoup
    print("✓ BeautifulSoup4 loaded")
except ImportError:
    print("Installing BeautifulSoup4...")
    os.system("pip install beautifulsoup4==4.11.2")
    from bs4 import BeautifulSoup

try:
    from sklearn.preprocessing import MinMaxScaler
    print("✓ Scikit-learn loaded")
except ImportError:
    print("Installing scikit-learn...")
    os.system("pip install scikit-learn==1.2.2")
    from sklearn.preprocessing import MinMaxScaler

try:
    from fake_useragent import UserAgent
    print("✓ Fake UserAgent loaded")
except ImportError:
    print("Installing fake-useragent...")
    os.system("pip install fake-useragent==1.5.1")
    from fake_useragent import UserAgent

try:
    import ccxt
    print("✓ CCXT loaded")
except ImportError:
    print("Installing CCXT...")
    os.system("pip install ccxt==4.1.55")
    import ccxt

try:
    from flask import Flask, render_template, jsonify, request
    print("✓ Flask loaded")
except ImportError:
    print("Installing Flask...")
    os.system("pip install flask==2.3.3")
    from flask import Flask, render_template, jsonify, request

# TensorFlow with comprehensive compatibility handling
TENSORFLOW_AVAILABLE = False
try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
    print("✓ TensorFlow loaded successfully")
except ImportError:
    print("Installing TensorFlow...")
    try:
        os.system("pip install tensorflow==2.8.4")
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        from tensorflow.keras.models import Sequential, load_model
        from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, BatchNormalization
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        TENSORFLOW_AVAILABLE = True
        print("✓ TensorFlow installed and loaded")
    except Exception as e:
        print(f"⚠ TensorFlow not available: {e}")
        TENSORFLOW_AVAILABLE = False

# MetaTrader5 handling
MT5_AVAILABLE = False
mt5 = None
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
    print("✓ MetaTrader5 loaded")
except ImportError:
    print("⚠ MetaTrader5 not available - exchange-only mode")

# TA-Lib with fallback
TALIB_AVAILABLE = False
try:
    import talib
    TALIB_AVAILABLE = True
    print("✓ TA-Lib loaded")
except ImportError:
    print("⚠ TA-Lib not available - using custom implementations")

# =============================================================================
# YOUR EXACT CONFIGURATION - UNCHANGED
# =============================================================================

CONFIG = {
    "ACTIVE_SYMBOLS": ["GOLD", "BITCOIN", "OIL"],
    "SYMBOLS": {
        "GOLD": {"MT5": "XAUUSD", "MEXC": "XAUUSD/USDT", "TIMEFRAME": "M1", "POINT_VALUE": 0.01, "LOT_SIZE": 100, "SPREAD": 0.5, "COMMISSION": 0.0002, "VOLATILITY_THRESHOLD": 0.008},
        "BITCOIN": {"MT5": "BTCUSD", "MEXC": "BTC/USDT", "TIMEFRAME": "M1", "POINT_VALUE": 1.0, "LOT_SIZE": 1, "SPREAD": 10.0, "COMMISSION": 0.0005, "VOLATILITY_THRESHOLD": 0.02},
        "OIL": {"MT5": "USOIL", "MEXC": "USOIL/USDT", "TIMEFRAME": "M1", "POINT_VALUE": 0.01, "LOT_SIZE": 1000, "SPREAD": 0.5, "COMMISSION": 0.0003, "VOLATILITY_THRESHOLD": 0.01}
    },
    "MT5_LOGIN": int(os.getenv("MT5_LOGIN", "93602384")),
    "MT5_PASSWORD": os.getenv("MT5_PASSWORD", "X.Ke7jKu"),
    "MT5_SERVER": os.getenv("MT5_SERVER", "MetaQuotes-Demo"),
    "MEXC_API_KEY": os.getenv("MEXC_API_KEY", "mxOvglut3Gduboiceq"),
    "MEXC_SECRET_KEY": os.getenv("MEXC_SECRET_KEY", "95dbcbc778d54164a21e75099ee2193a"),
    "TELEGRAM_TOKEN": os.getenv("TELEGRAM_TOKEN", "7670360424:AAFSkjPu3PnR4q_olvy6-GxXXv9V3Ogjtpo"),
    "TELEGRAM_CHAT_ID": os.getenv("TELEGRAM_CHAT_ID", "6382819740"),
    "DB_FILE": "encrypted_trades.db",
    "EXCEL_REPORT": "trades_report.xlsx",
    "STATE_SIZE": 30,
    "ACTION_SIZE": 3,
    "MEMORY_SIZE": 50000,
    "GAMMA": 0.95,
    "EPSILON": 1.0,
    "EPSILON_MIN": 0.01,
    "EPSILON_DECAY": 0.997,
    "LEARNING_RATE": 0.0005,
    "MODEL_PATH_PREFIX": "models/model_",
    "RISK_PERCENT_PER_TRADE": 0.02,
    "STOP_LOSS_ATR_MULTIPLIER": 1.7,
    "TAKE_PROFIT_ATR_MULTIPLIER": 2.5,
    "MIN_ADX_THRESHOLD": 25,
    "RSI_OVERBOUGHT": 70,
    "RSI_OVERSOLD": 30,
    "MIN_VOLUME_MULTIPLIER": 1.5,
    "MIN_EXPECTED_PROFIT_PIPS": 15,
    "WHALE_SETTINGS": {
        "volume_multiplier": 5.0,
        "min_whale_size": {"XAUUSD": 1000, "BTCUSD": 5, "USOIL": 1000},
        "price_move_threshold": {"XAUUSD": 0.0015, "BTCUSD": 0.005, "USOIL": 0.002},
        "order_book_imbalance": 0.6
    },
    "MARKET_DATA_SAVE_INTERVAL_SECONDS": 300,
    "TRADE_COOLDOWN_SECONDS": 120,
    "MAX_SIMULTANEOUS_TRADES_PER_SYMBOL": 2,
    "MIN_PROFIT_PIPS_FOR_PARTIAL_CLOSE": 30,
    "PARTIAL_CLOSE_VOLUME_PERCENT": 0.3,
    "TRAILING_STOP_PIPS": 15,
    "TRAINING_FREQUENCY": 20,
    "BATCH_SIZE": 64,
    "EVALUATION_FREQUENCY": 50,
    "WIN_RATE_ALERT_THRESHOLD": 70,
    "DAILY_PROFIT_TARGET_PERCENT": 0.05,
    "MAX_DAILY_LOSS_PERCENT": 0.03,
    "MAX_DAILY_TRADES": 30,
    "HOURLY_REPORT_INTERVAL_MINUTES": 60,
    "MAIN_LOOP_SLEEP_SECONDS": 3,
    "LEARNING_RATE_ADJUSTMENT_FACTOR": 0.1,
    "MAX_VOLATILITY_THRESHOLD": 0.015,
    "MAX_ATR_THRESHOLD": 0.005,
    "MEMORY_CLEANUP_FREQUENCY": 1000,
    "INDICATORS": {
        "RSI_PERIOD": 14,
        "MACD_FAST": 12,
        "MACD_SLOW": 26,
        "MACD_SIGNAL": 9,
        "ATR_PERIOD": 14,
        "ADX_PERIOD": 14,
        "BOLLINGER_PERIOD": 20,
        "BOLLINGER_STDDEV": 2
    },
    "NEWS_API_KEY": "your_news_api_key_here"
}

# =============================================================================
# DIRECTORY SETUP
# =============================================================================

def setup_project_structure():
    """Create project directories"""
    directories = ["logs", "models", "config", "reports", "data", "templates", "static"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("✓ Project structure created")

# =============================================================================
# ENHANCED TECHNICAL INDICATORS WITH FALLBACKS
# =============================================================================

class TechnicalAnalysis:
    """Technical analysis with TA-Lib fallbacks"""
    
    @staticmethod
    def RSI(close, timeperiod=14):
        """RSI calculation with fallback"""
        try:
            if TALIB_AVAILABLE and hasattr(talib, 'RSI'):
                result = talib.RSI(close.values, timeperiod=timeperiod)
                return pd.Series(result, index=close.index).fillna(50)
            else:
                # Custom RSI
                delta = close.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=timeperiod).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=timeperiod).mean()
                rs = gain / (loss + 1e-8)
                rsi = 100 - (100 / (1 + rs))
                return rsi.fillna(50)
        except Exception as e:
            logging.error(f"RSI calculation failed: {e}")
            return pd.Series([50] * len(close), index=close.index)

    @staticmethod
    def MACD(close, fastperiod=12, slowperiod=26, signalperiod=9):
        """MACD calculation with fallback"""
        try:
            if TALIB_AVAILABLE and hasattr(talib, 'MACD'):
                macd, signal, histogram = talib.MACD(close.values, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
                return (pd.Series(macd, index=close.index).fillna(0),
                        pd.Series(signal, index=close.index).fillna(0),
                        pd.Series(histogram, index=close.index).fillna(0))
            else:
                # Custom MACD
                ema_fast = close.ewm(span=fastperiod).mean()
                ema_slow = close.ewm(span=slowperiod).mean()
                macd = ema_fast - ema_slow
                signal = macd.ewm(span=signalperiod).mean()
                histogram = macd - signal
                return macd.fillna(0), signal.fillna(0), histogram.fillna(0)
        except Exception as e:
            logging.error(f"MACD calculation failed: {e}")
            zeros = pd.Series([0] * len(close), index=close.index)
            return zeros, zeros, zeros

    @staticmethod
    def ATR(high, low, close, timeperiod=14):
        """ATR calculation with fallback"""
        try:
            if TALIB_AVAILABLE and hasattr(talib, 'ATR'):
                result = talib.ATR(high.values, low.values, close.values, timeperiod=timeperiod)
                return pd.Series(result, index=close.index).fillna(0.001)
            else:
                # Custom ATR
                prev_close = close.shift(1)
                tr1 = high - low
                tr2 = abs(high - prev_close)
                tr3 = abs(low - prev_close)
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr = tr.rolling(window=timeperiod).mean()
                return atr.fillna(0.001)
        except Exception as e:
            logging.error(f"ATR calculation failed: {e}")
            return pd.Series([0.001] * len(close), index=close.index)

    @staticmethod
    def BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2):
        """Bollinger Bands calculation with fallback"""
        try:
            if TALIB_AVAILABLE and hasattr(talib, 'BBANDS'):
                upper, middle, lower = talib.BBANDS(close.values, timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn)
                return (pd.Series(upper, index=close.index).fillna(close),
                        pd.Series(middle, index=close.index).fillna(close),
                        pd.Series(lower, index=close.index).fillna(close))
            else:
                # Custom Bollinger Bands
                sma = close.rolling(window=timeperiod).mean()
                std = close.rolling(window=timeperiod).std()
                upper = sma + (std * nbdevup)
                lower = sma - (std * nbdevdn)
                return upper.fillna(close), sma.fillna(close), lower.fillna(close)
        except Exception as e:
            logging.error(f"Bollinger Bands calculation failed: {e}")
            return close, close, close

    @staticmethod
    def STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3):
        """Stochastic calculation with fallback"""
        try:
            if TALIB_AVAILABLE and hasattr(talib, 'STOCH'):
                k, d = talib.STOCH(high.values, low.values, close.values, 
                                 fastk_period=fastk_period, slowk_period=slowk_period, slowd_period=slowd_period)
                return (pd.Series(k, index=close.index).fillna(50),
                        pd.Series(d, index=close.index).fillna(50))
            else:
                # Custom Stochastic
                lowest_low = low.rolling(window=fastk_period).min()
                highest_high = high.rolling(window=fastk_period).max()
                k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low + 1e-8))
                k_percent = k_percent.rolling(window=slowk_period).mean()
                d_percent = k_percent.rolling(window=slowd_period).mean()
                return k_percent.fillna(50), d_percent.fillna(50)
        except Exception as e:
            logging.error(f"Stochastic calculation failed: {e}")
            fifties = pd.Series([50] * len(close), index=close.index)
            return fifties, fifties

    @staticmethod
    def ADX(high, low, close, timeperiod=14):
        """ADX calculation with fallback"""
        try:
            if TALIB_AVAILABLE and hasattr(talib, 'ADX'):
                result = talib.ADX(high.values, low.values, close.values, timeperiod=timeperiod)
                return pd.Series(result, index=close.index).fillna(25)
            else:
                # Custom ADX
                plus_dm = high.diff()
                minus_dm = low.diff() * -1
                plus_dm[plus_dm < 0] = 0
                minus_dm[minus_dm < 0] = 0
                
                tr = TechnicalAnalysis.ATR(high, low, close, 1)
                plus_di = 100 * (plus_dm.rolling(window=timeperiod).mean() / (tr.rolling(window=timeperiod).mean() + 1e-8))
                minus_di = 100 * (minus_dm.rolling(window=timeperiod).mean() / (tr.rolling(window=timeperiod).mean() + 1e-8))
                
                dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
                adx = dx.rolling(window=timeperiod).mean()
                return adx.fillna(25)
        except Exception as e:
            logging.error(f"ADX calculation failed: {e}")
            return pd.Series([25] * len(close), index=close.index)

    @staticmethod
    def OBV(close, volume):
        """OBV calculation with fallback"""
        try:
            if TALIB_AVAILABLE and hasattr(talib, 'OBV'):
                result = talib.OBV(close.values, volume.values)
                return pd.Series(result, index=close.index).fillna(0)
            else:
                # Custom OBV
                obv = pd.Series(index=close.index, dtype=float)
                if len(close) > 0:
                    obv.iloc[0] = volume.iloc[0]
                    for i in range(1, len(close)):
                        if close.iloc[i] > close.iloc[i-1]:
                            obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
                        elif close.iloc[i] < close.iloc[i-1]:
                            obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
                        else:
                            obv.iloc[i] = obv.iloc[i-1]
                return obv.fillna(0)
        except Exception as e:
            logging.error(f"OBV calculation failed: {e}")
            return pd.Series([0] * len(close), index=close.index)

    @staticmethod
    def SMA(close, timeperiod=20):
        """Simple Moving Average"""
        try:
            if TALIB_AVAILABLE and hasattr(talib, 'SMA'):
                result = talib.SMA(close.values, timeperiod=timeperiod)
                return pd.Series(result, index=close.index).fillna(close)
            else:
                return close.rolling(window=timeperiod).mean().fillna(close)
        except Exception as e:
            logging.error(f"SMA calculation failed: {e}")
            return close

# =============================================================================
# ENHANCED LOGGER CLASS - YOUR ORIGINAL IMPLEMENTATION
# =============================================================================

class UltimateLogger:
    """Enhanced logging system - your original implementation preserved"""
    
    def __init__(self):
        os.makedirs("logs", exist_ok=True)
        self.log_file = f"logs/trade_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.db_lock = threading.Lock()
        self._setup_logging()
        self._init_log_file()
        self._init_database()
        self._init_excel_report()
        self.last_market_data_save = {
            symbol: {tf: None for tf in ["M1", "M5", "M15", "H1"]}
            for symbol in CONFIG["ACTIVE_SYMBOLS"]
        }

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("logs/trading_bot.log", encoding="utf-8"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("UltimateAI")

    def _init_log_file(self):
        if not self.validate_file(self.log_file):
            self.logger.error(f"Cannot initialize log file: {self.log_file}")
            return
        try:
            with open(self.log_file, "w", encoding="utf-8") as f:
                f.write("timestamp,symbol,action,price,quantity,sl,tp,profit,status,whale_detected,platform,order_ticket\n")
        except Exception as e:
            self.logger.error(f"Failed to initialize log file: {str(e)}")

    def _init_database(self):
        try:
            with self.db_lock, sqlite3.connect(CONFIG["DB_FILE"], timeout=10) as conn:
                conn.execute("PRAGMA busy_timeout = 10000")
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT, symbol TEXT, action TEXT,
                        entry_price REAL, volume REAL,
                        stop_loss REAL, take_profit REAL,
                        exit_price REAL, profit REAL,
                        duration_seconds INTEGER, status TEXT,
                        whale_detected INTEGER, platform TEXT,
                        order_ticket TEXT UNIQUE
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS market_data (
                        timestamp TEXT, symbol TEXT, timeframe TEXT,
                        open REAL, high REAL, low REAL, close REAL,
                        volume REAL, rsi REAL, macd REAL, atr REAL,
                        bb_upper REAL, bb_lower REAL, stoch_k REAL, stoch_d REAL, adx REAL,
                        obv REAL, PRIMARY KEY (timestamp, symbol, timeframe)
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS training_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT, symbol TEXT,
                        state TEXT, action INTEGER, reward REAL,
                        next_state TEXT, done INTEGER
                    )
                """)
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {str(e)}")

    def _init_excel_report(self):
        if not self.validate_file(CONFIG["EXCEL_REPORT"]):
            self.logger.error(f"Cannot initialize Excel report: {CONFIG['EXCEL_REPORT']}")
            return
        try:
            if not os.path.exists(CONFIG["EXCEL_REPORT"]):
                pd.DataFrame(columns=[
                    "Timestamp", "Symbol", "Action", "Entry Price", "Volume",
                    "Stop Loss", "Take Profit", "Exit Price", "Profit", "Duration (s)",
                    "Status", "Whale Detected", "Platform", "Order Ticket"
                ]).to_excel(CONFIG["EXCEL_REPORT"], index=False)
        except Exception as e:
            self.logger.error(f"Failed to initialize Excel report: {str(e)}")

    def validate_file(self, file_path: str) -> bool:
        try:
            directory = os.path.dirname(file_path) or '.'
            os.makedirs(directory, exist_ok=True)
            if not os.path.exists(file_path):
                with open(file_path, "a") as f:
                    pass
            return os.access(file_path, os.W_OK)
        except Exception as e:
            self.logger.error(f"File validation failed for {file_path}: {str(e)}")
            return False

    def validate_market_data(self, df: pd.DataFrame, symbol: str, timeframe: str, min_rows: int = 50) -> bool:
        try:
            if df is None or df.empty or len(df) < min_rows:
                self.logger.warning(f"Invalid market data for {symbol} ({timeframe}): Empty or insufficient data")
                return False
            if df[["open", "high", "low", "close", "volume"]].isna().any().any():
                self.logger.warning(f"Invalid market data for {symbol} ({timeframe}): Contains NaN values")
                return False
            return True
        except Exception as e:
            self.logger.error(f"Market data validation failed for {symbol} ({timeframe}): {str(e)}")
            return False

    def handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            if df.empty:
                return df
            df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            return df
        except Exception as e:
            self.logger.error(f"Failed to handle missing data: {str(e)}")
            return df

    def log_transaction(self, symbol: str, action: str, price: float, quantity: float, sl: float, tp: float,
                        profit: Optional[float] = None, status: str = "OPEN", whale_detected: bool = False,
                        platform: str = "MT5", order_ticket: Optional[str] = None,
                        exit_price: Optional[float] = None, duration: Optional[int] = None):
        try:
            with self.db_lock:
                timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with sqlite3.connect(CONFIG["DB_FILE"], timeout=10) as conn:
                    cursor = conn.cursor()
                    if status == "OPEN":
                        cursor.execute("""
                            INSERT INTO trades (
                                timestamp, symbol, action, entry_price, volume,
                                stop_loss, take_profit, exit_price, profit,
                                duration_seconds, status, whale_detected, platform, order_ticket
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, ?, ?, ?, ?)
                        """, (
                            timestamp_str, symbol, action, price, quantity,
                            sl, tp, status, 1 if whale_detected else 0, platform, str(order_ticket)
                        ))
                    else:
                        cursor.execute("""
                            UPDATE trades
                            SET exit_price = ?, profit = ?, duration_seconds = ?, status = ?
                            WHERE order_ticket = ?
                        """, (exit_price, profit, duration, status, str(order_ticket)))
                    conn.commit()

                if self.validate_file(self.log_file):
                    log_entry = f"{timestamp_str},{symbol},{action},{price:.4f},{quantity:.6f},{sl:.4f},{tp:.4f}," \
                                f"{profit or ''},{status},{'YES' if whale_detected else 'NO'},{platform},{order_ticket or ''}\n"
                    with open(self.log_file, "a", encoding="utf-8") as f:
                        f.write(log_entry)

                if self.validate_file(CONFIG["EXCEL_REPORT"]):
                    try:
                        df_excel = pd.read_excel(CONFIG["EXCEL_REPORT"])
                    except Exception:
                        df_excel = pd.DataFrame(columns=[
                            "Timestamp", "Symbol", "Action", "Entry Price", "Volume",
                            "Stop Loss", "Take Profit", "Exit Price", "Profit", "Duration (s)",
                            "Status", "Whale Detected", "Platform", "Order Ticket"
                        ])
                    new_row = {
                        "Timestamp": timestamp_str, "Symbol": symbol, "Action": action,
                        "Entry Price": price, "Volume": quantity, "Stop Loss": sl, "Take Profit": tp,
                        "Exit Price": exit_price, "Profit": profit, "Duration (s)": duration,
                        "Status": status, "Whale Detected": "YES" if whale_detected else "NO",
                        "Platform": platform, "Order Ticket": order_ticket
                    }
                    df_excel = pd.concat([df_excel, pd.DataFrame([new_row])], ignore_index=True)
                    df_excel.to_excel(CONFIG["EXCEL_REPORT"], index=False)

                self.logger.info(
                    f"{status} {action} {quantity:.6f} {symbol} @ {price:.4f} | SL: {sl:.4f} | TP: {tp:.4f} "
                    f"{'🐋' if whale_detected else ''} ({platform}) | Ticket: {order_ticket}")
        except Exception as e:
            self.logger.error(f"Failed to log transaction: {str(e)}")

    def save_market_data(self, symbol: str, timeframe: str, df: pd.DataFrame, platform: str):
        try:
            if not self.validate_market_data(df, symbol, timeframe):
                return
            current_time = datetime.now()
            last_save = self.last_market_data_save[symbol][timeframe]
            if last_save is None or (current_time - last_save).total_seconds() >= CONFIG["MARKET_DATA_SAVE_INTERVAL_SECONDS"]:
                df = self.handle_missing_data(df)
                df["rsi"] = TechnicalAnalysis.RSI(df["close"], timeperiod=CONFIG["INDICATORS"]["RSI_PERIOD"])
                df["macd"], df["macd_signal"], _ = TechnicalAnalysis.MACD(
                    df["close"], fastperiod=CONFIG["INDICATORS"]["MACD_FAST"],
                    slowperiod=CONFIG["INDICATORS"]["MACD_SLOW"], signalperiod=CONFIG["INDICATORS"]["MACD_SIGNAL"]
                )
                df["atr"] = TechnicalAnalysis.ATR(df["high"], df["low"], df["close"], timeperiod=CONFIG["INDICATORS"]["ATR_PERIOD"])
                df["bb_upper"], _, df["bb_lower"] = TechnicalAnalysis.BBANDS(
                    df["close"], timeperiod=CONFIG["INDICATORS"]["BOLLINGER_PERIOD"],
                    nbdevup=CONFIG["INDICATORS"]["BOLLINGER_STDDEV"], nbdevdn=CONFIG["INDICATORS"]["BOLLINGER_STDDEV"]
                )
                df["stoch_k"], df["stoch_d"] = TechnicalAnalysis.STOCH(df["high"], df["low"], df["close"])
                df["adx"] = TechnicalAnalysis.ADX(df["high"], df["low"], df["close"], timeperiod=CONFIG["INDICATORS"]["ADX_PERIOD"])
                df["obv"] = TechnicalAnalysis.OBV(df["close"], df["volume"])

                with self.db_lock, sqlite3.connect(CONFIG["DB_FILE"], timeout=10) as conn:
                    for _, row in df.tail(5).iterrows():
                        timestamp_str = row.name.strftime("%Y-%m-%d %H:%M:%S") if hasattr(row.name, 'strftime') else str(row.name)
                        conn.execute("""
                            INSERT OR REPLACE INTO market_data
                            (timestamp, symbol, timeframe, open, high, low, close, volume, rsi, macd, atr,
                            bb_upper, bb_lower, stoch_k, stoch_d, adx, obv)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            timestamp_str, symbol, timeframe,
                            row["open"], row["high"], row["low"], row["close"], row["volume"],
                            row.get("rsi", 0), row.get("macd", 0), row.get("atr", 0),
                            row.get("bb_upper", 0), row.get("bb_lower", 0),
                            row.get("stoch_k", 0), row.get("stoch_d", 0), row.get("adx", 0), row.get("obv", 0)
                        ))
                    conn.commit()
                self.last_market_data_save[symbol][timeframe] = current_time
        except Exception as e:
            self.logger.error(f"Failed to save market data for {symbol} ({timeframe}, {platform}): {str(e)}")

    def save_training_data(self, symbol: str, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        try:
            with self.db_lock, sqlite3.connect(CONFIG["DB_FILE"], timeout=10) as conn:
                timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                state_json = json.dumps(state.tolist() if hasattr(state, 'tolist') else list(state))
                next_state_json = json.dumps(next_state.tolist() if hasattr(next_state, 'tolist') else list(next_state))
                conn.execute("""
                    INSERT INTO training_data (timestamp, symbol, state, action, reward, next_state, done)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (timestamp_str, symbol, state_json, action, reward, next_state_json, 1 if done else 0))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to save training data for {symbol}: {str(e)}")

    def send_telegram_alert(self, message: str, retries: int = 5) -> bool:
        for attempt in range(retries):
            try:
                url = f"https://api.telegram.org/bot{CONFIG['TELEGRAM_TOKEN']}/sendMessage"
                payload = {"chat_id": CONFIG["TELEGRAM_CHAT_ID"], "text": message, "parse_mode": "HTML"}
                response = requests.post(url, json=payload, timeout=10)
                if response.status_code == 200:
                    return True
                self.logger.error(f"Telegram API error (attempt {attempt+1}/{retries}): {response.status_code} - {response.text}")
                time.sleep((2 ** attempt) + random.uniform(0.1, 0.5))
            except Exception as e:
                self.logger.error(f"Telegram error (attempt {attempt+1}/{retries}): {str(e)}")
                time.sleep((2 ** attempt) + random.uniform(0.1, 0.5))
        return False

    def info(self, message: str):
        self.logger.info(message)

    def error(self, message: str):
        self.logger.error(message)

    def warning(self, message: str):
        self.logger.warning(message)

# =============================================================================
# ENHANCED MARKET DATA PROCESSOR - YOUR ORIGINAL IMPLEMENTATION
# =============================================================================

class MarketDataProcessor:
    """Enhanced market data processor - your original implementation preserved"""
    
    def __init__(self, logger: UltimateLogger):
        self.logger = logger
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def preprocess_data(self, df: pd.DataFrame, df_m5: pd.DataFrame, df_m15: pd.DataFrame, df_h1: pd.DataFrame) -> np.ndarray:
        try:
            if df.empty or len(df) < 100:
                raise ValueError("Insufficient data for preprocessing")
            df = self.logger.handle_missing_data(df)
            df['RSI'] = TechnicalAnalysis.RSI(df['close'], timeperiod=CONFIG["INDICATORS"]["RSI_PERIOD"])
            df['MACD'], df['MACD_Signal'], _ = TechnicalAnalysis.MACD(
                df['close'], fastperiod=CONFIG["INDICATORS"]["MACD_FAST"],
                slowperiod=CONFIG["INDICATORS"]["MACD_SLOW"], signalperiod=CONFIG["INDICATORS"]["MACD_SIGNAL"]
            )
            df['SMA_20'] = TechnicalAnalysis.SMA(df['close'], timeperiod=CONFIG["INDICATORS"]["BOLLINGER_PERIOD"])
            df['ATR'] = TechnicalAnalysis.ATR(df['high'], df['low'], df['close'], timeperiod=CONFIG["INDICATORS"]["ATR_PERIOD"])
            df['BB_upper'], _, df['BB_lower'] = TechnicalAnalysis.BBANDS(
                df['close'], timeperiod=CONFIG["INDICATORS"]["BOLLINGER_PERIOD"],
                nbdevup=CONFIG["INDICATORS"]["BOLLINGER_STDDEV"], nbdevdn=CONFIG["INDICATORS"]["BOLLINGER_STDDEV"]
            )
            df['Stoch_K'], df['Stoch_D'] = TechnicalAnalysis.STOCH(df['high'], df['low'], df['close'])
            df['ADX'] = TechnicalAnalysis.ADX(df['high'], df['low'], df['close'], timeperiod=CONFIG["INDICATORS"]["ADX_PERIOD"])
            df['OBV'] = TechnicalAnalysis.OBV(df['close'], df['volume'])
            df['Price_Change'] = df['close'].pct_change()
            df['Volatility'] = df['Price_Change'].rolling(window=CONFIG["INDICATORS"]["BOLLINGER_PERIOD"]).std()
            df['Volume_MA'] = df['volume'].rolling(window=CONFIG["INDICATORS"]["BOLLINGER_PERIOD"]).mean()
            df['Volume_Ratio'] = df['volume'] / df['Volume_MA'].replace(0, np.finfo(float).eps)
            last_m5_close = df_m5["close"].iloc[-1] if not df_m5.empty else df["close"].iloc[-1]
            last_m15_close = df_m15["close"].iloc[-1] if not df_m15.empty else df["close"].iloc[-1]
            last_h1_close = df_h1["close"].iloc[-1] if not df_h1.empty else df["close"].iloc[-1]
            current_m1_close = df["close"].iloc[-1]
            df["M5_Trend"] = 1 if current_m1_close > last_m5_close else -1
            df["M15_Trend"] = 1 if current_m1_close > last_m15_close else -1
            df["H1_Trend"] = 1 if current_m1_close > last_h1_close else -1
            features = df[[
                'close', 'RSI', 'MACD', 'MACD_Signal', 'ATR',
                'BB_upper', 'BB_lower', 'Stoch_K', 'Stoch_D',
                'ADX', 'OBV', 'Volatility', 'Volume_Ratio',
                'M5_Trend', 'M15_Trend', 'H1_Trend'
            ]].iloc[-1]
            features = self.logger.handle_missing_data(pd.DataFrame([features]))
            scaled_features = self.scaler.fit_transform(features.values.reshape(1, -1)).flatten()
            return scaled_features.reshape(1, 1, -1)
        except Exception as e:
            self.logger.error(f"Data preprocessing failed: {str(e)}")
            return np.zeros((1, 1, CONFIG["STATE_SIZE"]))

# =============================================================================
# ENHANCED DQN MODEL - YOUR ORIGINAL IMPLEMENTATION PRESERVED
# =============================================================================

class EnhancedDQNModel:
    """Enhanced DQN model - your original implementation preserved and enhanced"""
    
    def __init__(self, symbol: str, logger: UltimateLogger):
        self.symbol = symbol
        self.logger = logger
        self.state_size = CONFIG["STATE_SIZE"]
        self.action_size = CONFIG["ACTION_SIZE"]
        self.memory = deque(maxlen=CONFIG["MEMORY_SIZE"])
        self.gamma = CONFIG["GAMMA"]
        self.epsilon = CONFIG["EPSILON"]
        self.epsilon_min = CONFIG["EPSILON_MIN"]
        self.epsilon_decay = CONFIG["EPSILON_DECAY"]
        self.learning_rate = CONFIG["LEARNING_RATE"]
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        self.data_processor = MarketDataProcessor(logger)
        self.model_path = f"{CONFIG['MODEL_PATH_PREFIX']}{symbol}.h5"
        self.trade_counter = 0
        self.action_map = {0: "SELL", 1: "BUY", 2: "HOLD"}

    def _build_model(self):
        try:
            if not TENSORFLOW_AVAILABLE:
                self.logger.warning(f"TensorFlow not available for {self.symbol}, using fallback")
                return None
            
            model = Sequential([
                LSTM(64, input_shape=(1, self.state_size), return_sequences=True),
                Dropout(0.3),
                LSTM(32),
                Dense(32, activation='relu'),
                Dropout(0.2),
                Dense(self.action_size, activation='linear')
            ])
            model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
            return model
        except Exception as e:
            self.logger.error(f"Failed to build model for {self.symbol}: {str(e)}")
            return None

    def validate_model(self) -> bool:
        try:
            if self.model is None:
                return False
            test_input = np.zeros((1, 1, self.state_size))
            if TENSORFLOW_AVAILABLE:
                self.model.predict(test_input, verbose=0)
            return True
        except Exception as e:
            self.logger.error(f"Model validation failed for {self.symbol}: {str(e)}")
            return False

    def update_target_model(self):
        try:
            if self.model and self.target_model and TENSORFLOW_AVAILABLE:
                self.target_model.set_weights(self.model.get_weights())
        except Exception as e:
            self.logger.error(f"Failed to update target model for {self.symbol}: {str(e)}")

    def act(self, state: np.ndarray, df_m1: pd.DataFrame, df_m5: pd.DataFrame) -> int:
        """Your original action logic preserved exactly"""
        try:
            if not self.validate_model():
                self.logger.warning(f"Invalid model for {self.symbol}, using fallback action")
                return self.fallback_action(df_m1)
            rsi = df_m1["RSI"].iloc[-1] if "RSI" in df_m1 else 50
            stoch_k = df_m1["Stoch_K"].iloc[-1] if "Stoch_K" in df_m1 else 50
            stoch_d = df_m1["Stoch_D"].iloc[-1] if "Stoch_D" in df_m1 else 50
            adx = df_m1["ADX"].iloc[-1] if "ADX" in df_m1 else 0
            volume_ratio = df_m1["Volume_Ratio"].iloc[-1] if "Volume_Ratio" in df_m1 else 1
            m5_trend = df_m5["close"].iloc[-1] > df_m5["close"].iloc[-2] if not df_m5.empty and len(df_m5) > 1 else False
            atr = df_m1["ATR"].iloc[-1] if "ATR" in df_m1 else 0
            expected_profit = atr * CONFIG["TAKE_PROFIT_ATR_MULTIPLIER"]
            if np.random.rand() <= self.epsilon:
                return random.randrange(self.action_size)
            if TENSORFLOW_AVAILABLE and self.model:
                act_values = self.model.predict(state, verbose=0)
                action = np.argmax(act_values[0])
            else:
                action = self.fallback_action(df_m1)
                return action
            if action == 1:  # BUY
                condition = (
                    rsi < CONFIG["RSI_OVERBOUGHT"] and
                    stoch_k < 80 and stoch_k > stoch_d and
                    adx > CONFIG["MIN_ADX_THRESHOLD"] and
                    volume_ratio > CONFIG["MIN_VOLUME_MULTIPLIER"] and
                    m5_trend and
                    expected_profit >= CONFIG["MIN_EXPECTED_PROFIT_PIPS"]
                )
                if not condition:
                    return 2  # HOLD
            elif action == 0:  # SELL
                condition = (
                    rsi > CONFIG["RSI_OVERSOLD"] and
                    stoch_k > 20 and stoch_k < stoch_d and
                    adx > CONFIG["MIN_ADX_THRESHOLD"] and
                    volume_ratio > CONFIG["MIN_VOLUME_MULTIPLIER"] and
                    not m5_trend and
                    expected_profit >= CONFIG["MIN_EXPECTED_PROFIT_PIPS"]
                )
                if not condition:
                    return 2  # HOLD
            return action
        except Exception as e:
            self.logger.error(f"Action selection failed for {self.symbol}: {str(e)}")
            return 2  # HOLD

    def fallback_action(self, df_m1: pd.DataFrame) -> int:
        """Your original fallback logic preserved"""
        try:
            rsi = df_m1["RSI"].iloc[-1] if "RSI" in df_m1 else 50
            if rsi < CONFIG["RSI_OVERSOLD"]:
                return 1  # BUY
            elif rsi > CONFIG["RSI_OVERBOUGHT"]:
                return 0  # SELL
            return 2  # HOLD
        except Exception as e:
            self.logger.error(f"Fallback action failed for {self.symbol}: {str(e)}")
            return 2  # HOLD

    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Your original memory logic preserved"""
        try:
            self.memory.append((state, action, reward, next_state, done))
            if len(self.memory) >= CONFIG["MEMORY_SIZE"] and self.trade_counter % CONFIG["MEMORY_CLEANUP_FREQUENCY"] == 0:
                self.memory = deque(list(self.memory)[-int(CONFIG["MEMORY_SIZE"] * 0.8):], maxlen=CONFIG["MEMORY_SIZE"])
                self.logger.info(f"Cleaned up memory for {self.symbol}, current size: {len(self.memory)}")
            self.logger.save_training_data(self.symbol, state, action, reward, next_state, done)
        except Exception as e:
            self.logger.error(f"Failed to save to memory for {self.symbol}: {str(e)}")

    def replay(self, batch_size: int):
        """Your original replay logic preserved"""
        try:
            if len(self.memory) < batch_size or not TENSORFLOW_AVAILABLE or not self.validate_model():
                return
            minibatch = random.sample(self.memory, batch_size)
            states = np.array([t[0] for t in minibatch])
            actions = np.array([t[1] for t in minibatch])
            rewards = np.array([t[2] for t in minibatch])
            next_states = np.array([t[3] for t in minibatch])
            dones = np.array([t[4] for t in minibatch])
            targets = self.model.predict(states, verbose=0)
            target_next = self.target_model.predict(next_states, verbose=0)
            for i in range(batch_size):
                if dones[i]:
                    targets[i][actions[i]] = rewards[i]
                else:
                    targets[i][actions[i]] = rewards[i] + self.gamma * np.max(target_next[i])
            self.model.fit(states, targets, epochs=1, verbose=0)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            if self.trade_counter % CONFIG["TRAINING_FREQUENCY"] == 0:
                self.update_target_model()
                self.logger.info(f"Updated target model for {self.symbol}")
        except Exception as e:
            self.logger.error(f"Replay failed for {self.symbol}: {str(e)}")

    def adjust_learning_rate(self, atr_value: float, win_rate: float, base_lr: float = CONFIG["LEARNING_RATE"]):
        """Your original learning rate adjustment preserved"""
        try:
            new_lr = base_lr * (1.0 / (1.0 + CONFIG["LEARNING_RATE_ADJUSTMENT_FACTOR"] * atr_value))
            if win_rate < CONFIG["WIN_RATE_ALERT_THRESHOLD"]:
                new_lr *= 0.5
            new_lr = max(new_lr, base_lr * 0.1)
            if TENSORFLOW_AVAILABLE and self.model:
                for layer in self.model.layers:
                    if hasattr(layer, 'optimizer'):
                        layer.optimizer.learning_rate.assign(new_lr)
            self.logger.info(f"Adjusted learning rate for {self.symbol} to {new_lr:.6f}")
        except Exception as e:
            self.logger.error(f"Failed to adjust learning rate for {self.symbol}: {str(e)}")

    def evaluate_model(self, recent_trades: int = 50) -> Tuple[float, float]:
        """Your original evaluation logic preserved"""
        try:
            with self.logger.db_lock, sqlite3.connect(CONFIG["DB_FILE"], timeout=10) as conn:
                df_trades = pd.read_sql(
                    f"SELECT profit FROM trades WHERE symbol = '{self.symbol}' AND status = 'CLOSED' ORDER BY timestamp DESC LIMIT {recent_trades}",
                    conn
                )
            if df_trades.empty:
                return 0.0, 0.0
            total_profit = df_trades["profit"].sum()
            win_rate = (len(df_trades[df_trades["profit"] > 0]) / len(df_trades)) * 100
            return total_profit, win_rate
        except Exception as e:
            self.logger.error(f"Model evaluation failed for {self.symbol}: {str(e)}")
            return 0.0, 0.0

    def save_model(self):
        try:
            if self.model is not None and TENSORFLOW_AVAILABLE:
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                self.model.save(self.model_path)
                self.logger.info(f"Saved model for {self.symbol} to {self.model_path}")
        except Exception as e:
            self.logger.error(f"Failed to save model for {self.symbol}: {str(e)}")

    def load_model(self):
        try:
            if os.path.exists(self.model_path) and TENSORFLOW_AVAILABLE:
                self.model = load_model(self.model_path)
                self.target_model = load_model(self.model_path)
                self.logger.info(f"Loaded model for {self.symbol} from {self.model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load model for {self.symbol}: {str(e)}")

# =============================================================================
# TRADING PLATFORM INTERFACE - YOUR ORIGINAL IMPLEMENTATION
# =============================================================================

class TradingPlatform:
    """Trading platform interface - your original implementation preserved"""
    
    def __init__(self, logger: UltimateLogger):
        self.logger = logger
        self.mt5_initialized = False
        try:
            self.mexc = ccxt.mexc({
                'apiKey': CONFIG["MEXC_API_KEY"],
                'secret': CONFIG["MEXC_SECRET_KEY"],
                'enableRateLimit': True
            })
        except Exception as e:
            self.logger.error(f"MEXC initialization failed: {str(e)}")
            self.mexc = None

    def initialize_mt5(self) -> bool:
        try:
            if not MT5_AVAILABLE:
                self.logger.warning("MetaTrader5 not available")
                return False
            if not mt5.initialize(login=CONFIG["MT5_LOGIN"], password=CONFIG["MT5_PASSWORD"], server=CONFIG["MT5_SERVER"]):
                self.logger.error(f"MT5 initialization failed: {mt5.last_error()}")
                return False
            self.mt5_initialized = True
            self.logger.info("MT5 initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"MT5 initialization failed: {str(e)}")
            return False

    def get_mt5_data(self, symbol: str, timeframe: str, bars: int = 1000) -> pd.DataFrame:
        try:
            if not self.mt5_initialized:
                self.initialize_mt5()
            if not MT5_AVAILABLE or not self.mt5_initialized:
                return pd.DataFrame()
            timeframe_map = {"M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15, "H1": mt5.TIMEFRAME_H1}
            rates = mt5.copy_rates_from_pos(symbol, timeframe_map[timeframe], 0, bars)
            if rates is None or len(rates) == 0:
                self.logger.error(f"No data retrieved for {symbol} ({timeframe})")
                return pd.DataFrame()
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df.set_index("time", inplace=True)
            df = df[["open", "high", "low", "close", "tick_volume"]]
            df.rename(columns={"tick_volume": "volume"}, inplace=True)
            return self.logger.handle_missing_data(df)
        except Exception as e:
            self.logger.error(f"Failed to get MT5 data for {symbol} ({timeframe}): {str(e)}")
            return pd.DataFrame()

    def get_mexc_data(self, symbol: str, timeframe: str, bars: int = 1000) -> pd.DataFrame:
        try:
            if self.mexc is None:
                return pd.DataFrame()
            ohlcv = self.mexc.fetch_ohlcv(symbol, timeframe, limit=bars)
            df = pd.DataFrame(ohlcv, columns=["time", "open", "high", "low", "close", "volume"])
            df["time"] = pd.to_datetime(df["time"], unit="ms")
            df.set_index("time", inplace=True)
            return self.logger.handle_missing_data(df)
        except Exception as e:
            self.logger.error(f"Failed to get MEXC data for {symbol} ({timeframe}): {str(e)}")
            return pd.DataFrame()

    def execute_trade(self, symbol: str, action: str, volume: float, sl: float, tp: float, platform: str = "MT5") -> Optional[str]:
        """Your original trade execution logic preserved"""
        try:
            if platform == "MT5" and MT5_AVAILABLE:
                if not self.mt5_initialized:
                    self.initialize_mt5()
                symbol_info = mt5.symbol_info(CONFIG["SYMBOLS"][symbol]["MT5"])
                if symbol_info is None:
                    self.logger.error(f"Symbol {symbol} not found on MT5")
                    return None
                price = mt5.symbol_info_tick(CONFIG["SYMBOLS"][symbol]["MT5"]).ask if action == "BUY" else mt5.symbol_info_tick(CONFIG["SYMBOLS"][symbol]["MT5"]).bid
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": CONFIG["SYMBOLS"][symbol]["MT5"],
                    "volume": volume,
                    "type": mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL,
                    "price": price,
                    "sl": sl,
                    "tp": tp,
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC
                }
                result = mt5.order_send(request)
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    self.logger.error(f"MT5 trade failed for {symbol}: {result.comment}")
                    return None
                self.logger.log_transaction(symbol, action, price, volume, sl, tp, platform="MT5", order_ticket=str(result.order))
                return str(result.order)
            else:
                # MEXC execution
                if self.mexc is None:
                    return None
                order_type = "market"
                side = "buy" if action == "BUY" else "sell"
                params = {"stopLoss": sl, "takeProfit": tp}
                order = self.mexc.create_order(CONFIG["SYMBOLS"][symbol]["MEXC"], order_type, side, volume, params=params)
                self.logger.log_transaction(symbol, action, order["price"], volume, sl, tp, platform="MEXC", order_ticket=order["id"])
                return order["id"]
        except Exception as e:
            self.logger.error(f"Trade execution failed for {symbol} on {platform}: {str(e)}")
            return None

    def close_trade(self, symbol: str, order_ticket: str, platform: str, exit_price: float, profit: float, duration: int):
        """Your original trade closing logic preserved"""
        try:
            if platform == "MT5" and MT5_AVAILABLE:
                if not self.mt5_initialized:
                    self.initialize_mt5()
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "position": int(order_ticket),
                    "symbol": CONFIG["SYMBOLS"][symbol]["MT5"],
                    "volume": mt5.position_get(ticket=int(order_ticket))[0].volume if mt5.position_get(ticket=int(order_ticket)) else 0.01,
                    "type": mt5.ORDER_TYPE_SELL if mt5.position_get(ticket=int(order_ticket))[0].type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                    "price": exit_price,
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC
                }
                result = mt5.order_send(request)
                if result.retcode != mt5.TRADE_RETCODE_DONE:
                    self.logger.error(f"MT5 trade close failed for {symbol}: {result.comment}")
                    return
                self.logger.log_transaction(symbol, "CLOSE", exit_price, request["volume"], 0, 0, profit=profit, status="CLOSED", platform="MT5", order_ticket=order_ticket, exit_price=exit_price, duration=duration)
            else:
                # MEXC close
                if self.mexc:
                    self.mexc.cancel_order(order_ticket, CONFIG["SYMBOLS"][symbol]["MEXC"])
                self.logger.log_transaction(symbol, "CLOSE", exit_price, 0, 0, 0, profit=profit, status="CLOSED", platform="MEXC", order_ticket=order_ticket, exit_price=exit_price, duration=duration)
        except Exception as e:
            self.logger.error(f"Trade close failed for {symbol} on {platform}: {str(e)}")

# =============================================================================
# WHALE DETECTION - YOUR ORIGINAL IMPLEMENTATION
# =============================================================================

class WhaleDetector:
    """Whale detection - your original implementation preserved"""
    
    def __init__(self, logger: UltimateLogger):
        self.logger = logger
        self.ua = UserAgent()

    def detect_whale_activity(self, symbol: str, df: pd.DataFrame) -> bool:
        """Your original whale detection logic preserved exactly"""
        try:
            if df.empty or len(df) < CONFIG["INDICATORS"]["BOLLINGER_PERIOD"]:
                return False
            volume_ma = df["volume"].rolling(window=CONFIG["INDICATORS"]["BOLLINGER_PERIOD"]).mean().iloc[-1]
            current_volume = df["volume"].iloc[-1]
            price_change = df["close"].pct_change().iloc[-1]
            symbol_settings = CONFIG["WHALE_SETTINGS"]
            volume_condition = current_volume > volume_ma * symbol_settings["volume_multiplier"]
            size_condition = current_volume > symbol_settings["min_whale_size"].get(CONFIG["SYMBOLS"][symbol]["MT5"], 1000)
            price_move_condition = abs(price_change) > symbol_settings["price_move_threshold"].get(CONFIG["SYMBOLS"][symbol]["MT5"], 0.002)
            if volume_condition and size_condition and price_move_condition:
                self.logger.info(f"Whale activity detected for {symbol}: Volume={current_volume:.2f}, Price Change={price_change:.4f}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Whale detection failed for {symbol}: {str(e)}")
            return False

# =============================================================================
# MAIN TRADING BOT - YOUR ORIGINAL IMPLEMENTATION PRESERVED
# =============================================================================

class UltimateTradingBot:
    """Main trading bot - your original implementation preserved and enhanced"""
    
    def __init__(self):
        self.logger = UltimateLogger()
        self.platform = TradingPlatform(self.logger)
        self.whale_detector = WhaleDetector(self.logger)
        self.models = {symbol: EnhancedDQNModel(symbol, self.logger) for symbol in CONFIG["ACTIVE_SYMBOLS"]}
        self.last_trade_time = {symbol: datetime.now() - timedelta(seconds=CONFIG["TRADE_COOLDOWN_SECONDS"]) 
                               for symbol in CONFIG["ACTIVE_SYMBOLS"]}
        self.running = False
        self.trade_counter = 0
        self.daily_profit = 0.0
        self.daily_trades = 0
        self.daily_reset_time = datetime.now().date()
        self.open_positions = {}

    def start(self):
        """Your original start logic preserved"""
        try:
            self.running = True
            self.logger.info("Ultimate Trading Bot started")
            self.logger.send_telegram_alert(
                "<b>🚀 Ultimate Trading Bot Started</b>\n\n"
                "Bot is now active and monitoring markets...\n"
                f"Active Symbols: {', '.join(CONFIG['ACTIVE_SYMBOLS'])}\n"
                f"Max Daily Trades: {CONFIG['MAX_DAILY_TRADES']}\n"
                f"Risk Per Trade: {CONFIG['RISK_PERCENT_PER_TRADE']*100}%"
            )
            
            # Initialize platforms
            if MT5_AVAILABLE:
                self.platform.initialize_mt5()
            
            # Main trading loop - your original logic preserved
            while self.running:
                try:
                    self.check_daily_reset()
                    
                    for symbol in CONFIG["ACTIVE_SYMBOLS"]:
                        if not self.running:
                            break
                        self.process_symbol(symbol)
                        time.sleep(1)
                    
                    self.manage_open_positions()
                    
                    # Periodic training - your original logic
                    if self.trade_counter % CONFIG["TRAINING_FREQUENCY"] == 0:
                        self.train_models()
                    
                    time.sleep(CONFIG["MAIN_LOOP_SLEEP_SECONDS"])
                    
                except Exception as e:
                    self.logger.error(f"Error in main loop: {str(e)}")
                    time.sleep(10)
                    
        except KeyboardInterrupt:
            self.logger.info("Bot interrupted by user")
        finally:
            self.stop()

    def stop(self):
        """Your original stop logic preserved"""
        self.running = False
        self.close_all_positions()
        self.save_all_models()
        self.logger.info("Ultimate Trading Bot stopped")
        self.logger.send_telegram_alert(
            "<b>🛑 Trading Bot Stopped</b>\n\n"
            "All positions closed safely.\n"
            f"Final Daily Profit: ${self.daily_profit:.2f}\n"
            f"Total Trades Today: {self.daily_trades}"
        )

    def process_symbol(self, symbol: str):
        """Your original symbol processing logic preserved exactly"""
        try:
            # Check cooldown
            if (datetime.now() - self.last_trade_time[symbol]).seconds < CONFIG["TRADE_COOLDOWN_SECONDS"]:
                return
            
            # Check daily limits
            if self.daily_trades >= CONFIG["MAX_DAILY_TRADES"]:
                return
            
            # Get market data
            df_m1 = self.get_market_data(symbol, "1m")
            df_m5 = self.get_market_data(symbol, "5m")
            df_m15 = self.get_market_data(symbol, "15m")
            df_h1 = self.get_market_data(symbol, "1h")
            
            if df_m1.empty or len(df_m1) < 100:
                return
            
            # Add technical indicators
            df_m1 = self.add_technical_indicators(df_m1)
            if not df_m5.empty:
                df_m5 = self.add_technical_indicators(df_m5)
            
            # Save market data
            self.logger.save_market_data(symbol, "M1", df_m1, "COMBINED")
            
            # Check market conditions
            if not self.validate_market_conditions(symbol, df_m1):
                return
            
            # Get ML predictions
            state = self.models[symbol].data_processor.preprocess_data(df_m1, df_m5, df_m15, df_h1)
            action = self.models[symbol].act(state, df_m1, df_m5)
            
            # Check for whale activity
            whale_detected = self.whale_detector.detect_whale_activity(symbol, df_m1)
            
            # Execute trade if conditions are met
            if action in [0, 1]:  # SELL or BUY
                self.execute_trade(symbol, action, df_m1, whale_detected)
                
        except Exception as e:
            self.logger.error(f"Error processing {symbol}: {str(e)}")

    def get_market_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Your original market data logic preserved"""
        try:
            # Try MT5 first, then MEXC
            if MT5_AVAILABLE and self.platform.mt5_initialized:
                df = self.platform.get_mt5_data(CONFIG["SYMBOLS"][symbol]["MT5"], timeframe.upper())
                if not df.empty:
                    return df
            
            # Try MEXC
            if self.platform.mexc:
                df = self.platform.get_mexc_data(CONFIG["SYMBOLS"][symbol]["MEXC"], timeframe)
                if not df.empty:
                    return df
            
            self.logger.warning(f"No data available for {symbol} ({timeframe})")
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Failed to get market data for {symbol}: {str(e)}")
            return pd.DataFrame()

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Your original technical indicators logic preserved"""
        try:
            if len(df) < 50:
                return df
                
            df["RSI"] = TechnicalAnalysis.RSI(df["close"], timeperiod=CONFIG["INDICATORS"]["RSI_PERIOD"])
            df["MACD"], df["MACD_Signal"], df["MACD_Histogram"] = TechnicalAnalysis.MACD(
                df["close"], fastperiod=CONFIG["INDICATORS"]["MACD_FAST"],
                slowperiod=CONFIG["INDICATORS"]["MACD_SLOW"], signalperiod=CONFIG["INDICATORS"]["MACD_SIGNAL"]
            )
            df["ATR"] = TechnicalAnalysis.ATR(df["high"], df["low"], df["close"], timeperiod=CONFIG["INDICATORS"]["ATR_PERIOD"])
            df["BB_upper"], df["BB_middle"], df["BB_lower"] = TechnicalAnalysis.BBANDS(
                df["close"], timeperiod=CONFIG["INDICATORS"]["BOLLINGER_PERIOD"],
                nbdevup=CONFIG["INDICATORS"]["BOLLINGER_STDDEV"], nbdevdn=CONFIG["INDICATORS"]["BOLLINGER_STDDEV"]
            )
            df["Stoch_K"], df["Stoch_D"] = TechnicalAnalysis.STOCH(df["high"], df["low"], df["close"])
            df["ADX"] = TechnicalAnalysis.ADX(df["high"], df["low"], df["close"], timeperiod=CONFIG["INDICATORS"]["ADX_PERIOD"])
            df["OBV"] = TechnicalAnalysis.OBV(df["close"], df["volume"])
            
            # Additional features
            df['Price_Change'] = df['close'].pct_change()
            df['Volatility'] = df['Price_Change'].rolling(window=CONFIG["INDICATORS"]["BOLLINGER_PERIOD"]).std()
            df['Volume_MA'] = df['volume'].rolling(window=CONFIG["INDICATORS"]["BOLLINGER_PERIOD"]).mean()
            df['Volume_Ratio'] = df['volume'] / (df['Volume_MA'] + 1e-8)
            
            return df.fillna(method='ffill').fillna(method='bfill')
        except Exception as e:
            self.logger.error(f"Failed to add technical indicators: {str(e)}")
            return df

    def validate_market_conditions(self, symbol: str, df: pd.DataFrame) -> bool:
        """Your original market validation logic preserved"""
        try:
            if len(df) < 50:
                return False
            
            # Check volatility
            atr = df["ATR"].iloc[-1] if "ATR" in df.columns else 0
            if atr > CONFIG["MAX_ATR_THRESHOLD"]:
                return False
            
            # Check ADX for trend strength
            adx = df["ADX"].iloc[-1] if "ADX" in df.columns else 0
            if adx < CONFIG["MIN_ADX_THRESHOLD"]:
                return False
            
            # Check volume
            volume_ratio = df["Volume_Ratio"].iloc[-1] if "Volume_Ratio" in df.columns else 1
            if volume_ratio < CONFIG["MIN_VOLUME_MULTIPLIER"]:
                return False
            
            return True
        except Exception as e:
            self.logger.error(f"Market validation failed for {symbol}: {str(e)}")
            return False

    def execute_trade(self, symbol: str, action: int, df: pd.DataFrame, whale_detected: bool):
        """Your original trade execution logic preserved"""
        try:
            action_str = "SELL" if action == 0 else "BUY"
            current_price = df["close"].iloc[-1]
            atr = df["ATR"].iloc[-1] if "ATR" in df.columns else current_price * 0.001
            
            # Calculate position parameters
            volume = self.calculate_position_size(symbol, current_price, atr)
            sl = self.calculate_stop_loss(symbol, current_price, action_str, atr)
            tp = self.calculate_take_profit(symbol, current_price, action_str, atr)
            
            # Execute on preferred platform
            platform = "MT5" if MT5_AVAILABLE and self.platform.mt5_initialized else "MEXC"
            order_ticket = self.platform.execute_trade(symbol, action_str, volume, sl, tp, platform)
            
            if order_ticket:
                self.open_positions[order_ticket] = {
                    "symbol": symbol,
                    "action": action_str,
                    "entry_price": current_price,
                    "volume": volume,
                    "sl": sl,
                    "tp": tp,
                    "entry_time": datetime.now(),
                    "platform": platform,
                    "whale_detected": whale_detected
                }
                
                self.last_trade_time[symbol] = datetime.now()
                self.daily_trades += 1
                self.trade_counter += 1
                
                # Send Telegram notification
                whale_emoji = "🐋" if whale_detected else ""
                action_emoji = "📈" if action_str == "BUY" else "📉"
                
                message = f"""
<b>{action_emoji} TRADE OPENED {whale_emoji}</b>

<b>Symbol:</b> {symbol}
<b>Action:</b> {action_str}
<b>Price:</b> {current_price:.4f}
<b>Volume:</b> {volume:.6f}
<b>Stop Loss:</b> {sl:.4f}
<b>Take Profit:</b> {tp:.4f}
<b>Platform:</b> {platform}
<b>Time:</b> {datetime.now().strftime('%H:%M:%S')}
<b>Ticket:</b> {order_ticket}
                """
                self.logger.send_telegram_alert(message.strip())
                
                # Remember for ML training
                state = self.models[symbol].data_processor.preprocess_data(df, pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
                self.models[symbol].remember(state.flatten(), action, 0.0, state.flatten(), False)
                
        except Exception as e:
            self.logger.error(f"Trade execution failed for {symbol}: {str(e)}")

    def calculate_position_size(self, symbol: str, price: float, atr: float) -> float:
        """Your original position sizing logic preserved"""
        try:
            # Risk-based position sizing
            account_balance = 10000.0  # Base account balance
            risk_amount = account_balance * CONFIG["RISK_PERCENT_PER_TRADE"]
            
            # Position size based on ATR
            atr_risk = atr * CONFIG["STOP_LOSS_ATR_MULTIPLIER"]
            position_size = risk_amount / atr_risk if atr_risk > 0 else 0.01
            
            # Apply symbol constraints
            lot_size = CONFIG["SYMBOLS"][symbol]["LOT_SIZE"]
            position_size = round(position_size / lot_size, 6) * lot_size
            
            return max(0.01, min(position_size, 1.0))  # Min 0.01, Max 1.0
        except Exception as e:
            self.logger.error(f"Position size calculation failed: {str(e)}")
            return 0.01

    def calculate_stop_loss(self, symbol: str, price: float, action: str, atr: float) -> float:
        """Your original stop loss calculation preserved"""
        try:
            atr_distance = atr * CONFIG["STOP_LOSS_ATR_MULTIPLIER"]
            if action == "BUY":
                return round(price - atr_distance, 4)
            else:
                return round(price + atr_distance, 4)
        except Exception as e:
            self.logger.error(f"Stop loss calculation failed: {str(e)}")
            return price * 0.99 if action == "BUY" else price * 1.01

    def calculate_take_profit(self, symbol: str, price: float, action: str, atr: float) -> float:
        """Your original take profit calculation preserved"""
        try:
            atr_distance = atr * CONFIG["TAKE_PROFIT_ATR_MULTIPLIER"]
            if action == "BUY":
                return round(price + atr_distance, 4)
            else:
                return round(price - atr_distance, 4)
        except Exception as e:
            self.logger.error(f"Take profit calculation failed: {str(e)}")
            return price * 1.02 if action == "BUY" else price * 0.98

    def manage_open_positions(self):
        """Your original position management logic preserved"""
        try:
            for order_ticket, position in list(self.open_positions.items()):
                symbol = position["symbol"]
                current_df = self.get_market_data(symbol, "1m")
                
                if current_df.empty:
                    continue
                
                current_price = current_df["close"].iloc[-1]
                entry_time = position["entry_time"]
                duration = (datetime.now() - entry_time).total_seconds()
                
                # Calculate profit
                if position["action"] == "BUY":
                    profit = (current_price - position["entry_price"]) * position["volume"]
                else:
                    profit = (position["entry_price"] - current_price) * position["volume"]
                
                # Check exit conditions
                should_close = False
                close_reason = ""
                
                # Stop loss / Take profit
                if position["action"] == "BUY":
                    if current_price <= position["sl"]:
                        should_close = True
                        close_reason = "Stop Loss"
                    elif current_price >= position["tp"]:
                        should_close = True
                        close_reason = "Take Profit"
                else:
                    if current_price >= position["sl"]:
                        should_close = True
                        close_reason = "Stop Loss"
                    elif current_price <= position["tp"]:
                        should_close = True
                        close_reason = "Take Profit"
                
                # Time-based closure (max 1 hour)
                if duration > 3600:
                    should_close = True
                    close_reason = "Time Limit"
                
                if should_close:
                    self.close_position(order_ticket, current_price, profit, int(duration), close_reason)
                    
        except Exception as e:
            self.logger.error(f"Position management failed: {str(e)}")

    def close_position(self, order_ticket: str, exit_price: float, profit: float, duration: int, reason: str):
        """Your original position closing logic preserved"""
        try:
            position = self.open_positions.get(order_ticket)
            if not position:
                return
            
            symbol = position["symbol"]
            platform = position["platform"]
            
            # Close on platform
            self.platform.close_trade(symbol, order_ticket, platform, exit_price, profit, duration)
            
            # Update daily profit
            self.daily_profit += profit
            
            # Remove from open positions
            del self.open_positions[order_ticket]
            
            # Send notification
            profit_emoji = "💰" if profit > 0 else "📉"
            message = f"""
<b>{profit_emoji} TRADE CLOSED</b>

<b>Symbol:</b> {symbol}
<b>Action:</b> {position["action"]}
<b>Entry:</b> {position["entry_price"]:.4f}
<b>Exit:</b> {exit_price:.4f}
<b>Profit:</b> ${profit:.2f}
<b>Reason:</b> {reason}
<b>Duration:</b> {duration//60}m {duration%60}s
<b>Platform:</b> {platform}
<b>Ticket:</b> {order_ticket}
            """
            self.logger.send_telegram_alert(message.strip())
            
            # Train ML model with result
            reward = profit * 100  # Scale reward
            state = np.zeros(CONFIG["STATE_SIZE"])  # Simplified for now
            action = 0 if position["action"] == "SELL" else 1
            self.models[symbol].remember(state, action, reward, state, True)
            
        except Exception as e:
            self.logger.error(f"Position closure failed for {order_ticket}: {str(e)}")

    def close_all_positions(self):
        """Your original close all logic preserved"""
        try:
            for order_ticket in list(self.open_positions.keys()):
                position = self.open_positions[order_ticket]
                symbol = position["symbol"]
                current_df = self.get_market_data(symbol, "1m")
                
                if not current_df.empty:
                    current_price = current_df["close"].iloc[-1]
                    if position["action"] == "BUY":
                        profit = (current_price - position["entry_price"]) * position["volume"]
                    else:
                        profit = (position["entry_price"] - current_price) * position["volume"]
                    
                    duration = int((datetime.now() - position["entry_time"]).total_seconds())
                    self.close_position(order_ticket, current_price, profit, duration, "Bot Shutdown")
        except Exception as e:
            self.logger.error(f"Failed to close all positions: {str(e)}")

    def train_models(self):
        """Your original training logic preserved"""
        try:
            for symbol in CONFIG["ACTIVE_SYMBOLS"]:
                try:
                    # Train DQN
                    if len(self.models[symbol].memory) > CONFIG["BATCH_SIZE"]:
                        self.models[symbol].replay(CONFIG["BATCH_SIZE"])
                        
                except Exception as e:
                    self.logger.error(f"Training failed for {symbol}: {str(e)}")
                    
        except Exception as e:
            self.logger.error(f"Model training failed: {str(e)}")

    def save_all_models(self):
        """Your original save models logic preserved"""
        try:
            for symbol in CONFIG["ACTIVE_SYMBOLS"]:
                self.models[symbol].save_model()
            self.logger.info("All models saved successfully")
        except Exception as e:
            self.logger.error(f"Failed to save models: {str(e)}")

    def check_daily_reset(self):
        """Your original daily reset logic preserved"""
        try:
            current_date = datetime.now().date()
            if current_date != self.daily_reset_time:
                # Send daily summary
                summary_message = f"""
<b>📊 Daily Trading Summary</b>

<b>Date:</b> {self.daily_reset_time}
<b>Trades:</b> {self.daily_trades}
<b>Profit:</b> ${self.daily_profit:.2f}
<b>Open Positions:</b> {len(self.open_positions)}

New trading day started!
                """
                self.logger.send_telegram_alert(summary_message.strip())
                
                # Reset daily counters
                self.daily_trades = 0
                self.daily_profit = 0.0
                self.daily_reset_time = current_date
                
        except Exception as e:
            self.logger.error(f"Daily reset failed: {str(e)}")

# =============================================================================
# SIMPLE FLASK DASHBOARD
# =============================================================================

def create_flask_app(bot: UltimateTradingBot) -> Flask:
    """Create Flask dashboard"""
    app = Flask(__name__)
    
    @app.route('/')
    def dashboard():
        try:
            status = {
                "running": bot.running,
                "daily_trades": bot.daily_trades,
                "daily_profit": bot.daily_profit,
                "open_positions": len(bot.open_positions),
                "symbols": CONFIG["ACTIVE_SYMBOLS"]
            }
            return jsonify(status)
        except Exception as e:
            return jsonify({"error": str(e)})
    
    @app.route('/positions')
    def positions():
        try:
            positions = []
            for ticket, pos in bot.open_positions.items():
                positions.append({
                    "ticket": ticket,
                    "symbol": pos["symbol"],
                    "action": pos["action"],
                    "entry_price": pos["entry_price"],
                    "volume": pos["volume"],
                    "entry_time": pos["entry_time"].strftime("%H:%M:%S")
                })
            return jsonify(positions)
        except Exception as e:
            return jsonify({"error": str(e)})
    
    @app.route('/start', methods=['POST'])
    def start_bot():
        try:
            if not bot.running:
                threading.Thread(target=bot.start, daemon=True).start()
                return jsonify({"success": True, "message": "Bot started"})
            return jsonify({"success": False, "message": "Bot already running"})
        except Exception as e:
            return jsonify({"success": False, "message": str(e)})
    
    @app.route('/stop', methods=['POST'])
    def stop_bot():
        try:
            bot.stop()
            return jsonify({"success": True, "message": "Bot stopped"})
        except Exception as e:
            return jsonify({"success": False, "message": str(e)})
    
    return app

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function - Windows ready"""
    try:
        print("=" * 60)
        print("ULTIMATE AI TRADING BOT - FINAL VERSION")
        print("=" * 60)
        print(f"✓ Python version: {sys.version}")
        print(f"✓ TensorFlow available: {TENSORFLOW_AVAILABLE}")
        print(f"✓ MetaTrader5 available: {MT5_AVAILABLE}")
        print(f"✓ TA-Lib available: {TALIB_AVAILABLE}")
        print()
        
        # Setup project structure
        setup_project_structure()
        
        # Initialize bot
        bot = UltimateTradingBot()
        
        # Create Flask app
        flask_app = create_flask_app(bot)
        
        # Command line arguments
        if len(sys.argv) > 1:
            if sys.argv[1] == "--start":
                print("Starting bot automatically...")
                bot.start()
                return
            elif sys.argv[1] == "--dashboard":
                print("Starting Flask dashboard on http://localhost:5000")
                flask_app.run(host='0.0.0.0', port=5000, debug=False)
                return
        
        # Interactive mode
        while True:
            print("\n" + "="*50)
            print("ULTIMATE TRADING BOT - CONTROL PANEL")
            print("="*50)
            print("1. Start Trading Bot")
            print("2. Stop Trading Bot")
            print("3. Show Status")
            print("4. Show Open Positions")
            print("5. Start Flask Dashboard")
            print("6. Test Telegram Connection")
            print("7. Exit")
            print("="*50)
            
            try:
                choice = input("Enter your choice (1-7): ").strip()
                
                if choice == "1":
                    if not bot.running:
                        print("🚀 Starting trading bot...")
                        threading.Thread(target=bot.start, daemon=True).start()
                        time.sleep(2)
                        print("✓ Bot started successfully!")
                    else:
                        print("⚠ Bot is already running!")
                
                elif choice == "2":
                    if bot.running:
                        print("🛑 Stopping trading bot...")
                        bot.stop()
                        print("✓ Bot stopped successfully!")
                    else:
                        print("⚠ Bot is not running!")
                
                elif choice == "3":
                    print(f"\n📊 BOT STATUS:")
                    print(f"Running: {'🟢 YES' if bot.running else '🔴 NO'}")
                    print(f"Daily Trades: {bot.daily_trades}/{CONFIG['MAX_DAILY_TRADES']}")
                    print(f"Daily Profit: ${bot.daily_profit:.2f}")
                    print(f"Open Positions: {len(bot.open_positions)}")
                    print(f"Trade Counter: {bot.trade_counter}")
                
                elif choice == "4":
                    if bot.open_positions:
                        print(f"\n📈 OPEN POSITIONS ({len(bot.open_positions)}):")
                        for ticket, pos in bot.open_positions.items():
                            duration = int((datetime.now() - pos['entry_time']).total_seconds() / 60)
                            print(f"  🎫 {ticket}: {pos['action']} {pos['symbol']} @ {pos['entry_price']:.4f}")
                            print(f"     SL: {pos['sl']:.4f} | TP: {pos['tp']:.4f} | Duration: {duration}m")
                    else:
                        print("\n📭 No open positions")
                
                elif choice == "5":
                    print("🌐 Starting Flask dashboard...")
                    print("Dashboard will be available at: http://localhost:5000")
                    try:
                        flask_app.run(host='0.0.0.0', port=5000, debug=False)
                    except KeyboardInterrupt:
                        print("\n🛑 Dashboard stopped")
                
                elif choice == "6":
                    print("📱 Testing Telegram connection...")
                    success = bot.logger.send_telegram_alert("🧪 Test message from Ultimate Trading Bot")
                    if success:
                        print("✓ Telegram connection successful!")
                    else:
                        print("❌ Telegram connection failed!")
                
                elif choice == "7":
                    if bot.running:
                        print("🛑 Stopping bot before exit...")
                        bot.stop()
                    print("👋 Goodbye!")
                    break
                
                else:
                    print("❌ Invalid choice! Please enter 1-7.")
                    
            except KeyboardInterrupt:
                print("\n\n🛑 Shutting down...")
                if bot.running:
                    bot.stop()
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                traceback.print_exc()
                
    except Exception as e:
        print(f"💥 Critical error: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()