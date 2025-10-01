import os
from dotenv import load_dotenv

load_dotenv()

# Env variables
RPC_URLS = os.getenv('RPC_URLS', '').split(',')
JUPITER_API_URL = os.getenv('JUPITER_API_URL')
JUPITER_SWAP_URL = os.getenv('JUPITER_SWAP_URL')
WALLET_PRIVATE_KEY = os.getenv('WALLET_PRIVATE_KEY')
TOKEN_BALANCE_ADDRESS = os.getenv('TOKEN_BALANCE_ADDRESS')
HELIUS_API_KEY = os.getenv('HELIUS_API_KEY')
DEXSCREENER_API_URL = os.getenv('DEXSCREENER_API_URL')

# Parse Telegram creds
telegram_creds = os.getenv('TELEGRAM_CREDENTIALS', '|').split('|')
TELEGRAM_BOT_TOKENS = telegram_creds[0].split(',') if len(telegram_creds) > 0 else []
TELEGRAM_CHAT_IDS = telegram_creds[1].split(',') if len(telegram_creds) > 1 else []

# Trading parameters
TRADING_CONFIG = {
    'fee_lamports': 520_000, # fee on solana blockchain
    'rate_limit': 45,
    'slippage_bps': 20, # 20%
    'max_retries': 3,
    
    # Trade sizes in lamports
    'trade_sizes': {
        'buy1': 1_000_000_000,
        'buy3': 3_000_000_000,
        'buy5': 5_000_000_000,
        'buy7': 7_000_000_000,
        'buy9': 9_000_000_000,
        'buy10': 10_000_000_000,
    },
    
    # Liquidity thresholds for trade sizing
    'liquidity_thresholds': [
        (150_000, 'buy1'),
        (200_000, 'buy3'),
        (300_000, 'buy5'),
        (400_000, 'buy7'),
        (500_000, 'buy9'),
        (float('inf'), 'buy10'),
    ],
    
    # Stop loss and take profit percentages
    'stop_loss_percent': 7.0,
    'take_profit_multiplier': 2.0,  # Take profit is 2x the stop loss
    'price_adjustment': 0.3,
    'stop_adjustment': 0.5,
}

# Token filters
TOKEN_FILTERS = {
    'min_mcap': 1_000_000,
    'min_liquidity': 50_000,
    'min_holders': 200,
    'min_age_hours': 12, # time since listing
    'min_volume_liquidity_ratio': 0.1,
    
    # Tokens to trade against (SOL, USDC)
    'base_tokens': [
        'So11111111111111111111111111111111111111112',
        'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v'
    ],
    
    # Blacklisted scam tokens
    'blacklist': [
        '6WNva7iLjTvxSfXPSmbjceW5Yc41LUH4SJNqKom5pump',
        # Add more blacklisted tokens here
    ]
}

# DCA Types configuration
DCA_TYPES = {
    'TYPE1': {
        'min_mcap': 1_000_000,
        'min_liquidity': 100_000,
        'min_volume_ratio': 0.1
    },
    'TYPE2': {
        'min_mcap': 100_000,
        'max_mcap': 500_000,
        'min_liquidity': 30_000,
        'min_volume_ratio': 0.125
    },
    'TYPE3': {
        'min_mcap': 5_000_000,
        'min_liquidity': 200_000,
        'min_volume_ratio': 0.05
    },
}