import time
import logging
import csv
import os

from collections import deque
from datetime import datetime
from typing import Dict, Any, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RateLimiter:
    """Rate limiter to prevent API overload"""
    
    def __init__(self, max_requests_per_second: int):
        self.max_requests = max_requests_per_second
        self.requests = deque()
    
    def wait_if_needed(self):
        """Wait if rate limit is reached"""
        now = datetime.now().timestamp()
        
        # Remove old requests (older than 1 second)
        while self.requests and now - self.requests[0] >= 1.02:
            self.requests.popleft()
        
        # If we've hit the limit, wait
        if len(self.requests) >= self.max_requests:
            sleep_time = 1.02 - (now - self.requests[0])
            if sleep_time > 0:
                logger.debug(f"Rate limit reached, waiting {sleep_time:.3f}s")
                time.sleep(sleep_time)
            self.requests.popleft()
        
        self.requests.append(now)

def format_number(num: float) -> str:
    """Format large numbers with K, M, B, ... suffixes"""
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '%.2f%s' % (num, ['', 'K', 'M', 'B', 'T'][magnitude])

def calculate_eta(seconds_per_item: float, items_count: int) -> str:
    """Calculate estimated time of when DCA will end"""
    total_seconds = seconds_per_item * items_count
    
    time_units = {
        'days': total_seconds // (24 * 3600),
        'hours': (total_seconds % (24 * 3600)) // 3600,
        'minutes': (total_seconds % 3600) // 60,
        'seconds': total_seconds % 60
    }
    
    result = [f"{int(val)} {unit}" for unit, val in time_units.items() if val > 0]
    return ' '.join(result) if result else "0 seconds"

def format_cycle_frequency(frequency: int) -> str:
    """Format cycle frequency for display"""
    if frequency >= 60:
        minutes = frequency / 60
        return f"Every {int(minutes)} minute{'s' if minutes > 1 else ''}"
    return f'Every {frequency} seconds'

def check_timestamp_difference(timestamp_ms: int, max_hours: int = 12) -> bool:
    """Check if timestamp is older than max_hours"""
    time_created = datetime.fromtimestamp(timestamp_ms / 1000)
    current_time = datetime.now()
    time_diff = abs(current_time - time_created)
    return time_diff.total_seconds() > (max_hours * 3600)

def save_trade_to_csv(mint: str, coin_data: Dict[str, Any], status: str, filepath: str = 'trades.csv'):
    """Save trade data to CSV file""" 
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    file_exists = os.path.isfile(filepath)
    
    row_data = {
        'timestamp': current_time,
        'mint': mint,
        'status': status,
        'type': coin_data.get('type', ''),
        'first_price': coin_data.get('firstPrice', ''),
        'last_price': coin_data.get('lastPrice', ''),
        'profit_loss': ((coin_data.get('lastPrice', 0) - coin_data.get('firstPrice', 0)) / 
                       coin_data.get('firstPrice', 1)) * 100 if coin_data.get('firstPrice') else 0
    }
    
    try:
        with open(filepath, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=row_data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row_data)
        logger.info(f"Trade saved to CSV: {mint} - {status}")
    except Exception as e:
        logger.error(f"Failed to save trade to CSV: {e}")