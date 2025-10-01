import logging
import telebot

from typing import Optional
from config import TELEGRAM_BOT_TOKENS, TELEGRAM_CHAT_IDS
from datetime import datetime
from utils import format_number, format_cycle_frequency, calculate_eta

logger = logging.getLogger(__name__)

class TelegramNotifier:
    """Handles Telegram notifications"""
    
    def __init__(self):
        self.bot_tokens = TELEGRAM_BOT_TOKENS
        self.chat_ids = TELEGRAM_CHAT_IDS
        self.bots = {}
        
        # Init bots
        for i, token in enumerate(self.bot_tokens):
            try:
                self.bots[f'TYPE{i+1}'] = telebot.TeleBot(token)
                logger.info(f"Initialized Telegram bot TYPE{i+1}")
            except Exception as e:
                logger.error(f"Failed to initialize bot {i}: {e}")
    
    def send_notification(self, message: str, message_type: str = 'TYPE1', disable_preview: bool = True):
        """Send notification to Telegram"""
        try:
            # Extract type number
            type_num = int(''.join(filter(str.isdigit, message_type))) - 1
            
            if type_num >= len(self.chat_ids) or type_num >= len(self.bot_tokens):
                logger.error(f"Invalid message type: {message_type}")
                return False
            
            chat_id = self.chat_ids[type_num]
            bot = self.bots.get(message_type)
            
            if not bot:
                logger.error(f"Bot not found for type: {message_type}")
                return False
            
            bot.send_message(
                chat_id,
                message,
                parse_mode='HTML',
                disable_web_page_preview=disable_preview
            )
            
            logger.info(f"Notification sent to {message_type}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
            return False
    
    def format_trade_message(
        self, 
        trade_type: str,
        symbol: str,
        amount_usd: float,
        price: float,
        liquidity: float,
        market_cap: float,
        volume_ratio: float,
        tx_id: str,
        contract: str,
        process: str = 'BUY'
    ) -> str:
        """Format trade notification message to Telegram"""
        current_time = datetime.now().strftime("%H:%M:%S")
        
        message = f"""
🔢 ${trade_type.upper()}
💰 {amount_usd:.2f} USD {process} ${symbol.upper()}

💵 Token Price: {price}
💧 Liquidity: {format_number(liquidity)} USD
📊 MCAP: {format_number(market_cap)}
📈 Volume {process}/Liquidity: {volume_ratio:.5f}

⏰ Time: {current_time}
🔗 Tx: <a href="https://solscan.io/tx/{tx_id}">{tx_id[:8]}...</a>
📝 Contract: <a href="https://dexscreener.com/solana/{contract}">{contract[:8]}...</a>
"""
        return message