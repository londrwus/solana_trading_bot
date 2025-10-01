import asyncio
import logging
import sys
import time
import random
import requests
import json

from typing import Dict, Any, Optional
from solana_client import SolanaManager, DexScreenerAPI
from trading_bot import TradingBot
from telegram_notifier import TelegramNotifier
from config import TOKEN_FILTERS, DCA_TYPES
from utils import logger, check_timestamp_difference, format_number
from solana.rpc.api import Pubkey
from solana_tx_parser import parse_transaction

# For debug
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

class DCAMonitor:
    """Monitor for new DCA transactions"""
    
    def __init__(self, trading_bot: TradingBot, notifier: TelegramNotifier):
        self.bot = trading_bot
        self.notifier = notifier
        self.tx_history = []
        self.max_history = 200
    
    async def monitor_dca_transactions(self):
        dca_program_address = 'DCA265Vj8a9CEuX1eb1LWRnDT7uK6q1xMipnNyatn23M' # Jupiter DCA address
        
        while True:
            try:
                # Get recent transactions
                transactions = await self._get_recent_transactions(dca_program_address)
                
                for tx in transactions:
                    if tx in self.tx_history:
                        continue
                    
                    # Process new transaction
                    await self._process_dca_transaction(tx)
                    
                    # Add to history
                    self.tx_history.append(tx)
                    if len(self.tx_history) > self.max_history:
                        self.tx_history.pop(0)
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"DCA monitor error: {e}")
                await asyncio.sleep(5)
    
    async def _get_recent_transactions(self, program_address: str):
        """Get recent transactions from DCA account"""
        try:
            result = self.bot.solana.client.get_signatures_for_address(
                Pubkey.from_string(program_address),
                limit=100
            )
            
            if result and result.value:
                return [str(tx.signature) for tx in result.value]
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to get transactions: {e}")
            return []
    
    async def _process_dca_transaction(self, tx_id: str):
        """Process a DCA transaction"""
        try:
            # Get transaction details
            tx_data = await self._get_transaction_details(tx_id)
            
            if not tx_data or 'OpenDcaV2' not in str(tx_data):
                return
            
            # Parse transaction
            parsed_data = self._parse_dca_transaction(tx_data)
            
            if not parsed_data:
                return
            
            # Check if it's a buy opportunity (not sell as we are only buing DCA)
            if parsed_data['input_mint'] not in TOKEN_FILTERS['base_tokens']:
                return
            
            # Get token info
            token_info = DexScreenerAPI.get_token_info(parsed_data['output_mint'])
            
            if not token_info:
                return
            
            # Prepare DCA data
            dca_data = {
                'input_mint': parsed_data['output_mint'],
                'market_cap': token_info['market_cap'],
                'liquidity_usd': token_info['liquidity_usd'],
                'total_buy_amount': parsed_data['total_buy_amount'],
                'created_at': token_info['created_at'],
                'symbol': token_info['symbol'],
                'price': token_info['price_usd'],
                'tx_id': tx_id,
                'user': parsed_data['user'],
                'cycle_frequency': parsed_data['cycle_frequency'],
                'amount_per_cycle': parsed_data['amount_per_cycle']
            }
            
            # Evaluate opportunity
            dca_type = self.bot.evaluate_dca_opportunity(dca_data)
            
            if dca_type:
                await self._handle_valid_opportunity(dca_data, dca_type)
                
        except Exception as e:
            logger.error(f"Failed to process DCA transaction {tx_id}: {e}")
    
    async def _get_transaction_details(self, tx_id: str):
        """Get full transaction details"""
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getTransaction",
            "params": [
                tx_id,
                {
                    "encoding": "json",
                    "commitment": "confirmed",
                    "maxSupportedTransactionVersion": 0
                }
            ]
        }
        
        headers = {"Content-Type": "application/json"}
        
        try:
            response = requests.post(random.choice(RPC_URLS), json=payload, headers=headers)
            
            if response.status_code == 429:
                logger.warning("Rate limit exceeded when getting transaction details")
                await asyncio.sleep(3)
                return None
                
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"Error fetching transaction details for {tx_id}: {e}")
            return None
    
    def _parse_dca_transaction(self, tx_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse DCA transaction data"""
        try:
            # Check if it's a DCA transaction
            if not tx_data or 'OpenDcaV2' not in str(tx_data):
                return None
            
            # Convert to JSON and parse with solana_tx_parser
            tx_json = json.dumps(tx_data)
            parsed_result = parse_transaction(tx_json)
            
            if not parsed_result or not parsed_result[0].get('actions'):
                return None
            
            # Find the correct action
            action = parsed_result[0]['actions'][0]
            cnt = 0
            while len(action) < 10 and cnt < len(parsed_result[0]['actions']) - 1:
                cnt += 1
                action = parsed_result[0]['actions'][cnt]
            
            # Extract transaction data
            result = {
                'input_mint': str(action.get('inputMint', '')),
                'output_mint': str(action.get('outputMint', '')),
                'user': str(action.get('user', '')),
                'amount_per_cycle': action.get('inAmountPerCycle', 0),
                'total_amount': action.get('inAmount', 0),
                'cycle_frequency': action.get('cycleFrequency', 0)
            }
            
            # Get the correct total amount from post token balances
            if 'rawTx' in parsed_result[0] and 'meta' in parsed_result[0]['rawTx']:
                post_balances = parsed_result[0]['rawTx']['meta'].get('postTokenBalances', [])
                if post_balances:
                    # Find matching total amount
                    total_amount_check = post_balances[0]['uiTokenAmount']['amount']
                    cnt_amount = 0
                    
                    while int(result['total_amount']) != int(total_amount_check) and cnt_amount < len(post_balances) - 1:
                        cnt_amount += 1
                        total_amount_check = post_balances[cnt_amount]['uiTokenAmount']['amount']
                    
                    result['total_amount_ui'] = post_balances[cnt_amount]['uiTokenAmount']['uiAmount']
            
            return result
            
        except Exception as e:
            logger.error(f"Error parsing DCA transaction: {e}")
            return None
    
    async def _handle_valid_opportunity(self, dca_data: Dict[str, Any], dca_type: str):
        """Handle a valid DCA transaction"""
        
        volume_liquidity_ratio = dca_data['total_buy_amount'] / dca_data['liquidity_usd']
        impact_percent = ((1 + volume_liquidity_ratio) ** 2 - 1) * 100
        
        # Send notification
        message = self.notifier.format_trade_message(
            trade_type=dca_type,
            symbol=dca_data['symbol'],
            amount_usd=dca_data['total_buy_amount'],
            price=dca_data['price'],
            liquidity=dca_data['liquidity_usd'],
            market_cap=dca_data['market_cap'],
            volume_ratio=volume_liquidity_ratio,
            tx_id=dca_data['tx_id'],
            contract=dca_data['input_mint'],
            process='BUY'
        )
        
        self.notifier.send_notification(message, dca_type)
        
        # Add to active monitoring
        self.bot.active_dcas[dca_data['user']] = {
            'inputMint': dca_data['input_mint'],
            'tx': dca_data['tx_id'],
            'type': dca_type
        }
        
        # Calculate end time
        total_cycles = int(dca_data['total_buy_amount'] / dca_data['amount_per_cycle'])
        total_seconds = dca_data['cycle_frequency'] * total_cycles
        end_time = time.time() + total_seconds
        
        # Add to active coins
        self.bot.active_coins[dca_data['input_mint']] = {
            'end_time': end_time,
            'entry_time': time.time() + 60,  # Entry after 1 minute
            'type': dca_type,
            'dca_status': 'pending',
            'percentToClose': impact_percent,
            'firstPrice': dca_data['price'],
            'lastPrice': dca_data['price'],
            'liquidity_provided': int(dca_data['liquidity_usd']),
            'after_1min': dca_data['price']
        }
        
        logger.info(f"Added DCA opportunity: {dca_data['input_mint']} - Type: {dca_type}")

async def main():
    """Main function to run the trading bot"""
    
    logger.info("Starting DCA Trading Bot")
    
    # Initialize components
    solana_manager = SolanaManager()
    trading_bot = TradingBot(solana_manager)
    notifier = TelegramNotifier()
    dca_monitor = DCAMonitor(trading_bot, notifier)
    
    # Start monitoring tasks
    tasks = [
        asyncio.create_task(trading_bot.monitor_prices()),
        asyncio.create_task(dca_monitor.monitor_dca_transactions()),
    ]
    
    try:
        # Run forever
        await asyncio.gather(*tasks)
        
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot crashed: {e}")
    finally:
        for task in tasks:
            task.cancel()
        
        logger.info("Bot shutdown complete")

def run_with_retry():
    """Run the bot with automatic retry on failure"""
    while True:
        try:
            asyncio.run(main())
        except Exception as e:
            logger.error(f"Bot failed: {e}")
            logger.info("Restarting in 10 seconds...")
            time.sleep(10)

if __name__ == "__main__":
    run_with_retry()