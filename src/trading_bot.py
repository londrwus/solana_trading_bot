import asyncio
import base64
import logging
import time

from typing import Dict, Any, List, Optional
from datetime import datetime
from solders.transaction import VersionedTransaction
from solders import message
from solana.rpc.types import TxOpts
from solana.rpc.commitment import Processed
from config import TRADING_CONFIG, TOKEN_FILTERS, DCA_TYPES
from solana_client import SolanaManager, DexScreenerAPI

from utils import (
    RateLimiter, format_number, calculate_eta, 
    format_cycle_frequency, check_timestamp_difference, save_trade_to_csv
)

logger = logging.getLogger(__name__)

class TradingBot:
    """Main trading bot logic"""
    
    def __init__(self, solana_manager: SolanaManager):
        self.solana = solana_manager
        self.rate_limiter = RateLimiter(TRADING_CONFIG['rate_limit'])
        self.active_coins: Dict[str, Any] = {}
        self.active_dcas: Dict[str, Any] = {}
        self.active_positions: Dict[str, Any] = {}
        self.coins_for_types: Dict[str, Any] = {}
    
    async def swap_tokens(
        self, 
        input_token: str, 
        output_token: str, 
        process: str,
        amount: Optional[int] = None,
        attempt: int = 0
    ) -> bool:
        """Token swap"""
        
        if attempt >= TRADING_CONFIG['max_retries']:
            logger.error(f"Max swap attempts reached for {input_token} -> {output_token}")
            return False
        
        fee_lamports = TRADING_CONFIG['fee_lamports']
        slippage_bps = TRADING_CONFIG['slippage_bps']
        
        try:
            # Determine swap amount
            if process.startswith('buy'):
                trade_size = TRADING_CONFIG['trade_sizes'].get(process, TRADING_CONFIG['trade_sizes']['buy10'])
                swap_amount = trade_size
            else:
                # For sells, use the full token balance
                swap_amount = amount or self._get_token_amount_for_sell(output_token)
            
            logger.info(f"Swapping {input_token} -> {output_token}, amount: {swap_amount}, process: {process}")
            
            # Execute swap via Jupiter
            transaction_data = await self.solana.jupiter.swap(
                input_mint=input_token if process.startswith('buy') else output_token,
                output_mint=output_token if process.startswith('buy') else input_token,
                amount=swap_amount,
                prioritization_fee_lamports=fee_lamports,
                slippage_bps=slippage_bps
            )
            
            # Sign and send transaction
            raw_transaction = VersionedTransaction.from_bytes(base64.b64decode(transaction_data))
            signature = self.solana.keypair.sign_message(message.to_bytes_versioned(raw_transaction.message))
            signed_txn = VersionedTransaction.populate(raw_transaction.message, [signature])
            
            opts = TxOpts(skip_preflight=False, preflight_commitment=Processed)
            result = await self.solana.async_client.send_raw_transaction(
                txn=bytes(signed_txn), 
                opts=opts
            )
            
            tx_id = result.to_json()
            logger.info(f"Transaction sent: https://explorer.solana.com/tx/{tx_id}")
            
            # Wait and verify transaction
            await asyncio.sleep(3)
            success = await self._verify_transaction(tx_id, output_token, process)
            
            if success:
                if process == 'sell':
                    self._remove_position(output_token)
                else:
                    self._record_position(output_token, process)
                return True
            
            # Retry with adjusted parameters
            return await self.swap_tokens(
                input_token, output_token, process, amount,
                attempt=attempt + 1
            )
            
        except Exception as e:
            logger.error(f"Swap error: {e}")
            if "any route" in str(e).lower() and attempt < TRADING_CONFIG['max_retries']:
                # Increase slippage and retry
                TRADING_CONFIG['slippage_bps'] = int(slippage_bps * 1.25)
                return await self.swap_tokens(
                    input_token, output_token, process, amount,
                    attempt=attempt + 1
                )
            return False
    
    def _get_token_amount_for_sell(self, token: str) -> int:
        """Get token amount for selling"""
        if token in self.active_positions:
            for type_key, amount in self.active_positions[token].items():
                if amount and amount != '0':
                    return int(amount)
        return 0
    
    async def _verify_transaction(self, tx_id: str, token: str, process: str) -> bool:
        """Verify transaction completion"""
        for _ in range(7):
            status = await self.solana.check_transaction(tx_id)
            if status is True:
                return True
            elif status is False:
                return False
            await asyncio.sleep(8)
        return False
    
    def _record_position(self, token: str, process: str):
        """Record a new position"""
        token_type = self.coins_for_types.get(token, {}).get('type', 'unknown')
        if token not in self.active_positions:
            self.active_positions[token] = {}
        self.active_positions[token][token_type] = '0'
        logger.info(f"Position recorded: {token} - {token_type}")
    
    def _remove_position(self, token: str):
        """Remove a position"""
        if token in self.active_positions:
            del self.active_positions[token]
        if token in self.active_coins:
            del self.active_coins[token]
        logger.info(f"Position removed: {token}")
    
    async def monitor_prices(self):
        """Monitor and update prices for active coins"""
        
        # NOTE: it's better to update prices as ws connection (I believe)
        while True:
            try:
                if not self.active_coins:
                    await asyncio.sleep(10)
                    continue
                
                # Update prices
                updated_coins = await self._update_all_prices()
                
                # Check for exit conditions
                await self._check_exit_conditions(updated_coins)
                
                # Check for pending entries
                await self._process_pending_entries()
                
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Price monitor error: {e}")
                await asyncio.sleep(10)
    
    async def _update_all_prices(self) -> Dict[str, Any]:
        """Update prices for all active coins"""
        mint_addresses = list(self.active_coins.keys())
        
        if not mint_addresses:
            return self.active_coins
        
        try:
            # Batch price update via Jupiter API
            ids_string = ','.join(mint_addresses)
            prices = DexScreenerAPI.get_jupiter_price(ids_string)
            
            for mint in self.active_coins:
                if mint in prices:
                    self.active_coins[mint]['lastPrice'] = prices[mint]
                    
        except Exception as e:
            logger.error(f"Price update error: {e}")
        
        return self.active_coins
    
    async def _check_exit_conditions(self, coins: Dict[str, Any]):
        """Check if any positions should be closed"""
        coins_to_remove = []
        current_time = time.time()
        
        for mint, data in coins.items():
            try:
                first_price = data['firstPrice']
                last_price = data['lastPrice']
                percent_to_close = data['percentToClose']
                
                # Calculate price change
                price_change_percent = ((last_price - first_price) / first_price) * 100
                
                # Calculate adjusted thresholds
                adjusted_take_profit = percent_to_close - 0.3
                adjusted_stop_loss = -(percent_to_close / 2) + 0.5
                
                should_close = False
                status = ""
                
                # Check exit conditions
                if price_change_percent >= adjusted_take_profit:
                    should_close = True
                    status = "profit"
                    logger.info(f"Take profit triggered for {mint}: {price_change_percent:.2f}%")
                    
                # Stop loss
                elif price_change_percent <= adjusted_stop_loss:
                    should_close = True
                    status = "stop"
                    logger.info(f"Stop loss triggered for {mint}: {price_change_percent:.2f}%")
                    
                # Max stop loss (like 10%)
                elif price_change_percent <= -TRADING_CONFIG['stop_loss_percent']:
                    should_close = True
                    status = "stop_max"
                    logger.info(f"Max stop loss triggered for {mint}: {price_change_percent:.2f}%")
                    
                # all transactions from DCA has been bought
                elif current_time >= data.get('end_time', float('inf')):
                    should_close = True
                    status = "timeout"
                    logger.info(f"Position timeout for {mint}")
                    
                # DCA closed (like from the user)
                elif data.get('dca_status') == 'close':
                    should_close = True
                    status = "dca_close"
                    logger.info(f"DCA closed for {mint}")
                
                if should_close:
                    # Save to CSV
                    save_trade_to_csv(mint, data, status)
                    
                    # Execute sell
                    if mint in self.active_positions:
                        await self.swap_tokens(
                            'So11111111111111111111111111111111111111112', # from SOL
                            mint,
                            'sell'
                        )
                    
                    coins_to_remove.append(mint)
                    
            except Exception as e:
                logger.error(f"Exit check error for {mint}: {e}")
        
        # Remove closed positions
        for mint in coins_to_remove:
            self._remove_position(mint)
    
    async def _process_pending_entries(self):
        """Process pending entry signals"""
        current_time = time.time()
        
        pending_entries = {
            mint: data for mint, data in self.active_coins.items()
            if data.get('dca_status') == 'pending' and current_time >= data.get('entry_time', 0)
        }
        
        for mint, data in pending_entries.items():
            if mint not in self.active_coins:
                continue
            
            logger.info(f"Processing entry for {mint}")
            
            # Determine trade size based on liquidity
            liquidity = data.get('liquidity_provided', 0)
            trade_type = 'buy10'  # Default
            
            for threshold, trade_size in TRADING_CONFIG['liquidity_thresholds']:
                if liquidity <= threshold:
                    trade_type = trade_size
                    break
            
            # Execute buy
            success = await self.swap_tokens(
                'So11111111111111111111111111111111111111112', # SOL
                mint,
                trade_type
            )
            
            if success:
                # Update status
                self.active_coins[mint]['dca_status'] = 'open'
                
                # Check if price moved up after entry
                new_price = DexScreenerAPI.get_jupiter_price(mint)
                if new_price and new_price > data['firstPrice']:
                    self.active_coins[mint]['after_1min'] = new_price
    
    def evaluate_dca_opportunity(self, dca_data: Dict[str, Any]) -> Optional[str]:
        """If a DCA opportunity meets criteria"""
        
        # Extract data
        input_mint = dca_data.get('input_mint')
        parsed_mcap = dca_data.get('market_cap', 0)
        parsed_liquidity = dca_data.get('liquidity_usd', 0)
        total_buy_amount = dca_data.get('total_buy_amount', 0)
        created_at = dca_data.get('created_at', 0)
        
        # Check blacklist (scam tokens)
        if input_mint in TOKEN_FILTERS['blacklist']:
            return None
        
        # Calculate volume/liquidity ratio
        volume_liquidity_ratio = total_buy_amount / parsed_liquidity if parsed_liquidity > 0 else 0
        
        # Determine DCA type
        for dca_type, criteria in DCA_TYPES.items():
            if (parsed_mcap >= criteria.get('min_mcap', 0) and
                parsed_mcap <= criteria.get('max_mcap', float('inf')) and
                parsed_liquidity >= criteria.get('min_liquidity', 0) and
                volume_liquidity_ratio >= criteria.get('min_volume_ratio', 0)):
                
                # Additional filters
                if (parsed_mcap >= TOKEN_FILTERS['min_mcap'] and
                    parsed_liquidity >= TOKEN_FILTERS['min_liquidity'] and
                    check_timestamp_difference(created_at, TOKEN_FILTERS['min_age_hours'])):
                    
                    return dca_type
        
        return None