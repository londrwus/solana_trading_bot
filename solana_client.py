import json
import base58
import asyncio
import logging
import requests

from typing import Dict, Any, Optional, List
from solana.rpc.api import Client
from solana.rpc.async_api import AsyncClient
from solana.rpc.types import TxOpts
from solana.rpc.commitment import Processed
from solders.keypair import Keypair
from solders.transaction import VersionedTransaction
from solders import message
from jupiter_python_sdk.jupiter import Jupiter
from solana.transaction import Signature
from base58 import b58decode

from config import (
    RPC_URLS, JUPITER_API_URL, JUPITER_SWAP_URL, 
    WALLET_PRIVATE_KEY, HELIUS_API_KEY, DEXSCREENER_API_URL
)

logger = logging.getLogger(__name__)

class SolanaManager:
    """Manages Solana blockchain"""
    
    def __init__(self):
        self.rpc_urls = RPC_URLS
        self.current_rpc_index = 0
        self.client = Client(self.rpc_urls[0])
        self.async_client = AsyncClient(self.rpc_urls[0])
        
        # Init wallet
        if WALLET_PRIVATE_KEY:
            secret_key_bytes = base58.b58decode(WALLET_PRIVATE_KEY)
            self.keypair = Keypair.from_bytes(secret_key_bytes)
        else:
            logger.warning("No wallet private key configured")
            self.keypair = None
        
        # Init Jupiter
        self.jupiter = self._init_jupiter()
    
    def _init_jupiter(self) -> Optional[Jupiter]:
        """Init Jupiter"""
        if not self.keypair:
            return None
            
        return Jupiter(
            async_client=self.async_client,
            keypair=self.keypair,
            quote_api_url=JUPITER_API_URL,
            swap_api_url=JUPITER_SWAP_URL
        )
    
    def switch_rpc(self):
        """Switch to next RPC URL if multiple are configured"""
        if len(self.rpc_urls) > 1:
            self.current_rpc_index = (self.current_rpc_index + 1) % len(self.rpc_urls)
            new_url = self.rpc_urls[self.current_rpc_index]
            self.client = Client(new_url)
            self.async_client = AsyncClient(new_url)
            logger.info(f"Switched to RPC: {new_url}")
    
    def get_token_balance(self, token_contract: str, wallet_address: str) -> float:
        """Get token balance for a wallet"""
        try:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getTokenAccountsByOwner",
                "params": [
                    wallet_address,
                    {"mint": token_contract},
                    {"encoding": "jsonParsed"},
                ]
            }
            
            response = requests.post(self.rpc_urls[self.current_rpc_index], json=payload)
            response.raise_for_status()
            result = response.json()
            
            if result["result"]["value"]:
                balance = result["result"]["value"][0]["account"]["data"]["parsed"]["info"]["tokenAmount"]["uiAmount"]
                return float(balance)
            return 0.0
            
        except Exception as e:
            logger.error(f"Error getting token balance: {e}")
            return 0.0
    
    def get_token_decimals(self, token_contract: str) -> int:
        """Get token decimals"""
        try:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getTokenSupply",
                "params": [token_contract]
            }
            
            response = requests.post(self.rpc_urls[self.current_rpc_index], json=payload)
            result = response.json()
            return int(result['result']['value']['decimals'])
            
        except Exception as e:
            logger.error(f"Error getting token decimals: {e}")
            return 9  # Default to 9 decimals
    
    def get_token_holders(self, token_address: str) -> int:
        """Get number of token holders"""
        
        # NOTE: this is available only on helius as it's custom function
        try:
            response = requests.post(
                f"https://mainnet.helius-rpc.com/?api-key={HELIUS_API_KEY}",
                headers={"Content-Type": "application/json"},
                json={
                    "jsonrpc": "2.0",
                    "id": "text",
                    "method": "getTokenAccounts",
                    "params": {
                        "mint": token_address,
                        "page": 1,
                        "limit": 251
                    }
                }
            )
            data = response.json()
            return data['result']['total']
            
        except Exception as e:
            logger.error(f"Error getting token holders: {e}")
            return 0
    
    async def check_transaction(self, tx_id: str) -> Optional[bool]:
        """Check transaction status"""
        try:
            response = self.client.get_transaction(
                Signature(b58decode(tx_id)),
                commitment="confirmed",
                max_supported_transaction_version=0
            )
            
            if response and str(response) != 'GetTransactionResp(None)':
                if "failed" in str(response):
                    logger.info(f"Transaction failed: {tx_id}")
                    return False
                logger.info(f"Transaction successful: {tx_id}")
                return True
            
            return None  # Transaction not found yet
            
        except Exception as e:
            logger.error(f"Error checking transaction: {e}")
            return None

class DexScreenerAPI:
    """DexScreener API integration"""
    
    @staticmethod
    def get_token_info(contract_address: str) -> Optional[Dict[str, Any]]:
        """Get token information from DexScreener"""
        try:
            response = requests.get(f"{DEXSCREENER_API_URL}{contract_address}")
            data = response.json().get("pairs", [])
            
            if not data:
                return None
            
            pair = data[0]
            return {
                'symbol': pair['baseToken']['symbol'],
                'price_usd': float(pair['priceUsd']),
                'liquidity_usd': float(pair['liquidity']['usd']),
                'market_cap': float(pair.get('marketCap', 0)),
                'created_at': pair.get('pairCreatedAt', 0),
                'volume_24h': float(pair.get('volume', {}).get('h24', 0))
            }
            
        except Exception as e:
            logger.error(f"Error fetching from DexScreener: {e}")
            return None
    
    @staticmethod
    def get_jupiter_price(token_address: str) -> Optional[float]:
        """Get token price from Jupiter"""
        try:
            response = requests.get(f"https://api.jup.ag/price/v2?ids={token_address}")
            
            if response.status_code != 200:
                return None
            
            data = response.json()
            if "data" in data and token_address in data["data"]:
                return float(data["data"][token_address]["price"])
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching Jupiter price: {e}")
            return None