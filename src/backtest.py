import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from typing import Dict, List, Tuple, Any
from datetime import datetime

class Backtester:
    """Backtesting system for the DCA trading bot"""
    
    def __init__(self, csv_filepath: str = 'completed_trades.csv', initial_balance: float = 20000):
        self.csv_filepath = csv_filepath
        self.initial_balance = initial_balance
        self.position_size = 0.03  # 3% of balance per trade
        self.commission = 0.5  # $0.5 commission per trade
        self.entry_slippage = 0.005  # 0.5% slippage on entry
        
        self.trades_df = None
        self.balance_history = []
        self.trade_info = []
        
    def load_trades(self) -> pd.DataFrame:
        """Load trades from CSV file"""
        try:
            # Try to load CSV
            self.trades_df = pd.read_csv(self.csv_filepath, 
                names=['timestamp', 'mint', 'status', 'type', 'firstPrice', 'lastPrice'],
                header=0)
            
            # Convert timestamp to datetime
            self.trades_df['timestamp'] = pd.to_datetime(self.trades_df['timestamp'])
            
            # Convert prices to numeric
            self.trades_df['firstPrice'] = pd.to_numeric(self.trades_df['firstPrice'], errors='coerce')
            self.trades_df['lastPrice'] = pd.to_numeric(self.trades_df['lastPrice'], errors='coerce')
            
            # Calculate percent change
            self.trades_df['percent_change'] = ((self.trades_df['lastPrice'] - self.trades_df['firstPrice']) / 
                                                 self.trades_df['firstPrice']) * 100
            
            return self.trades_df
            
        except FileNotFoundError:
            print(f"No trades file found at {self.csv_filepath}")
            print("Creating sample data for demonstration...")
            # Create sample data for demonstration
            self.trades_df = self.create_sample_data()
            return self.trades_df
    
    def create_sample_data(self) -> pd.DataFrame:
        """Create sample trading data for demonstration"""
        sample_data = [
            ['2025-03-01 10:00:00', 'Token1ABC...pump', 'profit', 'TYPE4', 0.001, 0.0011],
            ['2025-03-01 11:30:00', 'Token2XYZ...pump', 'stop', 'TYPE5', 0.005, 0.0047],
            ['2025-03-01 14:00:00', 'Token3DEF...pump', 'profit', 'TYPE3', 0.002, 0.0022],
            ['2025-03-01 16:45:00', 'Token4GHI...pump', 'profit_ETA', 'TYPE4', 0.0015, 0.00157],
            ['2025-03-01 18:20:00', 'Token5JKL...pump', 'stop_1min', 'TYPE5', 0.008, 0.0076],
            ['2025-03-02 09:15:00', 'Token6MNO...pump', 'profit', 'TYPE1', 0.003, 0.0034],
            ['2025-03-02 12:00:00', 'Token7PQR...pump', 'stop_10', 'TYPE4', 0.004, 0.0036],
            ['2025-03-02 15:30:00', 'Token8STU...pump', 'profit', 'TYPE3', 0.0025, 0.0028],
            ['2025-03-02 18:00:00', 'Token9VWX...pump', 'profit_1min', 'TYPE5', 0.006, 0.0063],
            ['2025-03-03 10:00:00', 'Token10YZ...pump', 'stop', 'TYPE4', 0.007, 0.0065],
            ['2025-03-03 12:00:00', 'Token11ABC...pump', 'profit', 'TYPE2', 0.0045, 0.0050],
            ['2025-03-03 14:30:00', 'Token12DEF...pump', 'stop', 'TYPE3', 0.0032, 0.0030],
            ['2025-03-03 17:00:00', 'Token13GHI...pump', 'profit', 'TYPE4', 0.0018, 0.0020],
            ['2025-03-04 09:00:00', 'Token14JKL...pump', 'profit', 'TYPE1', 0.0055, 0.0062],
            ['2025-03-04 11:15:00', 'Token15MNO...pump', 'stop_10', 'TYPE5', 0.0042, 0.0038],
            ['2025-03-04 13:45:00', 'Token16PQR...pump', 'profit', 'TYPE3', 0.0028, 0.0031],
            ['2025-03-04 16:00:00', 'Token17STU...pump', 'stop', 'TYPE4', 0.0065, 0.0060],
            ['2025-03-04 18:30:00', 'Token18VWX...pump', 'profit_ETA', 'TYPE2', 0.0038, 0.0041],
            ['2025-03-05 10:00:00', 'Token19YZ1...pump', 'profit', 'TYPE5', 0.0022, 0.0025],
            ['2025-03-05 12:30:00', 'Token20ABC...pump', 'stop', 'TYPE1', 0.0048, 0.0045],
        ]
        
        df = pd.DataFrame(sample_data, columns=['timestamp', 'mint', 'status', 'type', 'firstPrice', 'lastPrice'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['firstPrice'] = pd.to_numeric(df['firstPrice'])
        df['lastPrice'] = pd.to_numeric(df['lastPrice'])
        df['percent_change'] = ((df['lastPrice'] - df['firstPrice']) / df['firstPrice']) * 100
        
        return df
    
    def simulate_trading(self):
        """Simulate trading with position sizing and commissions"""
        if self.trades_df is None or self.trades_df.empty:
            print("No trades data to simulate")
            return
        
        current_balance = self.initial_balance
        self.balance_history = [(self.trades_df['timestamp'].iloc[0], current_balance)]
        self.trade_info = []
        
        for index, row in self.trades_df.iterrows():
            # Calculate position size
            position_amount = current_balance * self.position_size
            
            # Apply entry slippage (simulating worse entry price)
            adjusted_first_price = row['firstPrice'] * (1 + self.entry_slippage)
            adjusted_percent_change = ((row['lastPrice'] - adjusted_first_price) / adjusted_first_price) * 100
            
            # Calculate P&L
            profit_loss = position_amount * (adjusted_percent_change / 100)
            
            # Apply commission
            current_balance += profit_loss - self.commission
            
            # Store trade info
            self.balance_history.append((row['timestamp'], current_balance))
            self.trade_info.append({
                'datetime': row['timestamp'],
                'balance': current_balance,
                'trade_num': index + 1,
                'mint': row['mint'][:12] + '...' if len(row['mint']) > 12 else row['mint'],
                'status': row['status'],
                'type': row['type'],
                'percent_change': adjusted_percent_change,
                'profit_loss': profit_loss,
                'is_profit': 'profit' in row['status'].lower()
            })
        
        print(f"Simulation complete. Processed {len(self.trade_info)} trades.")
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive trading metrics"""
        if not self.trade_info:
            return {}
        
        df = pd.DataFrame(self.trade_info)
        
        # Basic metrics
        total_trades = len(df)
        winning_trades = len(df[df['is_profit']])
        losing_trades = total_trades - winning_trades
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Financial metrics
        profits = df[df['is_profit']]['profit_loss']
        losses = df[~df['is_profit']]['profit_loss']
        
        total_profit = profits.sum() if len(profits) > 0 else 0
        total_loss = losses.sum() if len(losses) > 0 else 0
        net_profit = total_profit + total_loss
        
        avg_win = profits.mean() if len(profits) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0
        
        # Risk metrics
        profit_factor = abs(total_profit / total_loss) if total_loss != 0 else float('inf')
        
        # Calculate max drawdown
        running_max = self.initial_balance
        max_drawdown = 0
        
        for _, balance in self.balance_history:
            if balance > running_max:
                running_max = balance
            drawdown = (running_max - balance) / running_max * 100
            max_drawdown = max(max_drawdown, drawdown)
        
        # Calculate Sharpe ratio (simplified)
        returns = df['percent_change'].values
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'net_profit': net_profit,
            'total_return': ((self.balance_history[-1][1] - self.initial_balance) / self.initial_balance * 100),
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'final_balance': self.balance_history[-1][1]
        }
    
    def plot_results(self):
        """Create comprehensive visualization of trading results"""
        if not self.balance_history or not self.trade_info:
            print("No data to plot")
            return
        
        # Set dark theme
        plt.style.use('dark_background')
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 10))
        
        # Main balance chart
        ax1 = plt.subplot(2, 2, (1, 2))
        balance_df = pd.DataFrame(self.balance_history, columns=['datetime', 'balance'])
        ax1.plot(balance_df['datetime'], balance_df['balance'], 'b-', linewidth=2, label='Balance')
        
        # Mark trades
        for trade in self.trade_info:
            color = 'lime' if trade['is_profit'] else 'red'
            marker = '^' if trade['is_profit'] else 'v'
            ax1.plot(trade['datetime'], trade['balance'], color=color, marker=marker, markersize=8, alpha=0.7)
            
            # Add annotations for significant trades (top 5 gains and losses)
            if abs(trade['percent_change']) > 5:
                ax1.annotate(
                    f"{trade['type']}\n{trade['percent_change']:.1f}%",
                    xy=(trade['datetime'], trade['balance']),
                    xytext=(10, 10 if trade['is_profit'] else -20),
                    textcoords='offset points',
                    fontsize=7,
                    bbox=dict(boxstyle='round,pad=0.3', fc=color, alpha=0.3),
                    arrowprops=dict(arrowstyle='->', color=color, alpha=0.5)
                )
        
        ax1.set_title('Balance History with Trade Markers', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Balance (USD)')
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        plt.xticks(rotation=45)
        
        # Profit/Loss distribution
        ax2 = plt.subplot(2, 2, 3)
        trade_df = pd.DataFrame(self.trade_info)
        profits = trade_df[trade_df['is_profit']]['profit_loss']
        losses = trade_df[~trade_df['is_profit']]['profit_loss']
        
        if len(profits) > 0 or len(losses) > 0:
            ax2.hist([profits, losses], bins=10, label=['Profits', 'Losses'], 
                    color=['green', 'red'], alpha=0.7, edgecolor='white', linewidth=0.5)
        ax2.set_title('Profit/Loss Distribution')
        ax2.set_xlabel('P&L (USD)')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Performance by type
        ax3 = plt.subplot(2, 2, 4)
        type_performance = trade_df.groupby('type').agg({
            'profit_loss': 'sum',
            'is_profit': lambda x: (x.sum() / len(x) * 100)  # Win rate by type
        })
        
        if not type_performance.empty:
            x = np.arange(len(type_performance.index))
            width = 0.35
            
            bars1 = ax3.bar(x - width/2, type_performance['profit_loss'], width, 
                           label='Total P&L', color='blue', alpha=0.7)
            
            ax3_twin = ax3.twinx()
            bars2 = ax3_twin.bar(x + width/2, type_performance['is_profit'], width, 
                                label='Win Rate %', color='orange', alpha=0.7)
            
            ax3.set_xlabel('Strategy Type')
            ax3.set_ylabel('Total P&L (USD)', color='blue')
            ax3_twin.set_ylabel('Win Rate (%)', color='orange')
            ax3.set_xticks(x)
            ax3.set_xticklabels(type_performance.index, rotation=45)
            ax3.set_title('Performance by Strategy Type')
            ax3.grid(True, alpha=0.3)
        
        # Add statistics text box
        metrics = self.calculate_metrics()
        stats_text = (
            f"Initial Balance: ${self.initial_balance:,.2f}\n"
            f"Final Balance: ${metrics['final_balance']:,.2f}\n"
            f"Total Return: {metrics['total_return']:.1f}%\n"
            f"Total Trades: {metrics['total_trades']}\n"
            f"Win Rate: {metrics['win_rate']:.1f}%\n"
            f"Profit Factor: {metrics['profit_factor']:.2f}\n"
            f"Max Drawdown: {metrics['max_drawdown']:.1f}%"
        )
        
        fig.text(0.02, 0.02, stats_text, fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.8),
                verticalalignment='bottom')
        
        plt.tight_layout()
        
        # Save and show figure
        plt.savefig('backtest_results.png', dpi=100, bbox_inches='tight')
        print("Chart saved as 'backtest_results.png'")
        plt.show()
    
    def print_report(self):
        """Print detailed trading report"""
        metrics = self.calculate_metrics()
        
        if not metrics:
            print("No trading data available")
            return
        
        print("\n" + "=" * 60)
        print("TRADING PERFORMANCE REPORT")
        print("=" * 60)
        
        print("\n📊 OVERALL STATISTICS")
        print("-"*40)
        print(f"Total Trades:      {metrics['total_trades']}")
        print(f"Winning Trades:    {metrics['winning_trades']} ({metrics['win_rate']:.1f}%)")
        print(f"Losing Trades:     {metrics['losing_trades']} ({100 - metrics['win_rate']:.1f}%)")
        
        print("\n💰 FINANCIAL METRICS")
        print("-" * 40)
        print(f"Initial Balance:   ${self.initial_balance:,.2f}")
        print(f"Final Balance:     ${metrics['final_balance']:,.2f}")
        print(f"Net Profit:        ${metrics['net_profit']:,.2f}")
        print(f"Total Return:      {metrics['total_return']:.2f}%")
        print(f"Average Win:       ${metrics['avg_win']:,.2f}")
        print(f"Average Loss:      ${metrics['avg_loss']:,.2f}")
        
        print("\n📈 RISK METRICS")
        print("-" * 40)
        print(f"Profit Factor:     {metrics['profit_factor']:.2f}")
        print(f"Max Drawdown:      {metrics['max_drawdown']:.1f}%")
        print(f"Sharpe Ratio:      {metrics['sharpe_ratio']:.2f}")
        
        print("\n🎯 SAMPLE TRADES")
        print("-" * 40)
        if self.trade_info:
            # Show first 3 trades as examples
            for i, trade in enumerate(self.trade_info[:3]):
                status = "✅" if trade['is_profit'] else "❌"
                print(f"Trade #{i+1}: {status} {trade['type']} | "
                      f"P&L: ${trade['profit_loss']:.2f} ({trade['percent_change']:.1f}%)")
        
        print("="*60)

def run_backtest(csv_file: str = 'completed_trades.csv', initial_balance: float = 20000):
    """Run complete backtest analysis"""
    print("Starting backtest analysis...")
    
    backtester = Backtester(csv_file, initial_balance)
    
    # Load and process trades
    trades = backtester.load_trades()
    
    if trades.empty:
        print("No trades to analyze")
        return backtester
    
    print(f"Loaded {len(trades)} trades")
    
    # Run simulation
    backtester.simulate_trading()
    
    # Print report
    backtester.print_report()
    
    # Create visualizations
    backtester.plot_results()
    
    return backtester

if __name__ == "__main__":
    # Run backtest
    backtester = run_backtest()