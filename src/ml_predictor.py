"""ML predictor for trading"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class TradePredictor:
    
    def __init__(self):
        # classifier for profit/loss
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        # regression for percent prediction
        self.regressor = LinearRegression()
        
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def load_backtest_data(self, csv_path='completed_trades.csv'):
        """load data from csv"""
        try:
            df = pd.read_csv(csv_path)
            expected_cols = ['timestamp', 'mint', 'status', 'type', 'firstPrice', 'lastPrice']
            if len(df.columns) == 6:
                df.columns = expected_cols
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['firstPrice'] = pd.to_numeric(df['firstPrice'], errors='coerce')
            df['lastPrice'] = pd.to_numeric(df['lastPrice'], errors='coerce')
            
            # calculate percent change
            df['percent_change'] = ((df['lastPrice'] - df['firstPrice']) / df['firstPrice']) * 100
            
            # profitable or not
            df['is_profitable'] = df['status'].str.contains('profit', case=False, na=False)
            
            print(f"Loaded {len(df)} trades")
            return df
        except FileNotFoundError:
            print("file not found, generating sample data...")
            return self.generate_sample_data()
    
    def generate_sample_data(self):
        """generate sample data if no real data available"""
        np.random.seed(42)
        n = 400
        
        data = []
        for i in range(n):
            ttype = f'TYPE{np.random.randint(1, 8)}'
            type_n = int(ttype.replace('TYPE', ''))
            
            # TYPE1-TYPE3 better, TYPE6-7 worse
            prob = 0.75 - (type_n * 0.06)
            is_profit = np.random.random() < prob
            
            fprice = np.random.uniform(0.0001, 5.0)
            
            if is_profit:
                stat = np.random.choice(['profit', 'profit_ETA', 'profit_1min'])
                chg = np.random.uniform(1.0, 15.0)
            else:
                stat = np.random.choice(['stop', 'stop_10', 'stop_1min'])
                chg = np.random.uniform(-10.0, -0.5)
            
            lprice = fprice * (1 + chg / 100)
            ts = datetime.now() - pd.Timedelta(days=n-i, hours=np.random.randint(0, 24))
            
            data.append({
                'timestamp': ts,
                'mint': f'Token{i}pump',
                'status': stat,
                'type': ttype,
                'firstPrice': fprice,
                'lastPrice': lprice,
                'percent_change': chg,
                'is_profitable': is_profit
            })
        
        return pd.DataFrame(data)
    
    def create_features(self, df):
        """create features for the model"""
        feats = pd.DataFrame()
        
        # price in different forms
        feats['price'] = df['firstPrice']
        feats['price_log'] = np.log1p(df['firstPrice'])
        feats['price_scaled'] = df['firstPrice'] / df['firstPrice'].mean()
        
        # type
        feats['type_num'] = df['type'].str.extract(r'(\d+)').astype(float)
        
        # time
        feats['hour'] = df['timestamp'].dt.hour
        feats['day_of_week'] = df['timestamp'].dt.dayofweek
        feats['is_weekend'] = (feats['day_of_week'] >= 5).astype(int)
        feats['hour_sin'] = np.sin(2 * np.pi * feats['hour'] / 24)
        feats['hour_cos'] = np.cos(2 * np.pi * feats['hour'] / 24)
        
        # rolling windows for history
        windows = [5, 10, 20]
        for w in windows:
            feats[f'win_rate_{w}'] = df['is_profitable'].rolling(w, min_periods=1).mean().fillna(0.5)
            feats[f'avg_return_{w}'] = df['percent_change'].rolling(w, min_periods=1).mean().fillna(0)
            feats[f'volatility_{w}'] = df['percent_change'].rolling(w, min_periods=1).std().fillna(1)
        
        # stats by type
        type_stats = df.groupby('type').agg({
            'is_profitable': 'mean',
            'percent_change': ['mean', 'std']
        }).fillna(0)
        type_stats.columns = ['type_win_rate', 'type_avg_return', 'type_volatility']
        
        df_merged = df.merge(type_stats, left_on='type', right_index=True, how='left')
        for col in type_stats.columns:
            feats[col] = df_merged[col].fillna(0.5 if 'win_rate' in col else 0)
        
        # momentum and risk
        feats['price_momentum'] = df['firstPrice'].pct_change(3).fillna(0)
        feats['risk_score'] = feats['type_num'] * 0.4 + (1 / (1 + feats['price'])) * 0.6
        
        return feats.fillna(0)
    
    def train(self, df=None):
        """train the models"""
        if df is None:
            df = self.load_backtest_data()
        
        print(f"training on {len(df)} trades...")
        
        # create features
        X = self.create_features(df)
        y_class = df['is_profitable'].astype(int)
        y_reg = df['percent_change']
        
        self.feature_names = X.columns.tolist()
        
        # split train/test
        X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
            X, y_class, y_reg, test_size=0.2, random_state=42, stratify=y_class
        )
        
        # scale
        X_train_sc = self.scaler.fit_transform(X_train)
        X_test_sc = self.scaler.transform(X_test)
        
        # train classifier
        print("training classifier...")
        self.classifier.fit(X_train_sc, y_class_train)
        
        # train regressor
        print("training regressor...")
        self.regressor.fit(X_train_sc, y_reg_train)
        
        # check results
        self.evaluate(X_test_sc, y_class_test, y_reg_test, df)
        
        self.is_fitted = True
        return self
    
    def evaluate(self, X_test, y_class_test, y_reg_test, df):
        """check model quality"""
        # classification
        y_pred_c = self.classifier.predict(X_test)
        
        print("\n" + "="*60)
        print("CLASSIFIER RESULTS")
        print("="*60)
        print(classification_report(y_class_test, y_pred_c, 
                                   target_names=['Loss', 'Profit']))
        
        # regression
        y_pred_r = self.regressor.predict(X_test)
        mae = mean_absolute_error(y_reg_test, y_pred_r)
        r2 = r2_score(y_reg_test, y_pred_r)
        
        print("\n" + "="*60)
        print("REGRESSOR RESULTS")
        print("="*60)
        print(f"MAE: {mae:.2f}%")
        print(f"R2: {r2:.3f}")
        
        # feature importance
        print("\n" + "="*60)
        print("TOP-10 IMPORTANT FEATURES")
        print("="*60)
        feat_imp = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.classifier.feature_importances_
        }).sort_values('importance', ascending=False).head(10)
        
        for idx, row in feat_imp.iterrows():
            print(f"{row['feature']}: {row['importance']:.4f}")
        
        # regression coefficients
        print("\n" + "="*60)
        print("TOP-10 REGRESSION COEFFICIENTS")
        print("="*60)
        coeffs = pd.DataFrame({
            'feature': self.feature_names,
            'coef': self.regressor.coef_
        }).sort_values('coef', key=abs, ascending=False).head(10)
        
        for idx, row in coeffs.iterrows():
            print(f"{row['feature']}: {row['coef']:.4f}")
    
    def predict_trade(self, trade_type, first_price, recent_perf=None):
        """predict for new trade"""
        if not self.is_fitted:
            raise ValueError("model not trained, call .train() first")
        
        # extract type number
        tnum = int(trade_type.replace('TYPE', ''))
        
        # collect all features
        feat_dict = {
            'price': first_price,
            'price_log': np.log1p(first_price),
            'price_scaled': first_price / 0.5,  # approximate mean
            'type_num': tnum,
            'hour': datetime.now().hour,
            'day_of_week': datetime.now().weekday(),
            'is_weekend': 1 if datetime.now().weekday() >= 5 else 0,
            'hour_sin': np.sin(2 * np.pi * datetime.now().hour / 24),
            'hour_cos': np.cos(2 * np.pi * datetime.now().hour / 24),
            'price_momentum': 0,
            'risk_score': tnum * 0.4 + (1 / (1 + first_price)) * 0.6
        }
        
        # add rolling windows
        for w in [5, 10, 20]:
            if recent_perf:
                feat_dict[f'win_rate_{w}'] = recent_perf.get(f'win_rate_{w}', 0.5)
                feat_dict[f'avg_return_{w}'] = recent_perf.get(f'avg_return_{w}', 0)
                feat_dict[f'volatility_{w}'] = recent_perf.get(f'volatility_{w}', 1)
            else:
                feat_dict[f'win_rate_{w}'] = 0.5
                feat_dict[f'avg_return_{w}'] = 0
                feat_dict[f'volatility_{w}'] = 1
        
        # type stats (using defaults for now)
        feat_dict['type_win_rate'] = 0.5
        feat_dict['type_avg_return'] = 0
        feat_dict['type_volatility'] = 1
        
        feats = pd.DataFrame([feat_dict])
        
        # make sure everything is there
        for col in self.feature_names:
            if col not in feats.columns:
                feats[col] = 0
        
        feats = feats[self.feature_names]
        
        # scale
        feats_sc = self.scaler.transform(feats)
        
        # get predictions
        prob = self.classifier.predict_proba(feats_sc)[0][1]
        ret = self.regressor.predict(feats_sc)[0]
        
        # build result
        result = {
            'trade_type': trade_type,
            'price': first_price,
            'profit_probability': prob,
            'expected_return': ret,
            'recommendation': 'BUY' if prob > 0.6 and ret > 2 else 'SKIP',
            'risk_level': 'LOW' if prob > 0.7 else 'MEDIUM' if prob > 0.5 else 'HIGH'
        }
        
        return result
    
    def plot_analysis(self, df=None):
        """visualize results"""
        if df is None:
            df = self.load_backtest_data()
        
        plt.style.use('dark_background')
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # win rate by type
        ax1 = axes[0, 0]
        wr = df.groupby('type')['is_profitable'].mean().sort_values(ascending=False)
        wr.plot(kind='bar', ax=ax1, color='cyan', edgecolor='white')
        ax1.set_title('Win Rate by Type', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Win Rate')
        ax1.set_xlabel('Type')
        ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='50%')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # return distribution
        ax2 = axes[0, 1]
        profit_rets = df[df['is_profitable']]['percent_change']
        loss_rets = df[~df['is_profitable']]['percent_change']
        ax2.hist(profit_rets, bins=30, alpha=0.6, color='green', label='Profit')
        ax2.hist(loss_rets, bins=30, alpha=0.6, color='red', label='Loss')
        ax2.set_title('Return Distribution', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Return (%)')
        ax2.set_ylabel('Frequency')
        ax2.axvline(x=0, color='white', linestyle='--', alpha=0.7)
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # cumulative returns
        ax3 = axes[1, 0]
        df_sorted = df.sort_values('timestamp')
        cumul = (1 + df_sorted['percent_change'] / 100).cumprod()
        ax3.plot(range(len(cumul)), cumul, color='lime', linewidth=2)
        ax3.fill_between(range(len(cumul)), 1, cumul, alpha=0.3, color='lime')
        ax3.set_title('Cumulative Returns', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Trade Number')
        ax3.set_ylabel('Multiplier')
        ax3.axhline(y=1, color='white', linestyle='--', alpha=0.7)
        ax3.grid(alpha=0.3)
        
        # average return by type
        ax4 = axes[1, 1]
        avg_ret = df.groupby('type')['percent_change'].mean().sort_values(ascending=False)
        colors = ['green' if x > 0 else 'red' for x in avg_ret.values]
        avg_ret.plot(kind='barh', ax=ax4, color=colors, edgecolor='white')
        ax4.set_title('Average Return by Type', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Average Return (%)')
        ax4.set_ylabel('Type')
        ax4.axvline(x=0, color='white', linestyle='--', alpha=0.7)
        ax4.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('ml_analysis.png', dpi=100, bbox_inches='tight', facecolor='#1a1a1a')
        print("\nplot saved to ml_analysis.png")
        plt.show()
    
    def save_model(self, filepath='trading_model.pkl'):
        """save model"""
        if not self.is_fitted:
            raise ValueError("model not trained")
        
        model_data = {
            'classifier': self.classifier,
            'regressor': self.regressor,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        
        joblib.dump(model_data, filepath)
        print(f"model saved to {filepath}")
    
    def load_model(self, filepath='trading_model.pkl'):
        """load model"""
        data = joblib.load(filepath)
        self.classifier = data['classifier']
        self.regressor = data['regressor']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.is_fitted = True
        print(f"model loaded from {filepath}")


def run_analysis():
    """run full analysis"""
    print("="*60)
    print("ML TRADING STRATEGY ANALYSIS")
    print("="*60)
    
    predictor = TradePredictor()
    df = predictor.load_backtest_data()
    predictor.train(df)
    
    # visualization
    predictor.plot_analysis(df)
    
    # test predictions
    print("\n" + "="*60)
    print("TEST PREDICTIONS")
    print("="*60)
    
    test_cases = [
        ('TYPE1', 0.001),
        ('TYPE3', 0.005),
        ('TYPE5', 0.003),
        ('TYPE7', 0.002)
    ]
    
    for ttype, price in test_cases:
        pred = predictor.predict_trade(ttype, price)
        print(f"\n{ttype} @ ${price:.6f}:")
        print(f"   Profit probability: {pred['profit_probability']:.1%}")
        print(f"   Expected return: {pred['expected_return']:.2f}%")
        print(f"   Recommendation: {pred['recommendation']}")
        print(f"   Risk: {pred['risk_level']}")
    
    predictor.save_model()
    
    print("\n" + "="*60)
    print("analysis complete!")
    print("="*60)
    
    return predictor


if __name__ == "__main__":
    predictor = run_analysis()