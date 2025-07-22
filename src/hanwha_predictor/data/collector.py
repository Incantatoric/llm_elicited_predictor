import yfinance as yf
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('llm_stock_predictor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HanwhaDataCollector:
    """
    Collects and preprocesses data for 한화솔루션 stock prediction
    """
    
    def __init__(self):
        self.variables = {
            'hanwha_stock': '009830.KS',    # 한화솔루션
            'kospi': '^KS11',               # KOSPI Index
            'oil_price': 'CL=F',            # WTI Crude Oil
            'usd_krw': 'KRW=X',             # USD/KRW Exchange Rate
            'vix': '^VIX',                  # VIX Volatility Index
            'materials_etf': 'XLB'          # Materials Sector ETF (as chemical proxy)
        }
        
        # Date range: exactly 36 months ending June 2025 for proper 30 train + 6 test split
        self.end_date = datetime(2025, 6, 30)
        
        # Calculate start date for exactly 36 months of monthly returns
        # We need 37 months of price data to get 36 months of returns (pct_change drops first)
        # 37 months ending June 2025 means starting from June 2022
        self.start_date = datetime(2022, 6, 1)  # Start from June 1 to capture June 30 month-end
        
        logger.info(f"Initialized HanwhaDataCollector")
        logger.info(f"Date range: {self.start_date.date()} to {self.end_date.date()}")
        logger.info(f"Variables to collect: {list(self.variables.keys())}")
        
    def download_raw_data(self) -> Dict[str, pd.DataFrame]:
        """
        Download raw daily data from Yahoo Finance
        """
        logger.info("Starting raw data download from Yahoo Finance...")
        
        raw_data = {}
        failed_downloads = []
        
        for name, ticker in self.variables.items():
            try:
                logger.info(f"Downloading {name} ({ticker})...")
                
                data = yf.download(
                    ticker, 
                    start=self.start_date, 
                    end=self.end_date,
                    progress=False
                )
                
                if data.empty:
                    logger.warning(f"No data returned for {name} ({ticker})")
                    failed_downloads.append(name)
                    continue
                    
                # Log data info
                logger.info(f"{name}: {len(data)} days, {data.index[0].date()} to {data.index[-1].date()}")
                logger.info(f"{name} sample data:\n{data.head(2)}")
                
                raw_data[name] = data
                
            except Exception as e:
                logger.error(f"Failed to download {name} ({ticker}): {str(e)}")
                failed_downloads.append(name)
                
        if failed_downloads:
            logger.warning(f"Failed to download: {failed_downloads}")
            
        logger.info(f"Successfully downloaded {len(raw_data)} out of {len(self.variables)} variables")
        return raw_data
    
    def calculate_monthly_returns(self, raw_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate monthly returns from end-of-month prices
        """
        logger.info("Calculating monthly returns from end-of-month prices...")
        
        monthly_data = {}
        
        for name, data in raw_data.items():
            try:
                # Handle MultiIndex columns - get the Close price Series
                if isinstance(data.columns, pd.MultiIndex):
                    # If MultiIndex, get the first ticker's Close price
                    close_prices = data['Close'].iloc[:, 0]
                else:
                    # If regular columns, get Close directly
                    close_prices = data['Close']
                
                # Get end-of-month closing prices
                monthly_prices = close_prices.resample('M').last()
                
                # Calculate simple returns: (P_t - P_{t-1}) / P_{t-1}
                monthly_returns = monthly_prices.pct_change().dropna()
                
                monthly_data[name] = monthly_returns
                
                logger.info(f"{name} monthly returns: {len(monthly_returns)} months")
                
                # Safe logging of statistics
                mean_val = monthly_returns.mean()
                std_val = monthly_returns.std()
                if pd.isna(mean_val) or pd.isna(std_val):
                    logger.warning(f"{name} return stats: mean=NaN, std=NaN")
                else:
                    logger.info(f"{name} return stats: mean={mean_val:.4f}, std={std_val:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to calculate monthly returns for {name}: {str(e)}")
                continue
                
        # Check if we have any valid data
        if not monthly_data:
            raise ValueError("No valid monthly returns data could be calculated")
            
        # Combine all returns into single DataFrame
        monthly_df = pd.DataFrame(monthly_data)
        
        # Log combined dataset info
        logger.info(f"Combined monthly returns dataset shape: {monthly_df.shape}")
        logger.info(f"Date range: {monthly_df.index[0].date()} to {monthly_df.index[-1].date()}")
        logger.info(f"Missing values per variable:\n{monthly_df.isnull().sum()}")
        
        return monthly_df
    
    def prepare_features_and_target(self, monthly_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Prepare features (X) and target (y) for modeling
        """
        logger.info("Preparing features and target variables...")
        
        # Target: 한화솔루션 returns
        target_col = 'hanwha_stock'
        if target_col not in monthly_df.columns:
            raise ValueError(f"Target variable {target_col} not found in data")
            
        y = monthly_df[target_col].copy()
        
        # Features: all other variables
        feature_cols = [col for col in monthly_df.columns if col != target_col]
        X = monthly_df[feature_cols].copy()
        
        # Handle missing values by forward filling then dropping remaining NaNs
        logger.info("Handling missing values...")
        original_length = len(X)
        
        X = X.fillna(method='ffill').dropna()
        y = y.loc[X.index]  # Align target with features
        
        logger.info(f"Dropped {original_length - len(X)} rows due to missing values")
        logger.info(f"Final dataset shape: X={X.shape}, y={len(y)}")
        
        # Feature names for LLM prompts (more descriptive)
        feature_names = [
            'kospi_return',           # KOSPI market return
            'oil_price_change',       # Oil price monthly change  
            'usd_krw_change',         # Exchange rate change
            'vix_change',             # Volatility index change
            'materials_sector_return' # Materials sector return
        ]
        
        # Ensure we have the right number of features
        if len(feature_names) != X.shape[1]:
            logger.warning(f"Feature name count ({len(feature_names)}) doesn't match data columns ({X.shape[1]})")
            logger.warning(f"Data columns: {list(X.columns)}")
            # Use data column names as fallback
            feature_names = list(X.columns)
            
        logger.info(f"Feature names for LLM: {feature_names}")
        
        return X, y, feature_names
    
    def standardize_features(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        """
        Standardize features using z-score (mean=0, std=1)
        This matches what we tell the LLM in prompts
        """
        logger.info("Standardizing features (z-score normalization)...")
        
        # Calculate and store statistics for later use
        standardization_stats = {
            'means': X.mean().to_dict(),
            'stds': X.std().to_dict()
        }
        
        # Standardize
        X_standardized = (X - X.mean()) / X.std()
        
        logger.info("Standardization stats:")
        for col in X.columns:
            mean_val = standardization_stats['means'][col]
            std_val = standardization_stats['stds'][col]
            logger.info(f"  {col}: mean={mean_val:.6f}, std={std_val:.6f}")
            
        # Verify standardization
        logger.info("Verification - standardized data stats:")
        for col in X_standardized.columns:
            mean_check = X_standardized[col].mean()
            std_check = X_standardized[col].std()
            logger.info(f"  {col}: mean={mean_check:.6f}, std={std_check:.6f}")
            
        return X_standardized, standardization_stats
    
    def save_data_to_files(self, X: pd.DataFrame, y: pd.Series, feature_names: List[str], 
                       standardization_stats: dict, data_dir: str = "data/processed"):
        """
        Save processed data to files
        """
        import os
        os.makedirs(data_dir, exist_ok=True)
        
        # Rename columns to match descriptive feature names
        X_renamed = X.copy()
        if len(feature_names) == len(X.columns):
            X_renamed.columns = feature_names
        
        # Save processed data
        X_renamed.to_csv(f"{data_dir}/features_standardized.csv")
        y.to_csv(f"{data_dir}/target_returns.csv")
        
        # Create standardization stats with descriptive names
        renamed_stats = {}
        if len(feature_names) == len(X.columns):
            for old_name, new_name in zip(X.columns, feature_names):
                renamed_stats[new_name] = {
                    'mean': standardization_stats['means'][old_name],
                    'std': standardization_stats['stds'][old_name]
                }
        else:
            renamed_stats = standardization_stats
            
        # Save metadata
        metadata = {
            'feature_names': feature_names,
            'standardization_stats': renamed_stats,
            'data_shape': {'features': list(X.shape), 'target': len(y)},
            'date_range': {'start': str(X.index[0].date()), 'end': str(X.index[-1].date())}
        }
        
        import json
        with open(f"{data_dir}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Data saved to {data_dir}/ directory")
    
    def collect_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.Series, List[str], dict]:
        """
        Main method to collect and prepare all data for modeling
        """
        logger.info("="*50)
        logger.info("STARTING HANWHA DATA COLLECTION AND PREPARATION")
        logger.info("="*50)
        
        # Step 1: Download raw data
        raw_data = self.download_raw_data()
        
        # Step 2: Calculate monthly returns  
        monthly_df = self.calculate_monthly_returns(raw_data)
        
        # Step 3: Prepare features and target
        X, y, feature_names = self.prepare_features_and_target(monthly_df)
        
        # Step 4: Standardize features
        X_standardized, standardization_stats = self.standardize_features(X)
        
        # Step 5: Save data to files
        self.save_data_to_files(X_standardized, y, feature_names, standardization_stats)
        
        logger.info("="*50)
        logger.info("DATA COLLECTION COMPLETE")
        logger.info(f"Final dataset: {X_standardized.shape[0]} months, {X_standardized.shape[1]} features")
        logger.info(f"Target variable: 한화솔루션 monthly returns")
        logger.info(f"Features: {feature_names}")
        logger.info("="*50)
        
        return X_standardized, y, feature_names, standardization_stats


# Example usage and testing
if __name__ == "__main__":
    # Initialize data collector
    collector = HanwhaDataCollector()
    
    # Collect and prepare data
    X, y, feature_names, standardization_stats = collector.collect_and_prepare_data()
    
    # Display final results
    print("\n" + "="*60)
    print("FINAL DATA SUMMARY")
    print("="*60)
    print(f"Features (X): {X.shape}")
    print(f"Target (y): {y.shape}")
    print(f"Feature names: {feature_names}")
    print(f"\nFirst 5 rows of features:")
    print(X.head())
    print(f"\nFirst 5 target values:")
    print(y.head())
    print(f"\nTarget statistics:")
    print(f"Mean: {y.mean():.6f}")
    print(f"Std: {y.std():.6f}")
    print(f"Min: {y.min():.6f}")
    print(f"Max: {y.max():.6f}")
