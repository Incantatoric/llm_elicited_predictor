"""
Base evaluator class with common functionality for all prediction methods
"""

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')


class BaseEvaluator(ABC):
    """
    Base class for all prediction method evaluators
    
    Provides common functionality:
    - Data loading and preprocessing
    - Train/test split
    - Basic evaluation metrics
    - Common visualization plots
    - Result saving
    """
    
    def __init__(self, method_name: str):
        self.method_name = method_name
        self.project_root = Path(__file__).parent.parent
        self.results_dir = self.project_root / "data" / "results" / method_name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Data containers
        self.X = None
        self.y = None
        self.feature_names = None
        self.metadata = None
        
        # Train/test split
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Results
        self.results_df = None
        self.metrics = None
        
    def load_data(self):
        """Load processed data and metadata"""
        print(f"\n{'='*60}")
        print(f"LOADING DATA FOR {self.method_name.upper()}")
        print(f"{'='*60}")
        
        # Load features and target
        self.X = pd.read_csv(
            self.project_root / 'data/processed/features_standardized.csv', 
            index_col=0
        )
        self.y = pd.read_csv(
            self.project_root / 'data/processed/target_returns.csv', 
            index_col=0
        ).squeeze()
        
        # Load metadata
        with open(self.project_root / 'data/processed/metadata.json') as f:
            self.metadata = json.load(f)
        self.feature_names = self.metadata['feature_names']
        
        print(f"✓ Loaded {len(self.X)} data points")
        print(f"✓ Features: {self.feature_names}")
        print(f"✓ Date range: {self.X.index[0]} to {self.X.index[-1]}")
        print(f"✓ Target statistics: mean={self.y.mean():.4f}, std={self.y.std():.4f}")
        
    def create_train_test_split(self, test_months: int = 6):
        """Create train/test split"""
        split_idx = len(self.X) - test_months
        self.X_train = self.X.iloc[:split_idx]
        self.X_test = self.X.iloc[split_idx:]
        self.y_train = self.y.iloc[:split_idx]
        self.y_test = self.y.iloc[split_idx:]
        
        print(f"\n📊 TRAIN/TEST SPLIT:")
        print(f"✓ Training: {len(self.X_train)} points ({self.X_train.index[0]} to {self.X_train.index[-1]})")
        print(f"✓ Testing: {len(self.X_test)} points ({self.X_test.index[0]} to {self.X_test.index[-1]})")
        
    @abstractmethod
    def train_and_predict(self):
        """Train model and generate predictions - must be implemented by subclasses"""
        pass
    
    def calculate_metrics(self):
        """Calculate standard evaluation metrics"""
        if self.results_df is None:
            raise ValueError("No results available. Run train_and_predict() first.")
            
        print(f"\n📈 EVALUATION METRICS")
        print(f"{'-'*40}")
        
        # Basic metrics
        mae = np.mean(np.abs(self.results_df['Prediction_Error']))
        mse = np.mean(self.results_df['Prediction_Error']**2)
        rmse = np.sqrt(mse)
        
        # R-squared
        ss_res = np.sum(self.results_df['Prediction_Error']**2)
        ss_tot = np.sum((self.y_test - np.mean(self.y_test))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Directional accuracy (did we predict the right direction?)
        actual_direction = np.sign(self.results_df['Actual_Return'])
        predicted_direction = np.sign(self.results_df['Predicted_Mean'])
        directional_accuracy = np.mean(actual_direction == predicted_direction)
        
        self.metrics = {
            'mae': float(mae),
            'rmse': float(rmse),
            'r_squared': float(r_squared),
            'directional_accuracy': float(directional_accuracy)
        }
        
        # Add coverage if confidence intervals available
        if 'Lower_5%' in self.results_df.columns and 'Upper_95%' in self.results_df.columns:
            in_interval = ((self.results_df['Actual_Return'] >= self.results_df['Lower_5%']) & 
                          (self.results_df['Actual_Return'] <= self.results_df['Upper_95%']))
            coverage = in_interval.mean()
            self.metrics['coverage_95'] = float(coverage)
            print(f"✓ 95% Prediction Interval Coverage: {coverage:.2%}")
        
        print(f"✓ Mean Absolute Error (MAE): {mae:.4f}")
        print(f"✓ Root Mean Square Error (RMSE): {rmse:.4f}")
        print(f"✓ R-squared: {r_squared:.4f}")
        print(f"✓ Directional Accuracy: {directional_accuracy:.2%}")
        
    def create_basic_plots(self):
        """Create basic plots common to all methods"""
        print(f"\n🎨 CREATING BASIC PLOTS")
        print(f"{'-'*40}")
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Actual vs Predicted scatter plot
        axes[0].scatter(self.results_df['Actual_Return'], 
                       self.results_df['Predicted_Mean'], 
                       alpha=0.7, s=60)
        
        # Perfect prediction line
        min_val = min(self.results_df['Actual_Return'].min(), 
                     self.results_df['Predicted_Mean'].min())
        max_val = max(self.results_df['Actual_Return'].max(), 
                     self.results_df['Predicted_Mean'].max())
        axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
        
        axes[0].set_xlabel('Actual Return')
        axes[0].set_ylabel('Predicted Return')
        axes[0].set_title('Actual vs Predicted Returns')
        axes[0].grid(True, alpha=0.3)
        
        # Time series plot
        x_pos = range(len(self.results_df))
        axes[1].plot(x_pos, self.results_df['Actual_Return'], 
                    'o-', label='Actual', linewidth=2, markersize=6)
        axes[1].plot(x_pos, self.results_df['Predicted_Mean'], 
                    's-', label='Predicted', linewidth=2, markersize=6)
        
        axes[1].set_xlabel('Month')
        axes[1].set_ylabel('Return')
        axes[1].set_title('Time Series: Actual vs Predicted')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Set month labels
        month_labels = [date.split('-')[1] + '/' + date.split('-')[0][2:] 
                       for date in self.results_df['Date']]
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(month_labels, rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'basic_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✓ Basic plots saved to {self.results_dir}/basic_plots.png")
        
    def save_results(self):
        """Save evaluation results"""
        print(f"\n💾 SAVING RESULTS")
        print(f"{'-'*40}")
        
        # Save predictions
        self.results_df.to_csv(self.results_dir / "predictions.csv")
        print(f"✓ Predictions saved to predictions.csv")
        
        # Save metrics
        summary = {
            'method': self.method_name,
            'timestamp': datetime.now().isoformat(),
            'data_summary': {
                'total_points': len(self.X),
                'training_points': len(self.X_train),
                'test_points': len(self.X_test),
                'features': self.feature_names
            },
            'metrics': self.metrics
        }
        
        with open(self.results_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Summary saved to summary.json")
        
    def run_evaluation(self):
        """Run complete evaluation pipeline"""
        print(f"\n🚀 STARTING {self.method_name.upper()} EVALUATION")
        print(f"{'='*80}")
        
        # Load data
        self.load_data()
        
        # Create train/test split
        self.create_train_test_split()
        
        # Train and predict (method-specific)
        self.train_and_predict()
        
        # Calculate metrics
        self.calculate_metrics()
        
        # Create basic plots
        self.create_basic_plots()
        
        # Save results
        self.save_results()
        
        print(f"\n✅ {self.method_name.upper()} EVALUATION COMPLETE")
        print(f"📁 Results saved to: {self.results_dir}")
        print(f"{'='*80}")
        
        return self.metrics 