import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import os
import warnings
warnings.filterwarnings('ignore')
from django.conf import settings

class ARIMAXModel:
    def __init__(self, order=(1,1,1)):
        self.order = order
        self.model = None
        self.fitted_model = None
        self.exog_columns = None
        models_dir = os.path.join(settings.BASE_DIR, 'models')
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        self.model_path = models_dir
        
    def prepare_data(self, training_data):
        """Prepare data for ARIMAX model - simple time series format"""
        if isinstance(training_data, list):
            df = pd.DataFrame(training_data)
        else:
            df = pd.DataFrame(list(training_data.values()))
        
        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # CRITICAL: Convert ALL numeric columns to float64 explicitly
        numeric_columns = ['farmgate_price', 'oil_price_trend', 'peso_dollar_rate']
        for col in numeric_columns:
            if col in df.columns:
                # Convert to numeric, coercing errors to NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Explicitly cast to float64
                df[col] = df[col].astype('float64')
        
        # Remove rows where farmgate_price is missing (can't train without target)
        if 'farmgate_price' in df.columns:
            df = df.dropna(subset=['farmgate_price'])
        
        # Fill missing values in exogenous variables
        for col in ['oil_price_trend', 'peso_dollar_rate']:
            if col in df.columns:
                df[col] = df[col].ffill().bfill()
                if df[col].isna().any():
                    mean_val = df[col].mean()
                    df[col] = df[col].fillna(mean_val if pd.notna(mean_val) else 0.0)
                # Ensure still float64 after filling
                df[col] = df[col].astype('float64')
        
        # Sort by date BEFORE setting index
        df = df.sort_values('date')
        
        # Remove any remaining rows with NaN in ANY column
        df = df.dropna()
        
        # Reset index to ensure clean integer index
        df = df.reset_index(drop=True)
        
        # Final validation: ensure all numeric columns are float64
        for col in numeric_columns:
            if col in df.columns:
                assert df[col].dtype == 'float64', f"Column {col} is not float64: {df[col].dtype}"
        
        # NOW set date as index after all cleaning
        df.set_index('date', inplace=True)
        
        return df
    
    def train(self, training_data, train_ratio=0.7, test_ratio=0.2, val_ratio=0.1, test_size=None):
        """
        Train ARIMAX model with 70/20/10 train/test/validation split
        Args:
            training_data: Historical data for training
            train_ratio: Proportion for training (default: 0.7 = 70%)
            test_ratio: Proportion for testing (default: 0.2 = 20%)
            val_ratio: Proportion for validation (default: 0.1 = 10%)
            test_size: DEPRECATED - kept for backward compatibility
        
        Returns:
            Dictionary containing metrics for all three sets
        """
        # Backward compatibility: if test_size is provided, use old 80/20 split
        if test_size is not None:
            print("⚠️ WARNING: test_size parameter is deprecated. Using 70/20/10 split instead.")
            # Ignore test_size and use the new split ratios
        df = self.prepare_data(training_data)
        
        if len(df) < 30:
            return {"error": "Insufficient data. Need at least 30 records for 70/20/10 split."}
        
        # Check if farmgate_price exists in the data
        if 'farmgate_price' not in df.columns:
            return {"error": "farmgate_price column not found in training data."}
        
        # Endogenous variable (target to predict)
        endog = df['farmgate_price'].copy()
        # Ensure it's float64
        endog = endog.astype('float64')
        
        if len(endog) == 0:
            return {"error": "No valid farmgate_price data after cleaning."}
        
        # Verify data type
        print(f"Endog dtype: {endog.dtype}, shape: {endog.shape}")
        print(f"Total dataset size: {len(endog)} days")
        
        # Exogenous variables - ONLY use oil_price_trend and peso_dollar_rate
        available_exog = ['oil_price_trend', 'peso_dollar_rate']
        valid_exog_columns = []
        
        for col in available_exog:
            if col in df.columns and df[col].nunique() > 1 and not df[col].isnull().all():
                valid_exog_columns.append(col)
        
        self.exog_columns = valid_exog_columns
        print(f"Using exogenous columns: {self.exog_columns}")
        
        if self.exog_columns:
            exog = df[self.exog_columns].copy()
            # Ensure it's a DataFrame
            if not isinstance(exog, pd.DataFrame):
                exog = pd.DataFrame(exog)
            # Ensure all columns are float64
            for col in exog.columns:
                exog[col] = exog[col].astype('float64')
            # CRITICAL: Reset index to match endog
            exog = exog.reset_index(drop=True)
            print(f"Exog dtypes: {exog.dtypes.to_dict()}, shape: {exog.shape}")
        else:
            exog = None
        
        # Also reset endog index to integer
        endog = endog.reset_index(drop=True)
        
        try:
            # Calculate split points for 70/20/10 split
            total_samples = len(endog)
            train_end = int(total_samples * train_ratio)
            test_end = int(total_samples * (train_ratio + test_ratio))
            
            # Split data into train/test/validation
            train_endog = endog.iloc[:train_end].astype('float64')
            test_endog = endog.iloc[train_end:test_end].astype('float64')
            val_endog = endog.iloc[test_end:].astype('float64')
            
            if exog is not None:
                train_exog = exog.iloc[:train_end].astype('float64')
                test_exog = exog.iloc[train_end:test_end].astype('float64')
                val_exog = exog.iloc[test_end:].astype('float64')
            else:
                train_exog = None
                test_exog = None
                val_exog = None
            
            # Print split information
            print(f"\n📊 Dataset Split (70/20/10):")
            print(f"   Training:   {len(train_endog)} days ({len(train_endog)/total_samples*100:.1f}%)")
            print(f"   Testing:    {len(test_endog)} days ({len(test_endog)/total_samples*100:.1f}%)")
            print(f"   Validation: {len(val_endog)} days ({len(val_endog)/total_samples*100:.1f}%)")
            print(f"   Total:      {total_samples} days\n")
            
            # Validate data before training
            print(f"Train endog dtype: {train_endog.dtype}, shape: {train_endog.shape}")
            if train_exog is not None:
                print(f"Train exog dtypes: {train_exog.dtypes.to_dict()}, shape: {train_exog.shape}")
                # Verify no object types
                assert all(train_exog.dtypes == 'float64'), f"Not all exog columns are float64: {train_exog.dtypes}"
            
            # Convert to numpy arrays explicitly to verify
            train_endog_array = np.asarray(train_endog, dtype=np.float64)
            train_exog_array = np.asarray(train_exog, dtype=np.float64) if train_exog is not None else None
            
            print(f"Numpy endog dtype: {train_endog_array.dtype}, shape: {train_endog_array.shape}")
            if train_exog_array is not None:
                print(f"Numpy exog dtype: {train_exog_array.dtype}, shape: {train_exog_array.shape}")
            
            # Train ARIMA model using the validated arrays
            print(f"\n🚀 Training ARIMA{self.order} with {len(train_endog)} samples")
            print(f"   Exogenous vars: {self.exog_columns}")
            
            self.model = ARIMA(train_endog_array, exog=train_exog_array, order=self.order)
            self.fitted_model = self.model.fit()
            
            # ===== TESTING SET EVALUATION =====
            print(f"\n📈 Evaluating on Testing Set ({len(test_endog)} samples)...")
            test_exog_array = np.asarray(test_exog, dtype=np.float64) if test_exog is not None else None
            
            if test_exog_array is not None:
                test_predictions = self.fitted_model.forecast(steps=len(test_endog), exog=test_exog_array)
            else:
                test_predictions = self.fitted_model.forecast(steps=len(test_endog))
            
            # Calculate testing metrics
            test_mae = mean_absolute_error(test_endog, test_predictions)
            test_rmse = np.sqrt(mean_squared_error(test_endog, test_predictions))
            test_mape = np.mean(np.abs((test_endog - test_predictions) / test_endog)) * 100
            test_accuracy = max(0, 100 - test_mape)
            
            # ===== VALIDATION SET EVALUATION =====
            print(f"📊 Evaluating on Validation Set ({len(val_endog)} samples)...")
            val_exog_array = np.asarray(val_exog, dtype=np.float64) if val_exog is not None else None
            
            # For validation, we need to forecast from the end of test set
            combined_exog = None
            if test_exog_array is not None and val_exog_array is not None:
                combined_exog = np.vstack([test_exog_array, val_exog_array])
                # Forecast through test + validation period, then take validation portion
                full_predictions = self.fitted_model.forecast(
                    steps=len(test_endog) + len(val_endog), 
                    exog=combined_exog
                )
                val_predictions = full_predictions[len(test_endog):]
            else:
                full_predictions = self.fitted_model.forecast(steps=len(test_endog) + len(val_endog))
                val_predictions = full_predictions[len(test_endog):]
            
            # Calculate validation metrics
            val_mae = mean_absolute_error(val_endog, val_predictions)
            val_rmse = np.sqrt(mean_squared_error(val_endog, val_predictions))
            val_mape = np.mean(np.abs((val_endog - val_predictions) / val_endog)) * 100
            val_accuracy = max(0, 100 - val_mape)
            
            # ===== OVERALL METRICS =====
            # Combine test and validation for overall accuracy
            combined_actual = np.concatenate([test_endog, val_endog])
            combined_predictions = np.concatenate([test_predictions, val_predictions])
            
            overall_mae = mean_absolute_error(combined_actual, combined_predictions)
            overall_rmse = np.sqrt(mean_squared_error(combined_actual, combined_predictions))
            overall_mape = np.mean(np.abs((combined_actual - combined_predictions) / combined_actual)) * 100
            overall_accuracy = max(0, 100 - overall_mape)
            
            metrics = {
                # Training info
                'train_samples': len(train_endog),
                'aic': float(self.fitted_model.aic),
                'bic': float(self.fitted_model.bic),
                
                # Testing set metrics (20%)
                'test_samples': len(test_endog),
                'test_mae': float(test_mae),
                'test_rmse': float(test_rmse),
                'test_mape': float(test_mape),
                'test_accuracy': float(test_accuracy),
                
                # Validation set metrics (10%)
                'val_samples': len(val_endog),
                'val_mae': float(val_mae),
                'val_rmse': float(val_rmse),
                'val_mape': float(val_mape),
                'val_accuracy': float(val_accuracy),
                
                # Overall metrics (test + validation = 30%)
                'mae': float(overall_mae),
                'rmse': float(overall_rmse),
                'mape': float(overall_mape),
                'accuracy': float(overall_accuracy),  # This is the main accuracy displayed
                
                'exog_columns': self.exog_columns,
                'total_samples': total_samples,
                
                'plot_actual': combined_actual.tolist(),     # The real historical prices
                'plot_preds': combined_predictions.tolist(), # The model's guesses
                'val_accuracy': float(val_accuracy),
                'accuracy': float(overall_accuracy),
                
                'exog_columns': self.exog_columns,
                'total_samples': total_samples
            }
            
            print(f"\n✅ Training Complete!")
            print(f"   Testing Accuracy:    {test_accuracy:.2f}%")
            print(f"   Validation Accuracy: {val_accuracy:.2f}%")
            print(f"   Overall Accuracy:    {overall_accuracy:.2f}%")
            
            return metrics
            
        except Exception as e:
            print(f"❌ Error training model: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
    
    def save_model(self, model_name):
        """Save trained model to file"""
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        
        file_path = os.path.join(self.model_path, f"{model_name}.pkl")
        with open(file_path, 'wb') as f:
            pickle.dump({
                'model': self.fitted_model,
                'order': self.order,
                'p': self.order[0],
                'd': self.order[1],
                'q': self.order[2],
                'exog_columns': self.exog_columns,
                'timestamp': pd.Timestamp.now()
            }, f)
        
        return file_path
    
    def load_model(self, model_path):
        """Load trained model from file"""
        with open(model_path, 'rb') as f:
            saved_data = pickle.load(f)
            self.fitted_model = saved_data['model']
            self.order = saved_data.get('order', (saved_data.get('p', 1), saved_data.get('d', 1), saved_data.get('q', 1)))
            self.exog_columns = saved_data['exog_columns']
        return self.fitted_model
    
    def forecast(self, steps=14, exog_future=None):
        """Make future predictions"""
        if self.fitted_model is None:
            raise ValueError("Model not trained or loaded")
        
        if self.exog_columns is None or len(self.exog_columns) == 0:
            forecast_result = self.fitted_model.forecast(steps=steps)
        else:
            if exog_future is None:
                raise ValueError(f"This model requires exogenous variables: {self.exog_columns}")
            
            # Validate exog_future shape
            expected_cols = len(self.exog_columns)
            if isinstance(exog_future, np.ndarray):
                actual_cols = exog_future.shape[1] if len(exog_future.shape) > 1 else 1
                if actual_cols != expected_cols:
                    raise ValueError(
                        f"Exogenous variables mismatch. "
                        f"Model expects {expected_cols} columns {self.exog_columns}, "
                        f"but got {actual_cols} columns."
                    )
            
            forecast_result = self.fitted_model.forecast(steps=steps, exog=exog_future)
        
        return forecast_result