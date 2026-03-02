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
        self.last_known_exog = None  # Store last exog values for forecasting
        self.last_date = None  #  Store last date for calendar features
        self.original_data = None  # Store original data for lagged features
        
        models_dir = os.path.join(settings.BASE_DIR, 'models')
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        self.model_path = models_dir
    
    def prepare_data(self, training_data, create_lags=True):
        """
        Prepare data for ARIMAX model with lagged exogenous variables.
        """
        if isinstance(training_data, list):
            df = pd.DataFrame(training_data)
        else:
            df = pd.DataFrame(list(training_data.values()))
        
        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        #  Convert ALL numeric columns to float64 explicitly
        numeric_columns = ['farmgate_price', 'oil_price_trend', 'peso_dollar_rate']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].astype('float64')
        
        # Remove rows where farmgate_price is missing
        if 'farmgate_price' in df.columns:
            df = df.dropna(subset=['farmgate_price'])
        
        # Fill missing values in original exogenous variables
        for col in ['oil_price_trend', 'peso_dollar_rate']:
            if col in df.columns:
                df[col] = df[col].ffill().bfill()
                if df[col].isna().any():
                    mean_val = df[col].mean()
                    df[col] = df[col].fillna(mean_val if pd.notna(mean_val) else 0.0)
                df[col] = df[col].astype('float64')
        
        # Sort by date BEFORE creating features
        df = df.sort_values('date').reset_index(drop=True)
        
        if create_lags:
            # ===== CREATE LAGGED EXOGENOUS FEATURES =====
            print("[PREPARE] Creating lagged exogenous features...")
            
            # Lagged oil prices (1 day and 3 days ago)
            if 'oil_price_trend' in df.columns:
                df['oil_price_lag1'] = df['oil_price_trend'].shift(1)
                df['oil_price_lag3'] = df['oil_price_trend'].shift(3)
                df['oil_price_ma7'] = df['oil_price_trend'].rolling(window=7, min_periods=1).mean()
            
            # Lagged peso rates (1 day ago)
            if 'peso_dollar_rate' in df.columns:
                df['peso_rate_lag1'] = df['peso_dollar_rate'].shift(1)
                df['peso_rate_ma7'] = df['peso_dollar_rate'].rolling(window=7, min_periods=1).mean()
            
        
            df['month'] = df['date'].dt.month
      
      
            # Ensure all new columns are float64
            lag_columns = [
                'oil_price_lag1', 'oil_price_lag3', 'oil_price_ma7',
                'peso_rate_lag1', 'peso_rate_ma7',
                'month', 
            ]
            
            for col in lag_columns:
                if col in df.columns:
                    df[col] = df[col].astype('float64')
            
            initial_rows = len(df)
            df = df.dropna()
            dropped_rows = initial_rows - len(df)
            if dropped_rows > 0:
                print(f"[PREPARE] Dropped {dropped_rows} rows due to lagging/rolling operations")
        
        # Reset index
        df = df.reset_index(drop=True)
        
        # Final validation: ensure all numeric columns are float64
        for col in df.columns:
            if col != 'date' and df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].astype('float64')
        
        # Set date as index after all cleaning
        df.set_index('date', inplace=True)
        
        print(f"[PREPARE] Final dataset shape: {df.shape}")
        print(f"[PREPARE] Columns: {list(df.columns)}")
        
        return df
    
    def train(self, training_data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, is_deployment=False):
        """
        Train ARIMAX model with lagged exogenous variables.
        - is_deployment = False → use train/val/test split for evaluation
        - is_deployment = True → train on 100% of data for production
        """
        df = self.prepare_data(training_data, create_lags=True)
        
        # Store original data for future forecasting
        self.original_data = df.copy()
        
        if len(df) < 100:
            return {"error": "Insufficient data. Need at least 100 records."}
        
        if 'farmgate_price' not in df.columns:
            return {"error": "farmgate_price column not found in training data."}
        
        # Validate ratios sum to 1.0
        if not is_deployment and abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.01:
            return {"error": f"Ratios must sum to 1.0. Got {train_ratio + val_ratio + test_ratio}"}
        
        # --- Prepare Data ---
        endog = df['farmgate_price'].copy()
        
        # USE LAGGED EXOGENOUS VARIABLES
        available_exog = [
            'oil_price_lag1',       # Yesterday's oil price
            'peso_rate_lag1',       # Yesterday's peso rate
            'oil_price_ma7',        # 7-day oil price moving average
            'peso_rate_ma7',        # 7-day peso rate moving average
            'month',                # Calendar month (1-12)
        ]
        
        valid_exog_columns = [
            col for col in available_exog 
            if col in df.columns and df[col].nunique() > 1
        ]
        
        self.exog_columns = valid_exog_columns
        print(f"[TRAIN] Using exogenous variables: {self.exog_columns}")
        
        exog = None
        if self.exog_columns:
            exog = df[self.exog_columns].copy()
            # Store last known exogenous values for forecasting
            self.last_known_exog = exog.iloc[-10:].copy()  # Last 10 observations
            self.last_date = df.index[-1]  # Store last date
        
        try:
            if is_deployment:
                # ======================================================== 
                # DEPLOYMENT MODE: Train on 100% of Data
                # ========================================================
                full_endog = np.asarray(endog.values, dtype=np.float64)
                full_exog = np.asarray(exog.values, dtype=np.float64) if exog is not None else None
                
                print(f"[DEPLOYMENT] Training on {len(full_endog)} samples")
                print(f"[DEPLOYMENT] ARIMA order: {self.order}")
                print(f"[DEPLOYMENT] Exog columns: {self.exog_columns}")
                
                self.model = ARIMA(full_endog, exog=full_exog, order=self.order)
                self.fitted_model = self.model.fit()
                
                in_sample_preds = self.fitted_model.predict()
                
                mae = mean_absolute_error(full_endog, in_sample_preds)
                rmse = np.sqrt(mean_squared_error(full_endog, in_sample_preds))
                mape = np.mean(np.abs((full_endog - in_sample_preds) / (full_endog + 1e-10))) * 100
                
                print(f"[DEPLOYMENT] In-sample MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
                
                return {
                    'is_deployment': True,
                }
            
            else:
                # ========================================================
                # EVALUATION MODE: Train/Val/Test Split
                # ========================================================
                total_samples = len(endog)
                train_end = int(total_samples * train_ratio)
                val_end = int(total_samples * (train_ratio + val_ratio))
                test_end = total_samples
                
                print(f"[EVALUATION] Total samples: {total_samples}")
                print(f"[EVALUATION] Train: {train_end}, Val: {val_end - train_end}, Test: {test_end - val_end}")
                print(f"[EVALUATION] ARIMA order: {self.order}")
                print(f"[EVALUATION] Exog columns: {self.exog_columns}")
                
                # Split data
                train_endog_array = np.asarray(endog.iloc[:train_end].values, dtype=np.float64)
                val_endog_array = np.asarray(endog.iloc[train_end:val_end].values, dtype=np.float64)
                test_endog_array = np.asarray(endog.iloc[val_end:test_end].values, dtype=np.float64)
                
                train_exog_array = np.asarray(exog.iloc[:train_end].values, dtype=np.float64) if exog is not None else None
                val_exog_array = np.asarray(exog.iloc[train_end:val_end].values, dtype=np.float64) if exog is not None else None
                test_exog_array = np.asarray(exog.iloc[val_end:test_end].values, dtype=np.float64) if exog is not None else None
                
                # Print data statistics
                print(f"[EVALUATION] Train - Mean: {train_endog_array.mean():.2f}, Std: {train_endog_array.std():.2f}")
                print(f"[EVALUATION] Val - Mean: {val_endog_array.mean():.2f}, Std: {val_endog_array.std():.2f}")
                print(f"[EVALUATION] Test - Mean: {test_endog_array.mean():.2f}, Std: {test_endog_array.std():.2f}")
                
                # Train on train split
                self.model = ARIMA(train_endog_array, exog=train_exog_array, order=self.order)
                self.fitted_model = self.model.fit()
                
                print(f"[EVALUATION] Model fitted. AIC: {self.fitted_model.aic:.2f}")
                
                # --- VALIDATION SET EVALUATION ---
                val_predictions = self.fitted_model.forecast(steps=len(val_endog_array), exog=val_exog_array)
                
                val_mae = mean_absolute_error(val_endog_array, val_predictions)
                val_rmse = np.sqrt(mean_squared_error(val_endog_array, val_predictions))
                val_mape = np.mean(np.abs((val_endog_array - val_predictions) / (val_endog_array + 1e-10))) * 100
                
                print(f"[EVALUATION] Validation Metrics - MAE: {val_mae:.2f}, RMSE: {val_rmse:.2f}, MAPE: {val_mape:.2f}%")
                
                # --- TEST SET EVALUATION ---
                # For proper test evaluation, retrain on train+val
                trainval_endog = np.concatenate([train_endog_array, val_endog_array])
                trainval_exog = np.concatenate([train_exog_array, val_exog_array]) if exog is not None else None
                
                final_model = ARIMA(trainval_endog, exog=trainval_exog, order=self.order)
                final_fitted = final_model.fit()
                
                test_predictions = final_fitted.forecast(steps=len(test_endog_array), exog=test_exog_array)
                
                test_mae = mean_absolute_error(test_endog_array, test_predictions)
                test_rmse = np.sqrt(mean_squared_error(test_endog_array, test_predictions))
                test_mape = np.mean(np.abs((test_endog_array - test_predictions) / (test_endog_array + 1e-10))) * 100
                
                print(f"[EVALUATION] Test Metrics - MAE: {test_mae:.2f}, RMSE: {test_rmse:.2f}, MAPE: {test_mape:.2f}%")
                
                # Store the final model (trained on train+val)
                self.fitted_model = final_fitted
                print(self.fitted_model.summary())
                
                return {
                    'aic': float(final_fitted.aic),
                    # Validation metrics
                    'val_mae': val_mae,
                    'val_rmse': val_rmse,
                    'val_mape': val_mape,
                    # Test metrics
                    'mae': test_mae,
                    'rmse': test_rmse,
                    'mape': test_mape,
                    # Plotting data (test set only for cleaner viz)
                    'plot_actual': test_endog_array.tolist(),
                    'plot_preds': test_predictions.tolist(),
                    'is_deployment': False
                }
        
        except Exception as e:
            import traceback
            print("[ERROR] Exception during training:")
            traceback.print_exc()
            return {"error": str(e)}
    
    def save_model(self, model_name):
        """Save trained model to file with all necessary metadata"""
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
                'last_known_exog': self.last_known_exog,  # NEW: For forecasting
                'last_date': self.last_date,  # NEW: For calendar features
                'original_data': self.original_data,  # NEW: For reference
                'timestamp': pd.Timestamp.now()
            }, f)
        
        print(f"[SAVE] Model saved to {file_path}")
        return file_path
    
    def load_model(self, model_path):
        """Load trained model from file with all metadata"""
        with open(model_path, 'rb') as f:
            saved_data = pickle.load(f)
            self.fitted_model = saved_data['model']
            self.order = saved_data.get('order', (saved_data.get('p', 1), saved_data.get('d', 1), saved_data.get('q', 1)))
            self.exog_columns = saved_data.get('exog_columns', [])
            self.last_known_exog = saved_data.get('last_known_exog', None)  # NEW
            self.last_date = saved_data.get('last_date', None)  # NEW
            self.original_data = saved_data.get('original_data', None)  # NEW
        
        print(f"[LOAD] Model loaded from {model_path}")
        print(f"[LOAD] Order: {self.order}, Exog columns: {self.exog_columns}")
        
        return self.fitted_model
    def forecast(self, steps=14, exog_future=None, use_latest_values=False, latest_oil=None, latest_peso=None):
        """
        Returns:
            numpy array of forecasted prices
        """
        if self.fitted_model is None:
            raise ValueError("Model not trained or loaded")
        
        print(f"[FORECAST] Forecasting {steps} steps ahead...")
        
        # Handle exogenous variables
        if self.exog_columns is None or len(self.exog_columns) == 0:
            # No exogenous variables - simple forecast
            forecast_result = self.fitted_model.forecast(steps=steps)
        else:
            # Need exogenous variables
            if exog_future is None:
                if use_latest_values and latest_oil is not None and latest_peso is not None:
                    # DEPLOYMENT MODE: Use latest values from farmer + ICC
                    print(f"[FORECAST] Using latest values: Oil={latest_oil}, Peso={latest_peso}")
                    exog_future = self.create_future_exog_with_latest(steps, latest_oil, latest_peso)
                else:
                    # TRAINING MODE: Auto-generate from historical data
                    print("[FORECAST] Auto-generating future exogenous variables from historical data...")
                    exog_future = self.create_future_exog(steps)
                
                if exog_future is None:
                    raise ValueError(
                        "Could not generate future exogenous variables. "
                        "Please provide latest_oil and latest_peso for deployment forecasting."
                    )
            else:
                # Validate provided exog_future
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
        
        print(f"[FORECAST] Forecast completed. Mean: {forecast_result.mean():.2f}, Std: {forecast_result.std():.2f}")
        
        return forecast_result

    def create_future_exog_with_latest(self, steps, latest_oil, latest_peso):
        """
        Create future exogenous variables using LATEST values from farmer/ICC.
        Used for real-time deployment forecasting.
        
        Args:
            steps: Number of future periods to forecast
            latest_oil: Current oil price from ICC
            latest_peso: Current peso/dollar rate from farmer
            
        Returns:
            numpy array of shape (steps, n_exog_features)
        """
        if not self.exog_columns or self.last_date is None:
            return None
        
        print(f"[FORECAST] Creating future exog with latest values for {steps} steps...")
        
        # Calculate 7-day moving averages using last known data + latest values
        if self.last_known_exog is not None and 'oil_price_trend' in self.original_data.columns:
            # Get last 6 historical values + latest value for MA calculation
            last_6_oil = self.original_data['oil_price_trend'].iloc[-6:].tolist()
            oil_ma7 = np.mean(last_6_oil + [latest_oil])
            
            last_6_peso = self.original_data['peso_dollar_rate'].iloc[-6:].tolist()
            peso_ma7 = np.mean(last_6_peso + [latest_peso])
        else:
            # Fallback to latest values
            oil_ma7 = latest_oil
            peso_ma7 = latest_peso
        
        future_exog = []
        
        for i in range(steps):
            exog_row = []
            
            # Calculate future date
            future_date = self.last_date + pd.Timedelta(days=i+1)
            
            for col in self.exog_columns:
                if col == 'oil_price_lag1':
                    # Use latest oil price (from ICC or from farmer input)
                    exog_row.append(float(latest_oil))
                
                elif col == 'oil_price_lag3':
                    # Use latest oil price (assuming stable over 3 days)
                    exog_row.append(float(latest_oil))
                
                elif col == 'oil_price_ma7':
                    # Use calculated 7-day moving average
                    exog_row.append(float(oil_ma7))
                
                elif col == 'peso_rate_lag1':
                    # Use latest peso rate (from farmer input)
                    exog_row.append(float(latest_peso))
                
                elif col == 'peso_rate_ma7':
                    # Use calculated 7-day moving average
                    exog_row.append(float(peso_ma7))
                
                elif col == 'month':
                    exog_row.append(float(future_date.month))
                
    
                
                else:
                    # For any other column, use last known value
                    exog_row.append(float(self.last_known_exog[col].iloc[-1]))
            
            future_exog.append(exog_row)
        
        future_exog_array = np.array(future_exog, dtype=np.float64)
        print(f"[FORECAST] Future exog shape: {future_exog_array.shape}")
        print(f"[FORECAST] Using latest_oil={latest_oil}, latest_peso={latest_peso}")
        
        return future_exog_array
    