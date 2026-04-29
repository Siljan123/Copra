
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
        numeric_columns = ['farmgate_price', 'oil_price_trend', 'peso_dollar_rate', 'diesel_price', 'labor_min_wage']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].astype('float64')
        
        # Remove rows where farmgate_price is missing, as it's the target variable
        if 'farmgate_price' in df.columns:
            df = df.dropna(subset=['farmgate_price'])
        
        # Sort by date BEFORE creating features
        df = df.sort_values('date').reset_index(drop=True)
        
        if create_lags:
            # ===== CREATE LAGGED EXOGENOUS FEATURES =====
            print("[PREPARE] Creating lagged exogenous features...")
            
            # Lagged oil prices (1, 7 and 30 days ago)
            if 'oil_price_trend' in df.columns:
                df['oil_price_lag1']  = df['oil_price_trend'].shift(1)          # previous record
                df['oil_price_ma7']   = df['oil_price_trend'].rolling(7, min_periods=1).mean()   # last 7 records
                df['oil_price_ma30']  = df['oil_price_trend'].rolling(30, min_periods=1).mean() # last 30 records
                
            # Lagged peso rates (1 and 7 days ago)
            if 'peso_dollar_rate' in df.columns:
                df['peso_dollar_rate_lag1'] = df['peso_dollar_rate'].shift(1)   # previous record
                df['peso_rate_ma7']   = df['peso_dollar_rate'].rolling(7, min_periods=1).mean()  # last 7 records
            
            # Lagged diesel prices (1, 7 and 30 days ago)
            if 'diesel_price' in df.columns:
                df['diesel_price_lag1']  = df['diesel_price'].shift(1)          # previous record
                df['diesel_price_ma7']   = df['diesel_price'].rolling(7, min_periods=1).mean()   # last 7 records
                df['diesel_price_ma30']  = df['diesel_price'].rolling(30, min_periods=1).mean() # last 30 records
            
            # Lagged labor min wage (1, 7 and 30 days ago)
            if 'labor_min_wage' in df.columns:
                df['labor_min_wage_lag1']  = df['labor_min_wage'].shift(1)          # previous record
                df['labor_min_wage_ma7']   = df['labor_min_wage'].rolling(7, min_periods=1).mean()   # last 7 records
                df['labor_min_wage_ma30']  = df['labor_min_wage'].rolling(30, min_periods=1).mean() # last 30 records
            
            # Ensure all new columns are float64
            lag_columns = [
                'oil_price_lag1',
                'oil_price_ma7',
                'oil_price_ma30',
                'peso_dollar_rate_lag1',
                'peso_rate_ma7',
                'diesel_price_lag1',
                'diesel_price_ma7',
                'diesel_price_ma30',
                'labor_min_wage_lag1',
                'labor_min_wage_ma7',
                'labor_min_wage_ma30',
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
        
        # EVALUATION: include all variables to assess statistical significance
        evaluation_exog = [
            'oil_price_trend',    
            'peso_dollar_rate',   
            'oil_price_lag1',        
            'peso_dollar_rate_lag1', 
            'oil_price_ma7',         
            'oil_price_ma30',      
            'peso_rate_ma7',
            'diesel_price',
            'diesel_price_lag1',
            'diesel_price_ma7',
            'diesel_price_ma30',
            'labor_min_wage',
            'labor_min_wage_lag1',
            'labor_min_wage_ma7',
            'labor_min_wage_ma30',
        ]
        # DEPLOYMENT: only statistically significant variables (p < 0.05)
        deployment_exog = [
            'oil_price_trend',    
            'peso_dollar_rate',   
            'oil_price_ma7',       
            'oil_price_ma30',
            'diesel_price',
            'labor_min_wage',        
        ]

        # Choose variables based on mode
        exog_to_use = deployment_exog if is_deployment else evaluation_exog
        
        valid_exog_columns = [
            col for col in exog_to_use
            if col in df.columns and df[col].nunique() > 1
        ]
        
        self.exog_columns = valid_exog_columns
        print(f"[TRAIN] Mode: {'DEPLOYMENT' if is_deployment else 'EVALUATION'}")
        print(f"[TRAIN] Using exogenous variables: {self.exog_columns}")
        
        exog = None
        if self.exog_columns:
            exog = df[self.exog_columns].copy()
            self.last_known_exog = exog.iloc[-30:].copy()
            self.last_date = df.index[-1]
        
        try:
            if is_deployment:
                # ========================================================
                # DEPLOYMENT MODE: Train on 100% of Data
                # ========================================================
                # DEPLOYMENT MODE: Visualizing Test Split + Training 100%
                # ========================================================
                total_samples = len(endog)
                # Define the split point for the plot (e.g., last 15%)
                split_idx = int(total_samples * (1 - test_ratio))
                
                # --- PART A: Generate the Evaluation Plots ---
                # We train a temporary model on the past to "predict" the present
                temp_train_endog = endog.iloc[:split_idx]
                temp_test_actual = endog.iloc[split_idx:]
                
                temp_train_exog = exog.iloc[:split_idx] if exog is not None else None
                temp_test_exog = exog.iloc[split_idx:] if exog is not None else None

                temp_model = ARIMA(temp_train_endog, exog=temp_train_exog, order=self.order)
                temp_fitted = temp_model.fit()
                
                # These are the "Testing Split" predictions for your graph
                test_predictions = temp_fitted.forecast(steps=len(temp_test_actual), exog=temp_test_exog)

                # --- PART B: Train the Final Production Model (100% Data) ---
                full_endog = np.asarray(endog.values, dtype=np.float64)
                full_exog = np.asarray(exog.values, dtype=np.float64) if exog is not None else None
                
                self.model = ARIMA(full_endog, exog=full_exog, order=self.order)
                self.fitted_model = self.model.fit()
                
                print(f"[DEPLOYMENT] Production model trained on 100% ({len(full_endog)} samples).")

                # Return the split data for the graph, but the model itself is 100% trained
                return {
                    'is_deployment': True,
                    'aic': float(self.fitted_model.aic),
                    'plot_actual': temp_test_actual.values.tolist(),
                    'plot_preds': test_predictions.tolist(),
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
                val_endog_array   = np.asarray(endog.iloc[train_end:val_end].values, dtype=np.float64)
                test_endog_array  = np.asarray(endog.iloc[val_end:test_end].values, dtype=np.float64)
                
                train_exog_array = np.asarray(exog.iloc[:train_end].values, dtype=np.float64) if exog is not None else None
                val_exog_array   = np.asarray(exog.iloc[train_end:val_end].values, dtype=np.float64) if exog is not None else None
                test_exog_array  = np.asarray(exog.iloc[val_end:test_end].values, dtype=np.float64) if exog is not None else None
                
                # Print data statistics
                print(f"[EVALUATION] Train - Mean: {train_endog_array.mean():.2f}, Std: {train_endog_array.std():.2f}")
                print(f"[EVALUATION] Val   - Mean: {val_endog_array.mean():.2f}, Std: {val_endog_array.std():.2f}")
                print(f"[EVALUATION] Test  - Mean: {test_endog_array.mean():.2f}, Std: {test_endog_array.std():.2f}")
                
                # Train on train split
                self.model = ARIMA(train_endog_array, exog=train_exog_array, order=self.order)
                self.fitted_model = self.model.fit()
                
                print(f"[EVALUATION] Model fitted. AIC: {self.fitted_model.aic:.2f}")
                
                # --- VALIDATION SET EVALUATION ---
                val_predictions = self.fitted_model.forecast(steps=len(val_endog_array), exog=val_exog_array)
                
                val_mae  = mean_absolute_error(val_endog_array, val_predictions)
                val_rmse = np.sqrt(mean_squared_error(val_endog_array, val_predictions))
                val_mape = np.mean(np.abs((val_endog_array - val_predictions) / (val_endog_array + 1e-10))) * 100
                
                print(f"[EVALUATION] Validation - MAE: {val_mae:.2f}, RMSE: {val_rmse:.2f}, MAPE: {val_mape:.2f}%")
                
                # --- TEST SET EVALUATION ---
                # Retrain on train+val for unbiased test evaluation
                trainval_endog = np.concatenate([train_endog_array, val_endog_array])
                trainval_exog  = np.concatenate([train_exog_array, val_exog_array]) if exog is not None else None
                
                final_model  = ARIMA(trainval_endog, exog=trainval_exog, order=self.order)
                final_fitted = final_model.fit()
                
                test_predictions = final_fitted.forecast(steps=len(test_endog_array), exog=test_exog_array)
                
                test_mae  = mean_absolute_error(test_endog_array, test_predictions)
                test_rmse = np.sqrt(mean_squared_error(test_endog_array, test_predictions))
                test_mape = np.mean(np.abs((test_endog_array - test_predictions) / (test_endog_array + 1e-10))) * 100
                
                print(f"[EVALUATION] Test - MAE: {test_mae:.2f}, RMSE: {test_rmse:.2f}, MAPE: {test_mape:.2f}%")
                
                # Store final model (trained on train+val)
                self.fitted_model = final_fitted
                print(self.fitted_model.summary())
                
                return {
                    'aic': float(final_fitted.aic),
                    # Validation metrics
                    'val_mae':  val_mae,
                    'val_rmse': val_rmse,
                    'val_mape': val_mape,
                    # Test metrics
                    'mae':  test_mae,
                    'rmse': test_rmse,
                    'mape': test_mape,
                    # Plotting data
                    'plot_actual': test_endog_array.tolist(),
                    'plot_preds':  test_predictions.tolist(),
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
                'exog_columns': self.exog_columns,
                'last_known_exog': self.last_known_exog,  
                'last_date': self.last_date,  
                'original_data': self.original_data, 
                'timestamp': pd.Timestamp.now()
            }, f)
        
        print(f"[SAVE] Model saved to {file_path}")
        return file_path
    
    def load_model(self, model_path):
        """Load trained model from file with all metadata"""
        with open(model_path, 'rb') as f:
            saved_data = pickle.load(f)
            self.fitted_model = saved_data['model']
            self.order = saved_data.get('order', ...)
            self.exog_columns = saved_data.get('exog_columns', [])
            self.last_known_exog = saved_data.get('last_known_exog', None)  # NEW
            self.last_date = saved_data.get('last_date', None)  # NEW
            self.original_data = saved_data.get('original_data', None)  # NEW
        
        print(f"[LOAD] Model loaded from {model_path}")
        print(f"[LOAD] Order: {self.order}, Exog columns: {self.exog_columns}")
        
        return self.fitted_model
    def forecast(self, steps=14, exog_future=None, use_latest_values=False, latest_oil=None, latest_peso=None, latest_diesel=None, latest_labor=None):
        """Make future predictions"""
        if self.fitted_model is None:
            raise ValueError("Model not trained or loaded")

        if self.exog_columns is None or len(self.exog_columns) == 0:
            # No exogenous variables - simple forecast
            forecast_result = self.fitted_model.forecast(steps=steps)
        else:
            # Build future exog if not provided
            if exog_future is None and use_latest_values:
                exog_future = self.create_future_exog_with_latest(
                    steps=steps,
                    latest_oil=latest_oil,
                    latest_peso=latest_peso,
                    latest_diesel=latest_diesel,
                    latest_labor=latest_labor,
                )

            if exog_future is None:
                raise ValueError(
                    f"This model requires exogenous variables: {self.exog_columns}"
                )

            forecast_result = self.fitted_model.forecast(steps=steps, exog=exog_future)

        return forecast_result


    def create_future_exog_with_latest(self, steps, latest_oil, latest_peso, latest_diesel=None, latest_labor=None):
        """Build future exogenous variables with rolling moving averages"""
        if not self.exog_columns or self.last_date is None:
            return None

        if self.original_data is not None and 'oil_price_trend' in self.original_data.columns:
            # Initialize rolling windows from historical data
            rolling_oil_7  = self.original_data['oil_price_trend'].iloc[-6:].tolist()   # last 6
            rolling_oil_30 = self.original_data['oil_price_trend'].iloc[-29:].tolist()  # last 29
            rolling_peso_7 = self.original_data['peso_dollar_rate'].iloc[-6:].tolist()  # last 6
            rolling_diesel_7  = self.original_data['diesel_price'].iloc[-6:].tolist() if 'diesel_price' in self.original_data.columns else [latest_diesel or 0.0] * 6
            rolling_diesel_30 = self.original_data['diesel_price'].iloc[-29:].tolist() if 'diesel_price' in self.original_data.columns else [latest_diesel or 0.0] * 29
            rolling_labor_7  = self.original_data['labor_min_wage'].iloc[-6:].tolist() if 'labor_min_wage' in self.original_data.columns else [latest_labor or 0.0] * 6
            rolling_labor_30 = self.original_data['labor_min_wage'].iloc[-29:].tolist() if 'labor_min_wage' in self.original_data.columns else [latest_labor or 0.0] * 29
        else:
            # Fallback if no historical data
            rolling_oil_7  = [latest_oil]  * 6
            rolling_oil_30 = [latest_oil]  * 29
            rolling_peso_7 = [latest_peso] * 6
            rolling_diesel_7  = [latest_diesel or 0.0] * 6
            rolling_diesel_30 = [latest_diesel or 0.0] * 29
            rolling_labor_7  = [latest_labor or 0.0] * 6
            rolling_labor_30 = [latest_labor or 0.0] * 29

        future_exog = []

        for i in range(steps):
            exog_row = []

            # ✅ Add latest to window, remove oldest (rolling update per step)
            rolling_oil_7.append(latest_oil)
            rolling_oil_7.pop(0)

            rolling_oil_30.append(latest_oil)
            rolling_oil_30.pop(0)

            rolling_peso_7.append(latest_peso)
            rolling_peso_7.pop(0)

            rolling_diesel_7.append(latest_diesel or 0.0)
            rolling_diesel_7.pop(0)

            rolling_diesel_30.append(latest_diesel or 0.0)
            rolling_diesel_30.pop(0)

            rolling_labor_7.append(latest_labor or 0.0)
            rolling_labor_7.pop(0)

            rolling_labor_30.append(latest_labor or 0.0)
            rolling_labor_30.pop(0)

            # ✅ Recalculate MAs for this step
            oil_ma7  = np.mean(rolling_oil_7)
            oil_ma30 = np.mean(rolling_oil_30)
            peso_ma7 = np.mean(rolling_peso_7)
            diesel_ma7 = np.mean(rolling_diesel_7)
            diesel_ma30 = np.mean(rolling_diesel_30)
            labor_ma7 = np.mean(rolling_labor_7)
            labor_ma30 = np.mean(rolling_labor_30)

            for col in self.exog_columns:
                if col == 'oil_price_trend':
                    exog_row.append(float(latest_oil))
                elif col == 'peso_dollar_rate':
                    exog_row.append(float(latest_peso))
                elif col == 'oil_price_ma7':
                    exog_row.append(float(oil_ma7))
                elif col == 'oil_price_ma30':
                    exog_row.append(float(oil_ma30))
                elif col == 'peso_rate_ma7':
                    exog_row.append(float(peso_ma7))
                elif col == 'diesel_price':
                    exog_row.append(float(latest_diesel if latest_diesel is not None else (self.last_known_exog[col].iloc[-1] if self.last_known_exog is not None and col in self.last_known_exog.columns else 0.0)))
                elif col == 'diesel_price_ma7':
                    exog_row.append(float(diesel_ma7))
                elif col == 'diesel_price_ma30':
                    exog_row.append(float(diesel_ma30))
                elif col == 'labor_min_wage':
                    exog_row.append(float(latest_labor if latest_labor is not None else (self.last_known_exog[col].iloc[-1] if self.last_known_exog is not None and col in self.last_known_exog.columns else 0.0)))
                elif col == 'labor_min_wage_ma7':
                    exog_row.append(float(labor_ma7))
                elif col == 'labor_min_wage_ma30':
                    exog_row.append(float(labor_ma30))
                else:
                    exog_row.append(float(self.last_known_exog[col].iloc[-1]) if self.last_known_exog is not None and col in self.last_known_exog.columns else 0.0)

            future_exog.append(exog_row)

        return np.array(future_exog, dtype=np.float64)