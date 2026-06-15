import gc
import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Attempt to import FreqTrade base classes. 
# Provide robust fallback for testing/inspection outside FreqTrade environments.
try:
    from freqtrade.freqai.base_models.BaseRegressor import BaseRegressor
    from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
except ImportError:
    logger.warning("FreqTrade modules not found. Using mock BaseRegressor for standalone compatibility.")
    class BaseRegressor:
        def __init__(self, **kwargs):
            self.model = None
            self.freqai_info = {}
        def get_init_model(self, feature_parameters: Dict) -> Dict:
            return {}
    class FreqaiDataKitchen:
        pass


class LightGBMRegressorCPU(BaseRegressor):
    """
    Production machine learning predictive model for FreqAI.
    Strictly optimized for Windows / WSL2 CPU execution with a 16GB RAM constraint.
    
    This regressor maps technical indicators and volume regimes to continuous
    expected price return targets over a rolling multi-day window.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.feature_importances_: Optional[Dict[str, float]] = None

    def fit(self, data_dictionary: Dict[str, Any], dk: FreqaiDataKitchen, **kwargs) -> Any:
        """
        Trains the LightGBM Regressor model on rolling historical splits.
        Enforces explicit memory and multithreading controls for 16GB RAM systems.
        """
        logger.info("Initiating LightGBM CPU training cycle...")
        
        # Explicit garbage collection before heavy matrix allocations
        gc.collect()

        X_train: pd.DataFrame = data_dictionary["train_features"]
        y_train: pd.Series = data_dictionary["train_labels"]
        train_weights: Optional[pd.Series] = data_dictionary.get("train_weights")

        # Determine validation split if configured
        eval_set = None
        eval_weights = None
        test_size = self.freqai_info.get("data_split_parameters", {}).get("test_size", 0.2)
        
        if test_size > 0 and "test_features" in data_dictionary and "test_labels" in data_dictionary:
            eval_set = [(data_dictionary["test_features"], data_dictionary["test_labels"])]
            eval_weights = [data_dictionary["test_weights"]] if data_dictionary.get("test_weights") is not None else None

        # Fetch model parameters from freqai_config.json with explicit hardware overrides
        model_training_parameters = self.freqai_info.get("model_training_parameters", {})
        
        # Rigorous memory & CPU configuration for 16GB RAM host
        optimized_params = {
            "objective": model_training_parameters.get("objective", "regression"),
            "metric": model_training_parameters.get("metric", "rmse"),
            "n_estimators": model_training_parameters.get("n_estimators", 250),
            "learning_rate": model_training_parameters.get("learning_rate", 0.05),
            "num_leaves": model_training_parameters.get("num_leaves", 31),
            "max_depth": model_training_parameters.get("max_depth", 6),
            "max_bin": model_training_parameters.get("max_bin", 127),  # Crucial: 127 halves histogram memory
            "n_jobs": model_training_parameters.get("n_jobs", 4),      # Allocate exactly 4 cores to prevent thermal throttling
            "device_type": "cpu",
            "subsample": model_training_parameters.get("subsample", 0.8),
            "colsample_bytree": model_training_parameters.get("colsample_bytree", 0.8),
            "importance_type": model_training_parameters.get("importance_type", "gain"),
            "random_state": 42,
            "verbose": -1
        }

        logger.info(f"Applying LightGBM parameters: {optimized_params}")

        # Initialize Booster
        model = lgb.LGBMRegressor(**optimized_params)

        # Setup callbacks for early stopping if validation sets exist
        callbacks = []
        if eval_set:
            callbacks.append(lgb.early_stopping(stopping_rounds=40, verbose=False))

        # Fit model
        model.fit(
            X_train,
            y_train,
            sample_weight=train_weights,
            eval_set=eval_set,
            eval_sample_weight=eval_weights,
            callbacks=callbacks if callbacks else None
        )

        logger.info("LightGBM CPU training successfully completed.")

        # Extract and persist feature importances for downstream analytics
        try:
            if hasattr(model, "feature_name_") and hasattr(model, "feature_importances_"):
                feat_dict = dict(zip(model.feature_name_, [float(x) for x in model.feature_importances_]))
                # Sort descending
                sorted_feats = dict(sorted(feat_dict.items(), key=lambda item: item[1], reverse=True))
                self.feature_importances_ = sorted_feats
                
                # Save to user_data/models/ for Streamlit analytics dashboard
                if hasattr(dk, "data_path"):
                    model_dir = Path(dk.data_path)
                else:
                    model_dir = Path("user_data/models")
                model_dir.mkdir(parents=True, exist_ok=True)
                
                # Append pair name to tracking file if known
                pair_id = dk.pair.replace("/", "_") if hasattr(dk, "pair") else "general"
                dump_path = model_dir / f"feature_importance_{pair_id}.json"
                
                with open(dump_path, "w") as f:
                    json.dump(sorted_feats, f, indent=4)
                logger.info(f"Persisted top feature importances to {dump_path}")
        except Exception as e:
            logger.error(f"Failed to serialize feature importances: {e}", exc_info=True)

        # Force final garbage collection
        del X_train, y_train
        gc.collect()

        return model
