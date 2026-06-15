import gc
import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Attempt to import FreqTrade base classes
try:
    from freqtrade.freqai.base_models.BaseRegressor import BaseRegressor
    from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
except ImportError:
    logger.warning("FreqTrade modules not found. Using mock BaseRegressor for CatBoost standalone compatibility.")
    class BaseRegressor:
        def __init__(self, **kwargs):
            self.model = None
            self.freqai_info = {}
        def get_init_model(self, feature_parameters: Dict) -> Dict:
            return {}
    class FreqaiDataKitchen:
        pass


class CatBoostRegressorCPU(BaseRegressor):
    """
    Production CatBoost predictive model alternative for FreqAI.
    Highly optimized for multithreaded CPU inference on 16GB RAM machines.
    
    CatBoost handles raw orderbook and continuous numerical transformations
    with exceptional resistance to overfitting.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.feature_importances_: Optional[Dict[str, float]] = None

    def fit(self, data_dictionary: Dict[str, Any], dk: FreqaiDataKitchen, **kwargs) -> Any:
        """
        Trains the CatBoost Regressor model on rolling historical splits.
        """
        try:
            import catboost as cb
        except ImportError:
            raise ImportError("CatBoost is not installed. Please install it via 'pip install catboost'.")

        logger.info("Initiating CatBoost CPU training cycle...")
        gc.collect()

        X_train: pd.DataFrame = data_dictionary["train_features"]
        y_train: pd.Series = data_dictionary["train_labels"]
        train_weights: Optional[pd.Series] = data_dictionary.get("train_weights")

        # Prepare evaluation split if present
        eval_set = None
        test_size = self.freqai_info.get("data_split_parameters", {}).get("test_size", 0.2)
        
        if test_size > 0 and "test_features" in data_dictionary and "test_labels" in data_dictionary:
            eval_set = (data_dictionary["test_features"], data_dictionary["test_labels"])

        model_training_parameters = self.freqai_info.get("model_training_parameters", {})
        
        # CPU & 16GB memory optimized CatBoost hyperparameter matrix
        cb_params = {
            "iterations": model_training_parameters.get("n_estimators", 250),
            "learning_rate": model_training_parameters.get("learning_rate", 0.05),
            "depth": model_training_parameters.get("max_depth", 6),
            "thread_count": model_training_parameters.get("n_jobs", 4),
            "task_type": "CPU",
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "early_stopping_rounds": 40 if eval_set else None,
            "random_seed": 42,
            "verbose": False
        }

        logger.info(f"Applying CatBoost parameters: {cb_params}")

        # Initialize CatBoost Booster
        model = cb.CatBoostRegressor(**cb_params)

        # Fit model
        model.fit(
            X_train,
            y_train,
            sample_weight=train_weights,
            eval_set=eval_set,
            use_best_model=True if eval_set else False
        )

        logger.info("CatBoost CPU training successfully completed.")

        # Serialize feature importances
        try:
            if hasattr(model, "feature_names_") and hasattr(model, "feature_importances_"):
                feat_dict = dict(zip(model.feature_names_, [float(x) for x in model.feature_importances_]))
                sorted_feats = dict(sorted(feat_dict.items(), key=lambda item: item[1], reverse=True))
                self.feature_importances_ = sorted_feats
                
                if hasattr(dk, "data_path"):
                    model_dir = Path(dk.data_path)
                else:
                    model_dir = Path("user_data/models")
                model_dir.mkdir(parents=True, exist_ok=True)
                
                pair_id = dk.pair.replace("/", "_") if hasattr(dk, "pair") else "general"
                dump_path = model_dir / f"catboost_feature_importance_{pair_id}.json"
                
                with open(dump_path, "w") as f:
                    json.dump(sorted_feats, f, indent=4)
                logger.info(f"Persisted CatBoost top feature importances to {dump_path}")
        except Exception as e:
            logger.error(f"Failed to serialize CatBoost feature importances: {e}", exc_info=True)

        del X_train, y_train
        gc.collect()

        return model
