import gc
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Attempt to import FreqTrade base classes
try:
    from freqtrade.freqai.base_models.BaseClassifierModel import BaseClassifierModel
    from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
except ImportError:
    logger.warning("FreqTrade modules not found. Using mock BaseClassifierModel for standalone compatibility.")
    class BaseClassifierModel:
        def __init__(self, **kwargs):
            self.model = None
            self.freqai_info = {}
        def get_init_model(self, feature_parameters: Dict) -> Dict:
            return {}
    class FreqaiDataKitchen:
        pass


class ClassifierWrapper:
    """
    Wraps the LightGBM classifier to map predictions from [0, 1, 2] back to [-1, 0, 1]
    so they match the FreqTrade strategy label conventions.
    """
    def __init__(self, model: lgb.LGBMClassifier):
        self.model = model

    def predict(self, X):
        # Map [0, 1, 2] predictions back to [-1, 0, 1]
        preds = self.model.predict(X)
        return preds - 1

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    @property
    def classes_(self):
        return np.array(["-1", "0", "1"])

    def __getattr__(self, name):
        if "model" not in self.__dict__:
            raise AttributeError("model attribute is not initialized")
        return getattr(self.model, name)


class LightGBMClassifierCPU(BaseClassifierModel):
    """
    Ternary direction classifier: short / neutral / long (-1, 0, 1).
    Optimized for 16 GB Windows CPU execution.
    """

    def fit(self, data_dictionary: Dict[str, Any], dk: FreqaiDataKitchen, **kwargs) -> Any:
        X_train = data_dictionary["train_features"]
        
        # Multiclass labels must be in range [0, num_class-1]. Map [-1, 0, 1] to [0, 1, 2].
        y_train_raw = data_dictionary["train_labels"].values.ravel()
        y_train = y_train_raw + 1
        
        train_weights = data_dictionary.get("train_weights")

        params = {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "n_estimators": 250,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 6,
            "max_bin": 127,
            "n_jobs": 4,
            "device_type": "cpu",
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "class_weight": "balanced",
            "random_state": 42,
            "verbose": -1,
        }

        model = lgb.LGBMClassifier(**params)

        eval_set = None
        eval_weights = None
        if "test_features" in data_dictionary and "test_labels" in data_dictionary:
            y_test_raw = data_dictionary["test_labels"].values.ravel()
            y_test = y_test_raw + 1
            eval_set = [(data_dictionary["test_features"], y_test)]
            eval_weights = [data_dictionary["test_weights"]] if data_dictionary.get("test_weights") is not None else None

        callbacks = [lgb.early_stopping(40, verbose=False)] if eval_set else None
        model.fit(
            X_train, 
            y_train, 
            sample_weight=train_weights,
            eval_set=eval_set, 
            eval_sample_weight=eval_weights,
            callbacks=callbacks
        )

        # Extract and persist feature importances with timestamp
        try:
            if hasattr(model, "feature_name_") and hasattr(model, "feature_importances_"):
                feat_dict = dict(zip(model.feature_name_, [float(x) for x in model.feature_importances_]))
                sorted_feats = dict(sorted(feat_dict.items(), key=lambda item: item[1], reverse=True))
                
                parent_models_dir = (
                    dk.full_path.parent if hasattr(dk, "full_path") else Path("user_data/models")
                )
                parent_models_dir.mkdir(parents=True, exist_ok=True)
                
                pair_id = dk.pair.replace("/", "_") if hasattr(dk, "pair") else "general"
                
                timestamp = int(time.time())
                if hasattr(dk, "data_path") and "_" in str(dk.data_path):
                    try:
                        timestamp = int(str(dk.data_path).split("_")[-1])
                    except ValueError:
                        pass
                
                dump_path = parent_models_dir / f"feature_importance_{pair_id}_{timestamp}.json"
                with open(dump_path, "w") as f:
                    json.dump(sorted_feats, f, indent=4)
                logger.info(f"Persisted top feature importances to {dump_path}")
                
                # Prune old feature importance files in user_data/models
                try:
                    importance_files = sorted(parent_models_dir.glob(f"feature_importance_{pair_id}_*.json"),
                                              key=lambda p: p.stat().st_mtime, reverse=True)
                    for old in importance_files[3:]:
                        old.unlink(missing_ok=True)
                except Exception as ex:
                    logger.warning(f"Could not purge old feature importance files: {ex}")
        except Exception as e:
            logger.error(f"Failed to serialize feature importances: {e}", exc_info=True)

        del X_train, y_train
        gc.collect()
        
        # Ensure dk.data has class labels initialized to avoid KeyError/AttributeError in data_drawer
        if hasattr(dk, "data"):
            if "labels_mean" not in dk.data:
                dk.data["labels_mean"] = {}
            if "labels_std" not in dk.data:
                dk.data["labels_std"] = {}
            for class_label in ["-1", "0", "1"]:
                dk.data["labels_mean"][class_label] = 0.0
                dk.data["labels_std"][class_label] = 0.0

        # Return the wrapper to map predictions correctly
        return ClassifierWrapper(model)

    def predict(
        self, unfiltered_df: pd.DataFrame, dk: FreqaiDataKitchen, **kwargs
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Filter the prediction features data, predict with the model,
        and calculate the max prediction confidence for classification.
        """
        (pred_df, dk.do_predict) = super().predict(unfiltered_df, dk, **kwargs)

        # Calculate max prediction confidence from the class probability columns
        prob_cols = [c for c in ["-1", "0", "1"] if c in pred_df.columns]
        if prob_cols:
            pred_df["&-direction_max_prediction_confidence"] = pred_df[prob_cols].max(axis=1)
        else:
            pred_df["&-direction_max_prediction_confidence"] = 0.0

        return (pred_df, dk.do_predict)
