import os
import pickle
from utils import log_success, log_warning, log_error
from preprocessing import FeaturePipeline, SchemaError


class AnomalyDetector:
    """
    38-feature detector.
    - Reads models from src/models/
    - All encoding and scaling go through preprocessing.FeaturePipeline, the same
      code path train_models_v2.ipynb and evaluations_v2.ipynb use.
    - Refuses a schema-v1 feature_meta.json (categories stored as integer codes),
      which previously made every proto/state encode to -1.

    Feature contract validation is handled upstream in receiver.py
    (_check_feature_completeness), which reads self.meta.
    """

    def __init__(self, model_dir=None):
        if model_dir is None:
            here = os.path.dirname(os.path.abspath(__file__))
            model_dir = os.path.join(here, "models")

        self.model_dir = model_dir
        self.models = {}
        self.pipeline = None
        self.meta = None
        self.unknown_category_count = 0

        self._debug_counter = 0

        self._load_pipeline()
        self._load_models()

        # Backwards compatibility if any code calls .predict()
        self.predict = self.detect

    def _load_pipeline(self):
        try:
            self.pipeline = FeaturePipeline.load(self.model_dir)
            self.meta = self.pipeline.to_meta()
            log_success(f"Loaded feature pipeline (schema v{self.meta['schema_version']}) from: {self.model_dir}")
            log_success(f"Total features in contract: {len(self.meta['feature_cols'])}")
        except (SchemaError, FileNotFoundError) as e:
            log_error(f"Feature pipeline unavailable: {e}")
            self.pipeline = None
            self.meta = None
        except Exception as e:
            log_error(f"Failed to load feature pipeline: {e}")
            self.pipeline = None
            self.meta = None

    def _load_models(self):
        model_files = {
            "iforest": "isolation_forest.pkl",
            "lof": "local_outlier_factor.pkl",
            "ocsvm": "one_class_svm.pkl",
        }

        loaded = 0
        for name, filename in model_files.items():
            path = os.path.join(self.model_dir, filename)
            if not os.path.exists(path):
                log_warning(f"Model not found: {path}")
                continue
            try:
                with open(path, "rb") as f:
                    self.models[name] = pickle.load(f)
                log_success(f"Loaded {name} model from: {path}")
                loaded += 1
            except Exception as e:
                log_warning(f"Could not load {name} model ({path}): {e}")

        if loaded == 0:
            log_warning("No trained models loaded. Detection will not be reliable.")
        else:
            log_success(f"{loaded}/3 models loaded. Majority vote requires >{loaded / 2:g} agreeing.")

    def detect(self, feats: dict):
        """
        Returns:
            is_attack (bool)
            confidence (float 0..1)   fraction of models that voted anomaly
            votes (dict)              per-model verdict + metadata

        Note: The alert threshold applied to confidence lives in
        NIDSReceiver.alert_threshold (receiver.py), not here.
        """
        if self.pipeline is None:
            log_warning("Feature pipeline missing - skipping detection")
            return False, 0.0, {"error": "no_pipeline"}

        try:
            x, unknown = self.pipeline.prepare_record(feats)
        except Exception as e:
            log_error(f"Feature vector build failed: {e}")
            return False, 0.0, {"error": "vector_build_failed"}

        if unknown:
            self.unknown_category_count += 1

        self._debug_counter += 1
        if self._debug_counter % 500 == 1:
            print(f"[DEBUG] vector shape {x.shape}; unknown categoricals so far: "
                  f"{self.unknown_category_count}/{self._debug_counter}")

        votes = {}
        preds = []
        for name, model in self.models.items():
            try:
                pred = model.predict(x)[0]
                # sklearn convention: -1 = anomaly, +1 = normal
                is_anom = (pred == -1)
                votes[name] = bool(is_anom)
                preds.append(is_anom)
            except Exception as e:
                log_warning(f"{name} inference failed: {e}")

        votes["models_loaded"] = len(self.models)
        votes["models_expected"] = 3
        votes["unknown_categories"] = unknown

        if preds:
            attack_votes = int(sum(preds))
            total_votes = int(len(preds))
            is_attack = attack_votes > (total_votes / 2)
            confidence = attack_votes / total_votes
        else:
            is_attack = False
            confidence = 0.0

        return bool(is_attack), float(confidence), votes