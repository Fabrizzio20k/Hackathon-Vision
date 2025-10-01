import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    average_precision_score
)
from sklearn.preprocessing import label_binarize
from typing import Dict, Tuple, Optional
import pickle
from pathlib import Path
import time


class ModelEvaluator:
    def __init__(self, model_type: str = "linear_svm", **model_params):
        self.model_type = model_type
        self.model = self._create_model(model_type, model_params)
        self.is_trained = False
        self.class_names = [
            'airplane', 'bird', 'car', 'cat', 'deer',
            'dog', 'horse', 'monkey', 'ship', 'truck'
        ]

    def _create_model(self, model_type: str, params: dict):
        if model_type == "linear_svm":
            return LinearSVC(
                max_iter=params.get('max_iter', 1000),
                random_state=params.get('random_state', 42),
                verbose=params.get('verbose', 0)
            )

        elif model_type == "rbf_svm":
            return SVC(
                kernel='rbf',
                C=params.get('C', 1.0),
                gamma=params.get('gamma', 'scale'),
                random_state=params.get('random_state', 42),
                verbose=params.get('verbose', False)
            )

        elif model_type == "knn":
            return KNeighborsClassifier(
                n_neighbors=params.get('n_neighbors', 5),
                weights=params.get('weights', 'uniform'),
                metric=params.get('metric', 'euclidean'),
                n_jobs=params.get('n_jobs', -1)
            )

        elif model_type == "logistic":
            return LogisticRegression(
                max_iter=params.get('max_iter', 1000),
                random_state=params.get('random_state', 42),
                n_jobs=params.get('n_jobs', -1),
                verbose=params.get('verbose', 0)
            )

        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> float:
        print(f"Training {self.model_type}...")
        start_time = time.time()

        self.model.fit(X_train, y_train)
        self.is_trained = True

        train_time = time.time() - start_time
        print(f"Training completed in {train_time:.2f} seconds")

        return train_time

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        return self.model.predict(X)

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        verbose: bool = True
    ) -> Dict[str, float]:
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        start_time = time.time()
        y_pred = self.predict(X_test)
        inference_time = time.time() - start_time

        accuracy = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        weighted_f1 = f1_score(y_test, y_pred, average='weighted')

        metrics = {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'inference_time': inference_time,
            'inference_time_per_sample': inference_time / len(X_test)
        }

        if hasattr(self.model, 'decision_function'):
            try:
                y_scores = self.model.decision_function(X_test)
                y_test_bin = label_binarize(y_test, classes=np.unique(y_test))

                if y_test_bin.shape[1] == 1:
                    y_test_bin = np.hstack([1 - y_test_bin, y_test_bin])

                map_score = average_precision_score(
                    y_test_bin, y_scores, average='macro')
                metrics['mAP'] = map_score
            except:
                pass

        if verbose:
            print("\n" + "="*50)
            print("EVALUATION RESULTS")
            print("="*50)
            print(f"Accuracy (Top-1): {accuracy:.4f}")
            print(f"Macro F1: {macro_f1:.4f}")
            print(f"Weighted F1: {weighted_f1:.4f}")
            if 'mAP' in metrics:
                print(f"mAP: {metrics['mAP']:.4f}")
            print(f"Inference time: {inference_time:.2f}s")
            print(
                f"Time per sample: {metrics['inference_time_per_sample']*1000:.2f}ms")
            print("\nClassification Report:")
            print(classification_report(
                y_test, y_pred, target_names=self.class_names))

        return metrics

    def confusion_matrix(self, X_test: np.ndarray, y_test: np.ndarray) -> np.ndarray:
        y_pred = self.predict(X_test)
        return confusion_matrix(y_test, y_pred)

    def save(self, filepath: str):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        state = {
            'model': self.model,
            'model_type': self.model_type,
            'is_trained': self.is_trained
        }

        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, filepath: str):
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        evaluator = cls(model_type=state['model_type'])
        evaluator.model = state['model']
        evaluator.is_trained = state['is_trained']

        return evaluator


class RepeatabilityEvaluator:
    def __init__(self, n_runs: int = 3):
        self.n_runs = n_runs

    def run_multiple_experiments(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_type: str = "linear_svm",
        **model_params
    ) -> Dict[str, Tuple[float, float]]:

        results = {
            'accuracy': [],
            'macro_f1': [],
            'weighted_f1': [],
            'train_time': [],
            'inference_time': []
        }

        if 'mAP' in results:
            results['mAP'] = []

        for run in range(self.n_runs):
            print(f"\n{'='*50}")
            print(f"RUN {run + 1}/{self.n_runs}")
            print(f"{'='*50}")

            seed = 42 + run
            model_params['random_state'] = seed

            evaluator = ModelEvaluator(model_type, **model_params)
            train_time = evaluator.train(X_train, y_train)
            metrics = evaluator.evaluate(X_test, y_test, verbose=True)

            results['accuracy'].append(metrics['accuracy'])
            results['macro_f1'].append(metrics['macro_f1'])
            results['weighted_f1'].append(metrics['weighted_f1'])
            results['train_time'].append(train_time)
            results['inference_time'].append(metrics['inference_time'])

            if 'mAP' in metrics:
                if 'mAP' not in results:
                    results['mAP'] = []
                results['mAP'].append(metrics['mAP'])

        summary = {}
        for metric_name, values in results.items():
            mean = np.mean(values)
            std = np.std(values)
            summary[metric_name] = (mean, std)

        print("\n" + "="*50)
        print("SUMMARY OF RUNS")
        print("="*50)
        for metric_name, (mean, std) in summary.items():
            print(f"{metric_name}: {mean:.4f} ± {std:.4f}")

        return summary
