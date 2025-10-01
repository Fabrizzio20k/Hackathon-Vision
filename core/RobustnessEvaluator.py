import numpy as np
import cv2
from typing import Dict, Callable, List, Tuple
from tqdm import tqdm
import json
from pathlib import Path


class RobustnessEvaluator:
    def __init__(self):
        self.transformations = {
            'gaussian_blur': self.apply_gaussian_blur,
            'rotation': self.apply_rotation,
            'scale': self.apply_scale,
            'brightness': self.apply_brightness,
            'contrast': self.apply_contrast,
            'jpeg_compression': self.apply_jpeg_compression
        }

    def apply_gaussian_blur(self, image: np.ndarray, sigma: float = 1.5) -> np.ndarray:
        ksize = int(2 * np.ceil(3 * sigma) + 1)
        if ksize % 2 == 0:
            ksize += 1
        return cv2.GaussianBlur(image, (ksize, ksize), sigma)

    def apply_rotation(self, image: np.ndarray, angle: float = 15.0) -> np.ndarray:
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    def apply_scale(self, image: np.ndarray, scale_factor: float = 0.8) -> np.ndarray:
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)

        scaled = cv2.resize(image, (new_w, new_h),
                            interpolation=cv2.INTER_LINEAR)

        if scale_factor < 1.0:
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            result = np.zeros_like(image)
            result[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = scaled
            return result
        else:
            crop_h = (new_h - h) // 2
            crop_w = (new_w - w) // 2
            return scaled[crop_h:crop_h+h, crop_w:crop_w+w]

    def apply_brightness(self, image: np.ndarray, delta: int = 50) -> np.ndarray:
        result = image.astype(np.int16) + delta
        result = np.clip(result, 0, 255).astype(np.uint8)
        return result

    def apply_contrast(self, image: np.ndarray, factor: float = 1.5) -> np.ndarray:
        mean = image.mean()
        result = (image.astype(np.float32) - mean) * factor + mean
        result = np.clip(result, 0, 255).astype(np.uint8)
        return result

    def apply_jpeg_compression(self, image: np.ndarray, quality: int = 40) -> np.ndarray:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode('.jpg', cv2.cvtColor(
            image, cv2.COLOR_RGB2BGR), encode_param)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)

    def transform_images(
        self,
        images: np.ndarray,
        transformation: str,
        **kwargs
    ) -> np.ndarray:
        if transformation not in self.transformations:
            raise ValueError(f"Unknown transformation: {transformation}")

        transform_fn = self.transformations[transformation]
        transformed = []

        for img in tqdm(images, desc=f"Applying {transformation}"):
            transformed_img = transform_fn(img, **kwargs)
            transformed.append(transformed_img)

        return np.array(transformed)

    def evaluate_robustness(
        self,
        descriptor_extractor,
        classifier,
        test_images: np.ndarray,
        test_labels: np.ndarray,
        original_accuracy: float
    ) -> Dict[str, Dict]:

        results = {}

        transformations_params = {
            'gaussian_blur': {'sigma': 1.5},
            'rotation': {'angle': 15.0},
            'scale_down': {'scale_factor': 0.8},
            'scale_up': {'scale_factor': 1.2},
            'brightness_increase': {'delta': 50},
            'brightness_decrease': {'delta': -50},
            'contrast_increase': {'factor': 1.5},
            'contrast_decrease': {'factor': 0.7},
            'jpeg_compression': {'quality': 40}
        }

        for trans_name, params in transformations_params.items():
            print(f"\n{'='*60}")
            print(f"Evaluating: {trans_name}")
            print(f"Parameters: {params}")
            print('='*60)

            if trans_name.startswith('scale_'):
                base_trans = 'scale'
            elif trans_name.startswith('brightness_'):
                base_trans = 'brightness'
            elif trans_name.startswith('contrast_'):
                base_trans = 'contrast'
            else:
                base_trans = trans_name

            transformed_images = self.transform_images(
                test_images,
                base_trans,
                **params
            )

            print("Extracting features from transformed images...")
            if hasattr(descriptor_extractor, 'encode_images'):
                transformed_vectors = descriptor_extractor.encode_images(
                    transformed_images)
            else:
                transformed_vectors = np.array([descriptor_extractor.extract(img)
                                                for img in tqdm(transformed_images, desc="Encoding")])

            print("Evaluating on transformed images...")
            y_pred = classifier.predict(transformed_vectors)

            from sklearn.metrics import accuracy_score, f1_score
            accuracy = accuracy_score(test_labels, y_pred)
            macro_f1 = f1_score(test_labels, y_pred, average='macro')

            accuracy_drop = original_accuracy - accuracy
            relative_drop = (accuracy_drop / original_accuracy) * 100

            results[trans_name] = {
                'accuracy': float(accuracy),
                'macro_f1': float(macro_f1),
                'accuracy_drop': float(accuracy_drop),
                'relative_drop_percent': float(relative_drop),
                'parameters': params
            }

            print(f"Accuracy: {accuracy:.4f}")
            print(f"Accuracy drop: {accuracy_drop:.4f} ({relative_drop:.2f}%)")
            print(f"Macro F1: {macro_f1:.4f}")

        return results

    def save_results(self, results: Dict, filepath: str):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=4)

        print(f"\nRobustness results saved to: {filepath}")

    def print_summary(self, results: Dict, original_accuracy: float):
        print("\n" + "="*70)
        print("ROBUSTNESS EVALUATION SUMMARY")
        print("="*70)
        print(f"Original Accuracy: {original_accuracy:.4f}")
        print("\n{:<25} {:<12} {:<15} {:<15}".format(
            "Transformation", "Accuracy", "Drop", "Rel. Drop %"
        ))
        print("-"*70)

        for trans_name, metrics in results.items():
            print("{:<25} {:<12.4f} {:<15.4f} {:<15.2f}".format(
                trans_name,
                metrics['accuracy'],
                metrics['accuracy_drop'],
                metrics['relative_drop_percent']
            ))

        avg_accuracy = np.mean([m['accuracy'] for m in results.values()])
        avg_drop = np.mean([m['accuracy_drop'] for m in results.values()])
        avg_rel_drop = np.mean([m['relative_drop_percent']
                               for m in results.values()])

        print("-"*70)
        print("{:<25} {:<12.4f} {:<15.4f} {:<15.2f}".format(
            "AVERAGE",
            avg_accuracy,
            avg_drop,
            avg_rel_drop
        ))
        print("="*70)


def run_robustness_evaluation(
    bovw_model_path: str,
    classifier_model_path: str,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    original_accuracy: float,
    output_path: str = "./models/robustness_results.json"
):
    from utils.SIFTBoVW import SIFTBoVW
    from .ModelEvaluator import ModelEvaluator

    print("\n" + "="*70)
    print("ROBUSTNESS EVALUATION")
    print("="*70)

    print("Loading models...")
    bovw = SIFTBoVW.load(bovw_model_path)
    classifier = ModelEvaluator.load(classifier_model_path)

    evaluator = RobustnessEvaluator()

    results = evaluator.evaluate_robustness(
        bovw,
        classifier,
        test_images,
        test_labels,
        original_accuracy
    )

    evaluator.print_summary(results, original_accuracy)
    evaluator.save_results(results, output_path)

    return results
