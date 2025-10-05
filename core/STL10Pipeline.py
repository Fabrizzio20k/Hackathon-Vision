import numpy as np
import argparse
import json
from pathlib import Path
import time
import pickle

from config.InitialConfig import InitialConfig
from utils.SIFTBoVW import SIFTBoVW
from utils.FaissDatabase import FaissDatabase, IndexType
from utils.RegionalMultiDescriptor import RegionalMultiDescriptor
from core.ModelEvaluator import ModelEvaluator, RepeatabilityEvaluator
from core.RobustnessEvaluator import RobustnessEvaluator
from utils.HogExtractor import HOGExtractor


class STL10Pipeline:
    def __init__(
        self,
        data_path: str = "./dataset",
        models_path: str = "./models",
        descriptor: str = "sift",
        n_clusters: int = 1000,
        index_type: str = "flat_l2",
        classifier: str = "linear_svm",
        n_runs: int = 3
    ):
        self.data_path = data_path
        self.models_path = Path(models_path)
        self.models_path.mkdir(parents=True, exist_ok=True)

        self.n_clusters = n_clusters
        self.index_type = IndexType[index_type.upper()]
        self.classifier = classifier
        self.n_runs = n_runs

        self.config = InitialConfig(data_path=data_path)
        self.faiss_db = None

        self.descriptor_type = descriptor
        self.descriptor = None

    def step1_download_dataset(self):
        print("\n" + "="*70)
        print("STEP 1: DOWNLOAD DATASET")
        print("="*70)
        self.config.start()

    def step2_train_descriptor(self, sample_size=50000, max_desc_per_img=100):
        print("\n" + "="*70)
        print("STEP 2: TRAIN DESCRIPTOR (UNSUPERVISED)")
        print("="*70)

        if self.descriptor_type == "hog":
            print("Using HOG descriptor (no training needed)")
            self.descriptor = HOGExtractor()
            descriptor_path = self.models_path / "descriptor_model.pkl"
            with open(descriptor_path, 'wb') as f:
                pickle.dump(self.descriptor, f)

        elif self.descriptor_type == "regional_multi":
            print("Using RegionalMultiDescriptor (no training needed)")
            self.descriptor = RegionalMultiDescriptor(
                grid_size=6)  # CAMBIO: 4 en vez de 8
            descriptor_path = self.models_path / "descriptor_model.pkl"
            with open(descriptor_path, 'wb') as f:
                pickle.dump(self.descriptor, f)

        elif self.descriptor_type == "sift":
            print("Using SIFT BoVW - NO LABELS USED")
            unlabeled_images = self.config.load_unlabeled()
            self.descriptor = SIFTBoVW(n_clusters=self.n_clusters)
            self.descriptor.train_codebook(
                unlabeled_images, sample_size, max_desc_per_img)
            descriptor_path = self.models_path / "descriptor_model.pkl"
            self.descriptor.save(str(descriptor_path))

        elif self.descriptor_type == "fisher":
            print("Using Fisher Vector descriptor")
            from utils.FisherVectorExtractor import FisherVectorExtractor

            unlabeled_images = self.config.load_unlabeled()
            self.descriptor = FisherVectorExtractor(
                n_gaussians=48,
                pca_dim=42,
                scales=5,
                stride=8
            )
            self.descriptor.train(unlabeled_images, sample_size=10000)
            descriptor_path = self.models_path / "descriptor_model.pkl"
            self.descriptor.save(str(descriptor_path))

        print(
            f"Descriptor dimension: {self.descriptor.get_feature_dimension()}")

    def step3_extract_features(self):
        print("\n" + "="*70)
        print("STEP 3: EXTRACT FEATURES")
        print("="*70)

        descriptor_path = self.models_path / "descriptor_model.pkl"
        if self.descriptor is None:
            if self.descriptor_type == "sift":
                self.descriptor = SIFTBoVW.load(str(descriptor_path))
            elif self.descriptor_type == "hog":
                with open(descriptor_path, 'rb') as f:
                    self.descriptor = pickle.load(f)
            elif self.descriptor_type == "regional_multi":
                with open(descriptor_path, 'rb') as f:
                    self.descriptor = pickle.load(f)

        print("\nExtracting features from train split (5k images)...")
        train_images, train_labels = self.config.load_train(with_labels=True)

        if self.descriptor_type == "hog":
            print("Encoding train images with HOG...")
            train_vectors = np.array(
                [self.descriptor.extract(img) for img in train_images])
        elif self.descriptor_type == "regional_multi":
            print("Encoding train images with RegionalMulti...")
            train_vectors = np.array(
                [self.descriptor.extract(img) for img in train_images])
        else:
            train_vectors = self.descriptor.encode_images(train_images)

        print("\nExtracting features from test split (8k images)...")
        test_images, test_labels = self.config.load_test(with_labels=True)

        if self.descriptor_type == "hog":
            print("Encoding test images with HOG...")
            test_vectors = np.array(
                [self.descriptor.extract(img) for img in test_images])
        elif self.descriptor_type == "regional_multi":
            print("Encoding test images with RegionalMulti...")
            test_vectors = np.array(
                [self.descriptor.extract(img) for img in test_images])
        else:
            test_vectors = self.descriptor.encode_images(test_images)

        features_path = self.models_path / "features.npz"
        np.savez(
            features_path,
            train_vectors=train_vectors,
            train_labels=train_labels,
            test_vectors=test_vectors,
            test_labels=test_labels,
            test_images=test_images
        )
        print(f"\nFeatures saved to: {features_path}")

        avg_time = 0
        n_samples = min(100, len(test_images))
        start_time = time.time()
        for i in range(n_samples):
            if self.descriptor_type == "hog":
                _ = self.descriptor.extract(test_images[i])
            elif self.descriptor_type == "regional_multi":
                _ = self.descriptor.extract(test_images[i])
            else:
                _ = self.descriptor.encode_image(test_images[i])
        avg_time = (time.time() - start_time) / n_samples
        print(f"Average extraction time per image: {avg_time*1000:.2f}ms")

        return train_vectors, train_labels, test_vectors, test_labels, test_images

    def step4_build_faiss_index(self, train_vectors: np.ndarray):
        print("\n" + "="*70)
        print("STEP 4: BUILD FAISS INDEX")
        print("="*70)

        dimension = train_vectors.shape[1]
        print(
            f"Building {self.index_type.value} index with dimension {dimension}")

        self.faiss_db = FaissDatabase(
            dimension=dimension, index_type=self.index_type)
        self.faiss_db.add(train_vectors)

        faiss_path = self.models_path / "faiss_index"
        self.faiss_db.save(str(faiss_path))
        print(f"FAISS index saved to: {faiss_path}")
        print(f"Total vectors in index: {self.faiss_db.size()}")

    def step5_evaluate(self, train_vectors, train_labels, test_vectors, test_labels):
        print("\n" + "="*70)
        print("STEP 5: SUPERVISED EVALUATION")
        print("="*70)
        print("NOW using labels for evaluation ONLY")

        if self.n_runs > 1:
            print(f"\nRunning {self.n_runs} experiments for repeatability...")
            rep_evaluator = RepeatabilityEvaluator(n_runs=self.n_runs)
            summary = rep_evaluator.run_multiple_experiments(
                train_vectors, train_labels,
                test_vectors, test_labels,
                model_type=self.classifier
            )

            results_path = self.models_path / "evaluation_results.json"
            results_dict = {
                metric: {'mean': float(mean), 'std': float(std)}
                for metric, (mean, std) in summary.items()
            }

            with open(results_path, 'w') as f:
                json.dump(results_dict, f, indent=4)

            print(f"\nResults saved to: {results_path}")

            print("\nTraining final model for robustness evaluation...")
            evaluator = ModelEvaluator(model_type=self.classifier)
            evaluator.train(train_vectors, train_labels)
            model_path = self.models_path / f"{self.classifier}_model.pkl"
            evaluator.save(str(model_path))
            print(f"Final model saved to: {model_path}")

            return summary

        else:
            evaluator = ModelEvaluator(model_type=self.classifier)
            train_time = evaluator.train(train_vectors, train_labels)
            metrics = evaluator.evaluate(
                test_vectors, test_labels, verbose=True)

            model_path = self.models_path / f"{self.classifier}_model.pkl"
            evaluator.save(str(model_path))
            print(f"\nModel saved to: {model_path}")

            return metrics

    def step6_evaluate_robustness(self, test_images, test_labels, original_accuracy):
        print("\n" + "="*70)
        print("STEP 6: ROBUSTNESS EVALUATION")
        print("="*70)

        descriptor_path = self.models_path / "descriptor_model.pkl"
        classifier_path = self.models_path / f"{self.classifier}_model.pkl"

        if not descriptor_path.exists() or not classifier_path.exists():
            print("ERROR: Models not found. Run full pipeline first.")
            return None

        if self.descriptor_type == "sift":
            descriptor = SIFTBoVW.load(str(descriptor_path))
        elif self.descriptor_type == "hog":
            with open(descriptor_path, 'rb') as f:
                descriptor = pickle.load(f)
        elif self.descriptor_type == "regional_multi":
            with open(descriptor_path, 'rb') as f:
                descriptor = pickle.load(f)
        elif self.descriptor_type == "fisher":
            from utils.FisherVectorExtractor import FisherVectorExtractor
            descriptor = FisherVectorExtractor.load(str(descriptor_path))
        else:
            raise ValueError(
                f"Unknown descriptor type: {self.descriptor_type}")

        classifier = ModelEvaluator.load(str(classifier_path))

        evaluator = RobustnessEvaluator()
        results = evaluator.evaluate_robustness(
            descriptor,
            classifier,
            test_images,
            test_labels,
            original_accuracy
        )

        evaluator.print_summary(results, original_accuracy)

        robustness_path = self.models_path / "robustness_results.json"
        evaluator.save_results(results, str(robustness_path))

        return results

    def run_full_pipeline(self, skip_download: bool = False, skip_training: bool = False, skip_robustness: bool = False):
        print("\n" + "="*70)
        print("STL-10 DESCRIPTOR HACKATHON - FULL PIPELINE")
        print("="*70)
        print(f"Configuration:")
        print(f"  - Descriptor: {self.descriptor_type}")
        print(
            f"  - Vocabulary size: {self.n_clusters if self.descriptor_type == 'sift' else 'N/A'}")
        print(f"  - FAISS index: {self.index_type.value}")
        print(f"  - Classifier: {self.classifier}")
        print(f"  - Runs: {self.n_runs}")
        print("="*70)

        if not skip_download:
            self.step1_download_dataset()

        if not skip_training:
            self.step2_train_descriptor()

        train_vectors, train_labels, test_vectors, test_labels, test_images = self.step3_extract_features()

        self.step4_build_faiss_index(train_vectors)

        results = self.step5_evaluate(
            train_vectors, train_labels, test_vectors, test_labels)

        if not skip_robustness:
            if isinstance(results, dict):
                if 'accuracy' in results and isinstance(results['accuracy'], tuple):
                    original_accuracy = results['accuracy'][0]
                elif 'accuracy' in results:
                    original_accuracy = results['accuracy']
                else:
                    print("WARNING: Could not extract accuracy from results")
                    return results
            else:
                original_accuracy = results['accuracy'][0] if isinstance(
                    results['accuracy'], tuple) else results['accuracy']

            self.step6_evaluate_robustness(
                test_images, test_labels, original_accuracy)

        print("\n" + "="*70)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)

        return results


def main():
    parser = argparse.ArgumentParser(
        description='STL-10 Descriptor Hackathon Pipeline')

    parser.add_argument('--data_path', type=str, default='./dataset',
                        help='Path to store dataset')
    parser.add_argument('--models_path', type=str, default='./models',
                        help='Path to store models')
    parser.add_argument('--descriptor', type=str, default='fisher',
                        choices=['sift', 'hog', 'regional_multi', 'fisher'],
                        help='Descriptor type')
    parser.add_argument('--n_clusters', type=int, default=1000,
                        help='Number of visual words (BoVW vocabulary size)')
    parser.add_argument('--index_type', type=str, default='flat_l2',
                        choices=['flat_l2', 'flat_ip', 'hnsw',
                                 'lsh', 'ivf_flat', 'ivf_pq', 'pq'],
                        help='FAISS index type')
    parser.add_argument('--classifier', type=str, default='rbf_svm',
                        choices=['linear_svm', 'rbf_svm', 'knn', 'logistic'],
                        help='Classifier type for evaluation')
    parser.add_argument('--n_runs', type=int, default=3,
                        help='Number of runs for repeatability')
    parser.add_argument('--skip_download', action='store_true',
                        help='Skip dataset download')
    parser.add_argument('--skip_training', action='store_true',
                        help='Skip descriptor training (load existing model)')
    parser.add_argument('--skip_robustness', action='store_true',
                        help='Skip robustness evaluation')

    args = parser.parse_args()

    pipeline = STL10Pipeline(
        data_path=args.data_path,
        models_path=args.models_path,
        descriptor=args.descriptor,
        n_clusters=args.n_clusters,
        index_type=args.index_type,
        classifier=args.classifier,
        n_runs=args.n_runs
    )

    pipeline.run_full_pipeline(
        skip_download=args.skip_download,
        skip_training=args.skip_training,
        skip_robustness=args.skip_robustness
    )


if __name__ == "__main__":
    main()
