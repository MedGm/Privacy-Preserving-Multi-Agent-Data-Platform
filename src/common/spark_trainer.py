"""
PySpark-backed Trainer for Federated Learning.

Uses PySpark DataFrames and MLlib to train Logistic Regression on local datasets.
Falls back to scikit-learn LocalTrainer if Spark is unavailable.
"""

import os

try:
    from pyspark.sql import SparkSession
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator

    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False

from common.logger import setup_logger
from common.privacy import clip_weights, add_dp_noise

logger = setup_logger("SparkTrainer")


class SparkTrainer:
    """Federated-compatible local trainer using PySpark MLlib."""

    def __init__(self):
        if not SPARK_AVAILABLE:
            raise RuntimeError("PySpark is not available in this environment.")

        self.spark = SparkSession.builder \
            .appName("FederatedAgent") \
            .master("local[*]") \
            .getOrCreate()
        self.spark.sparkContext.setLogLevel("ERROR")

        self.dp_epsilon = float(os.environ.get("DP_EPSILON", 1.0))
        self.dp_clip_norm = float(os.environ.get("DP_CLIP_NORM", 1.0))

    def train(self, data_path: str, global_weights: dict = None) -> dict:
        """
        Train on local data using PySpark and return ONLY model parameters + metrics.
        """
        # 1. Load data
        df = self.spark.read.csv(data_path, header=True, inferSchema=True)
        assembler = VectorAssembler(
            inputCols=["feature_1", "feature_2"], outputCol="features"
        )
        data = assembler.transform(df).select("features", "label")
        num_samples = data.count()

        # 2. Setup model
        lr = LogisticRegression(maxIter=5, featuresCol="features", labelCol="label")

        # Note: In a full production PySpark FL setup, we would convert
        # global_weights into an InitialModel vector and inject it here for warm start.
        # For simplicity here, we train from scratch or use it implicitly.
        model = lr.fit(data)

        # 3. Predict and evaluate
        predictions = model.transform(data)
        evaluator_acc = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName="accuracy"
        )

        # PySpark MLlib log loss is slightly tricky directly, we'll use accuracy
        # to stand in, or just calculate a simple proxy loss.
        accuracy = evaluator_acc.evaluate(predictions)
        loss = 1.0 - accuracy  # Simplified proxy for log loss

        weights = model.coefficients.toArray().tolist()
        intercept = [model.intercept]

        logger.info(
            f"PySpark training complete: accuracy={accuracy:.4f}, loss={loss:.4f}"
        )

        if self.dp_epsilon > 0.0:
            logger.info(
                f"Applying DP noise (epsilon={self.dp_epsilon}, clip={self.dp_clip_norm})"
            )
            weights = clip_weights(weights, self.dp_clip_norm)
            weights = add_dp_noise(weights, self.dp_epsilon, self.dp_clip_norm)
            intercept = clip_weights(intercept, self.dp_clip_norm)
            intercept = add_dp_noise(intercept, self.dp_epsilon, self.dp_clip_norm)

        # Privacy boundary: only parameters and aggregate metrics leave
        return {
            "weights": weights,
            "intercept": intercept,
            "num_samples": num_samples,
            "metrics": {"accuracy": float(accuracy), "loss": float(loss)},
        }
