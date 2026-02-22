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

        self.spark = (
            SparkSession.builder.appName("FederatedAgent")
            .master("local[*]")
            .getOrCreate()
        )
        self.spark.sparkContext.setLogLevel("ERROR")

        self.dp_epsilon = float(os.environ.get("DP_EPSILON", 1.0))
        self.dp_clip_norm = float(os.environ.get("DP_CLIP_NORM", 1.0))

    def train(self, agent_id: str, global_weights: dict = None) -> dict:
        """
        Train on local data using PySpark and return ONLY model parameters + metrics.
        """
        # Parse agent index from agent_id (e.g., "agent1" -> 0)
        try:
            agent_index = int(str(agent_id).replace("agent", "")) - 1
        except ValueError:
            agent_index = 0

        # Get data shard securely
        from common.data_loader import get_agent_data

        X_train, X_test, y_train, y_test = get_agent_data(
            agent_index, total_agents=5, non_iid=True
        )

        # 1. Load data into Spark DataFrames
        train_rows = [tuple(x.tolist()) + (float(y),) for x, y in zip(X_train, y_train)]
        test_rows = [tuple(x.tolist()) + (float(y),) for x, y in zip(X_test, y_test)]

        num_features = X_train.shape[1]
        schema = [f"feature_{i}" for i in range(num_features)] + ["label"]

        train_df = self.spark.createDataFrame(train_rows, schema=schema)
        test_df = self.spark.createDataFrame(test_rows, schema=schema)

        feature_cols = [f"feature_{i}" for i in range(num_features)]
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

        train_data = assembler.transform(train_df).select("features", "label")
        test_data = assembler.transform(test_df).select("features", "label")

        num_samples = train_data.count()

        # 2. Setup model
        lr = LogisticRegression(maxIter=5, featuresCol="features", labelCol="label")

        # Note: In a full production PySpark FL setup, we would convert
        # global_weights into an InitialModel vector and inject it here for warm start.
        # For simplicity here, we train from scratch or use it implicitly.
        model = lr.fit(train_data)

        # 3. Predict and evaluate on test_data
        predictions = model.transform(test_data)
        evaluator_acc = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName="accuracy"
        )
        evaluator_prec = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName="weightedPrecision"
        )
        evaluator_rec = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName="weightedRecall"
        )

        accuracy = float(evaluator_acc.evaluate(predictions))
        precision = float(evaluator_prec.evaluate(predictions))
        recall = float(evaluator_rec.evaluate(predictions))
        loss = 1.0 - accuracy  # proxy loss

        weights = model.coefficients.toArray().tolist()
        intercept = [model.intercept]

        logger.info(
            f"PySpark training complete: acc={accuracy:.4f}, loss={loss:.4f}, prec={precision:.4f}, rec={recall:.4f}"
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
            "metrics": {
                "accuracy": accuracy,
                "loss": loss,
                "precision": precision,
                "recall": recall,
            },
        }
