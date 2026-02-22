import os
import random
import csv


def generate_mock_data(
    agent_id: str,
    num_samples: int = 500,
    seed: int = None,
    class_ratio: float = 0.5,
):
    """
    Generate synthetic data strictly isolated for a given agent.
    This simulates D_i which never leaves the agent's local environment.

    Args:
        agent_id: Unique agent identifier.
        num_samples: Number of samples to generate.
        seed: Random seed for reproducibility.
        class_ratio: Probability of generating class 0 (default 0.5 = IID).
                     Set to 0.8 for agent1 (mostly class 0) and 0.2 for agent2
                     (mostly class 1) to simulate non-IID data.
    """
    if seed is not None:
        random.seed(seed)

    os.makedirs("data", exist_ok=True)
    filepath = f"data/{agent_id}_dataset.csv"

    with open(filepath, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["feature_1", "feature_2", "label"])

        for _ in range(num_samples):
            if random.random() < class_ratio:
                # Class 0: centered around (0, 0)
                writer.writerow([random.gauss(0, 1), random.gauss(0, 1), 0])
            else:
                # Class 1: centered around (3, 3)
                writer.writerow([random.gauss(3, 1), random.gauss(3, 1), 1])

    return filepath


if __name__ == "__main__":
    # Non-IID: agent1 has mostly class 0, agent2 has mostly class 1
    generate_mock_data("agent1", 500, seed=42, class_ratio=0.7)
    generate_mock_data("agent2", 500, seed=43, class_ratio=0.3)
    print("Mock datasets generated in ./data/")
