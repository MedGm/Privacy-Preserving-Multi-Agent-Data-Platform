import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def get_agent_data(agent_index: int, total_agents: int, non_iid: bool = True):
    """
    Loads the Breast Cancer dataset, scales it, and slices it for a specific agent.
    If non_iid is True, it simulates non-IID distributions by sorting by a feature before splitting.
    """
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # Scale the complex medical features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    if non_iid:
        # Sort by the first feature (mean radius) to create non-IID shards
        sort_idx = np.argsort(X[:, 0])
        X = X[sort_idx]
        y = y[sort_idx]

    # Split the dataset into partitions
    partitions_X = np.array_split(X, total_agents)
    partitions_y = np.array_split(y, total_agents)

    agent_X = partitions_X[agent_index]
    agent_y = partitions_y[agent_index]

    # Split local agent data into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        agent_X, agent_y, test_size=0.2, random_state=42 + agent_index
    )

    return X_train, X_test, y_train, y_test
