# Privacy-Preserving Multi-Agent Data Platform

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Apache_Spark-E25A1C?style=for-the-badge&logo=apache-spark&logoColor=white" alt="PySpark" />
  <img src="https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn" />
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker" />
  <img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask" />
  <img src="https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white" alt="MLflow" />
</p>

## Overview

The Privacy-Preserving Multi-Agent Data Platform is a distributed analytics system designed to enable multiple autonomous agents to collaboratively train machine learning models without sharing underlying raw data. By utilizing Federated Learning (Federated Averaging), strict Differential Privacy (DP), and secure message brokering, this platform ensures mathematically bounded data privacy while achieving high predictive performance.

Repository: [Privacy-Preserving-Multi-Agent-Data-Platform](git@github.com:MedGm/Privacy-Preserving-Multi-Agent-Data-Platform.git)

## Architecture

The system is built on a highly containerized, multi-paradigm architecture capable of scaling dynamically. It integrates lightweight Python computation with distributed Big Data engines.

1.  **Coordinator Agent**: The central orchestration node running a Finite State Machine (FSM). It manages agent registration, dictates round synchronization, computes the weighted Federated Average of agent models, and broadcasts the updated global model.
2.  **Training Agents**:
    *   **Standard Agents (scikit-learn)**: Compute nodes executing strictly isolated local training over securely generated non-IID datasets using standard Python ML libraries.
    *   **Big Data Agents (PySpark)**: Enterprise-grade compute nodes utilizing Java Virtual Machines and PySpark DataFrames for distributed processing of local models.
3.  **XMPP Message Broker**: A Prosody server orchestrating asynchronous, secure communication protocols between the Coordinator and all computing Agents via the SPADE framework.
4.  **Privacy Engine**: An isolated differential privacy layer enforcing L2-norm clipping and calibrated Laplace/Gaussian noise to agent gradients before transmission, preventing data reconstruction attacks.
5.  **Telemetry & Dashboarding**: 
    *   **MLflow**: A centralized tracking server logging step-by-step convergence metrics and artifacting the final global model.
    *   **Flask UI**: A live, glassmorphism-styled dashboard polling a thread-safe `FederationStore` to chart progress and allow interactive start/stop control over the federation.

## Federated Workflow

The execution lifecycle follows a strictly orchestrated loop managed by the Coordinator's FSM:

1.  **Idle & Setup**: The Coordinator boots, initializes the MLflow run tracker, and waits in an `IdleState` until a user submits a Start command via the Dashboard.
2.  **Registration**: Agents boot independently, connect to the XMPP broker, and transmit a `JOIN` protocol message to the Coordinator. The Coordinator waits until the required quorum (`MIN_AGENTS`) is present.
3.  **Round Synchronization**: The Coordinator signals the beginning of a training round, sending the current global model state (if any) to all registries.
4.  **Local Computation (Isolated)**:
    *   Agents receive the broadcast.
    *   Agents compute gradients/loss locally on their siloed data splits.
    *   Agents enforce local Differential Privacy via weight clipping and noise injection.
5.  **Aggregation**: The Coordinator waits for all agents to report back. Once all perturbed local models are received, it calculates a weighted Global Average (FedAvg).
6.  **Termination or Loop**: The Coordinator logs the epoch's accuracy/loss to MLflow and the local UI. If the `Target Rounds` are met, the FSM saves the final model and returns to `IdleState`. Otherwise, it loops back to Step 3.

## Prerequisites

*   Docker (v20.10+)
*   Docker Compose (v2.0+)

## Deployment & Execution

1.  **Start the Environment:**
    ```bash
    docker compose up -d --build
    ```

2.  **Access the Platform:**
    *   **Live Dashboard / Control UI**: `http://localhost:8081`
    *   **MLflow Metrics Server**: `http://localhost:5000`

3.  **Run a Federation:**
    *   Open the Live Dashboard.
    *   Input the desired number of `Target Rounds`.
    *   Click **Start Federation**.
    *   Monitor the real-time convergence charts and agent status boards.

## Configuration Parameters

You can manipulate the federation behavior via environment variables defined in the `docker-compose.yml`:

| Variable | Target | Description | Default |
| :--- | :--- | :--- | :--- |
| `MIN_AGENTS` | Coordinator | The quorum required to start aggregation | 5 |
| `DP_EPSILON` | Agents | The privacy budget (ε). Lower is more private, 0 disables DP | 1.0 |
| `DP_CLIP_NORM` | Agents | L2-norm threshold for bounding gradients | 1.0 |
| `USE_SPARK` | Agents | Toggles PySpark integration vs scikit-learn | false |
| `XMPP_SERVER` | All | Network hostname of the message broker | localhost |
