import threading


class FederationStore:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(FederationStore, cls).__new__(cls)
                cls._instance._init_state()
        return cls._instance

    def _init_state(self):
        self.state = {
            "status": "Initializing",
            "current_round": 0,
            "max_rounds": 0,
            "total_agents": 0,
            "connected_agents": [],
            "rounds_history": [],
            "global_model": {"weights": [], "intercept": []},
            "privacy_budget": 0.0,
            "start_requested": False,
            "stop_requested": False,
            "target_rounds": 0,
        }
        self.lock = threading.Lock()

    def update_status(self, status: str):
        with self.lock:
            self.state["status"] = status

    def update_round(self, current: int, maximum: int):
        with self.lock:
            self.state["current_round"] = current
            self.state["max_rounds"] = maximum

    def set_agents(self, agents_list: list):
        with self.lock:
            self.state["connected_agents"] = agents_list
            self.state["total_agents"] = len(agents_list)

    def add_round_metrics(
        self, round_id: int, accuracy: float, loss: float, num_samples: int
    ):
        with self.lock:
            self.state["rounds_history"].append(
                {
                    "round": round_id,
                    "accuracy": accuracy,
                    "loss": loss,
                    "samples": num_samples,
                }
            )

    def update_global_model(self, model: dict):
        with self.lock:
            self.state["global_model"] = model

    def set_privacy_budget(self, budget: float):
        with self.lock:
            self.state["privacy_budget"] = budget

    def get_state(self):
        with self.lock:
            # Return a shallow copy to avoid concurrent mutation issues during serialization
            # Filter out internal control flags from the public API
            state_dict = dict(self.state)
            state_dict.pop("start_requested", None)
            state_dict.pop("stop_requested", None)
            state_dict.pop("target_rounds", None)
            return state_dict

    # Control Functions
    def request_start(self, rounds: int):
        with self.lock:
            self.state["start_requested"] = True
            self.state["target_rounds"] = rounds
            self.state["stop_requested"] = False

    def request_stop(self):
        with self.lock:
            self.state["stop_requested"] = True
            self.state["start_requested"] = False

    def pop_start_request(self):
        with self.lock:
            if self.state.get("start_requested", False):
                self.state["start_requested"] = False
                return True, self.state.get("target_rounds", 0)
            return False, 0

    def pop_stop_request(self):
        with self.lock:
            if self.state.get("stop_requested", False):
                self.state["stop_requested"] = False
                return True
            return False

    def reset_metrics(self):
        with self.lock:
            self.state["current_round"] = 0
            self.state["rounds_history"] = []
            self.state["global_model"] = {"weights": [], "intercept": []}


# Global singleton instance
store = FederationStore()
