import logging
from flask import Flask, jsonify, render_template, request
from common.federation_store import store

# Suppress werkzeug logs
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/state")
def get_state():
    return jsonify(store.get_state())


@app.route("/api/control", methods=["POST"])
def control_federation():
    data = request.json
    if not data or "command" not in data:
        return jsonify({"error": "Missing command"}), 400

    command = data["command"].lower()

    if command == "start":
        rounds = int(data.get("max_rounds", 10))
        store.request_start(rounds)
        return jsonify({"message": f"Start requested for {rounds} rounds"})

    elif command == "stop":
        store.request_stop()
        return jsonify({"message": "Stop requested"})

    return jsonify({"error": "Unknown command"}), 400


def start_dashboard(port=8080):
    """Start the dashboard in the current process."""
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
