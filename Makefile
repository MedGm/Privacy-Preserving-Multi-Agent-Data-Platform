.PHONY: lint test run clean up down format logs register

test:
	PYTHONPATH=src pytest tests/ -v

lint:
	flake8 src tests --max-line-length=120
	black --check src tests

format:
	black src tests

up:
	docker-compose up -d --build

down:
	docker-compose down

logs:
	docker-compose logs --tail=50

register:
	docker-compose exec -T xmpp-server prosodyctl register coordinator localhost password 2>/dev/null || true
	docker-compose exec -T xmpp-server prosodyctl register agent1 localhost password 2>/dev/null || true
	docker-compose exec -T xmpp-server prosodyctl register agent2 localhost password 2>/dev/null || true

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
