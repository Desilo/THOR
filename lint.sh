echo 'ruff format:'
poetry run ruff format src/
echo 'ruff check:'
poetry run ruff check --fix src/
