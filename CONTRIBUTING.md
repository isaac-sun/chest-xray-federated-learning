# Contributing

Thanks for taking a look at this project. Issues and pull requests are welcome.

## Development setup

```bash
git clone https://github.com/isaac-sun/chest-xray-federated-learning.git
cd chest-xray-federated-learning
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The chest X-ray dataset is **not** distributed with this repository; see the
[Dataset](README.md#dataset) section of the README to obtain it.

## Checks

```bash
make lint   # ruff check .
make test   # pytest
```

Both checks run in CI on every pull request (Python 3.10 and 3.12).

## Project layout

```text
src/xray_fl/   installable package (data, model, federated, evaluate, training entry points)
tests/         pytest suite, no dataset required
configs/       single YAML config for every run
results/       committed metrics + training histories for reference
docs/images/   figures embedded in the README
```

## Pull requests

- Keep changes focused; one topic per pull request.
- Add or update tests when behavior changes.
- Update `README.md` and `README_ZH.md` when commands, config keys, or the layout change.
- Record user-visible changes in `CHANGELOG.md`.

## Style

- Python 3.10+, 4-space indentation, 120-column lines (enforced by ruff).
- Type hints on public functions; docstrings on modules and public callables.
