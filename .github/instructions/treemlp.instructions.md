---
applyTo: "**"
---

# Coding Standards and Preferences for TreeMLP Project

## General Principles

- **Clarity and Readability**: Code should be easy to understand. Prioritize clarity over excessive brevity.
- **Modularity**: Break down complex logic into smaller, reusable functions or classes.
- **Consistency**: Follow consistent naming conventions and coding styles throughout the project.
- **DRY (Don't Repeat Yourself)**: Avoid redundant code. Use functions and loops effectively.
- **Comments**:
  - Write clear and concise comments to explain non-obvious logic, assumptions, or important decisions.
  - Avoid over-commenting obvious code.
  - Keep comments up-to-date with code changes.
- **Version Control**:
  - Commit frequently with clear, descriptive messages.
  - Use branches for new features or experiments.

## Python Specific

- **Style Guide**: Adhere to [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/).
  - Use a linter (e.g., Flake8, Pylint) and Black formatter to maintain consistency.
- **Naming Conventions**:
  - `snake_case` for functions, methods, variables, and module names.
  - `PascalCase` (or `CapWords`) for class names.
  - `UPPER_SNAKE_CASE` for constants.
  - Descriptive names are preferred over overly short ones.
- **Type Hinting**: Use type hints (PEP 484) for function signatures and important variables to improve code clarity and enable static analysis.
- **Docstrings**: Write comprehensive docstrings for all public modules, functions, classes, and methods, following [PEP 257 -- Docstring Conventions](https://www.python.org/dev/peps/pep-0257/). (e.g., Google style, NumPy style).
- **Imports**:
  - Import modules, not individual functions/classes from modules, where it improves clarity (e.g., `import numpy as np` vs. `from numpy import array`).
  - Group imports: standard library, third-party, local application/library specific.
- **Error Handling**: Use specific exception types. Handle exceptions gracefully.

## Jupyter Notebooks (`.ipynb`)

- **Cell Organization**:
  - Start notebooks with a Markdown cell providing a title and a brief overview.
  - Use Markdown cells to explain steps, methodologies, and results.
  - Keep code cells focused on a single logical step or a few related operations. Avoid overly long code cells.
  - Clearly separate data loading, preprocessing, model definition, training, evaluation, and visualization steps into distinct cells or sections.
- **Output Management**:
  - Clear irrelevant or excessive output before committing notebooks.
  - For long-running cells, consider adding print statements to indicate progress.
- **Reproducibility**:
  - Set random seeds (e.g., `numpy.random.seed()`, `random.seed()`, TensorFlow/PyTorch seeds) at the beginning of the notebook for reproducible results.
  - List all dependencies (e.g., in a `%pip freeze > requirements.txt` style or a dedicated cell for `%pip install`).
- **Kernel Management**: Ensure the notebook runs cleanly from top to bottom in a fresh kernel.

## Machine Learning Specific

- **Data Preprocessing**: Clearly document and implement all preprocessing steps. Make them reproducible.
- **Model Configuration**: Store model hyperparameters and configurations in a clear and accessible way (e.g., dictionaries, config files).
- **Evaluation Metrics**: Choose and clearly state appropriate evaluation metrics for the task. Report them consistently.
- **Experiment Tracking**: For extensive experiments, consider using tools like MLflow or Weights & Biases to log parameters, metrics, and artifacts.
- **Code Structure for ML Projects**:
  - Separate data loading/processing logic from model definition and training scripts.
  - Consider a structure like:
    - `data/` (or `datasets/`)
    - `notebooks/` (for exploration)
    - `src/` (or project_name/)
      - `data_processing.py`
      - `models.py`
      - `train.py`
      - `evaluate.py`
      - `utils.py`
    - `configs/`
    - `scripts/`

## Tooling

- **Linters**: Use Flake8 or Pylint.
- **Formatters**: Use Black or autopep8.
- **Testing**: Write unit tests for critical components, especially utility functions and data processing logic. Use a framework like `pytest`.

---

_This instructions file should be reviewed and updated periodically as the project evolves._
