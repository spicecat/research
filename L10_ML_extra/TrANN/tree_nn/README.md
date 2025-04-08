
# Tree-based Neural Network Models

This repository contains Python scripts to simulate various tree-based neural network models.

## Models

1. `TREENN1`: One tree in the input layer.
2. `TREENN2`: One tree in the hidden layer.
3. `TREENN3`: One tree in the output layer.
4. `FONN1`: Multiple trees in the input layer.
5. `FONN2`: Multiple trees in the hidden layer.
6. `FONN3`: Multiple trees in the output layer.

## Requirements

- Python 3.x
- numpy
- pandas

Install the required packages using:

```
pip install -r requirements.txt
```

## Running the Models

1. Import the desired model script.
2. Call the respective function with the input data.
3. Collect and save results.

Example:

```python
from TREENN1 import run_tree_in_input_layer

data = ...  # Your input data
results = run_tree_in_input_layer(data)
print(results)
```

## Main File

The `main.py` script is included to run all models and save their results to a CSV file.
