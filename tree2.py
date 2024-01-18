import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree

# Load the CSV file into a pandas DataFrame
data = pd.read_csv('boston.csv')

included_rows = [505, 324, 167, 129, 418, 471,
                 299, 270, 466, 187, 307, 481,  85, 277, 362]
included_columns = ['CRIM', 'RM', 'Price']
data = data.loc[included_rows, included_columns]

# Convert the DataFrame to a NumPy array
X = data.drop('Price', axis=1)
y = data['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create a regression tree regressor
regressor = DecisionTreeRegressor()

# Fit the regression tree regressor to the training data
regressor.fit(X_train, y_train)


# Visualize the regression tree regressor
plot_tree(regressor, filled=True)

plt.show()
