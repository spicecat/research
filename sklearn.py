import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree

# Load the CSV file into a pandas DataFrame
data = pd.read_csv('boston.csv')

included_rows = [505, 324, 167, 129, 418, 471,
                 299, 270, 466, 187, 307, 481,  85, 277, 362]
included_columns = ['CRIM', 'ZN', 'INDUS', 'Price']
data = data.loc[included_rows, included_columns]

# Convert the DataFrame to a NumPy array
X = data.drop('Price', axis=1)
y = data['Price']

# Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create a regression tree regressor
regressor = DecisionTreeRegressor(min_samples_leaf=2)

# Fit the regression tree regressor to the training data
# regressor.fit(X_train, y_train)
regressor.fit(X, y)
print(X)

# Visualize the regression tree regressor
plot_tree(regressor, filled=True, fontsize=5)

# plt.savefig('tree2.png')
plt.show()
