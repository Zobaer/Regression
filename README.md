# Regression

# Single variable Regression:

From the plots of both training data and test data, it was clear that linear regression will not work because the data is nowhere close to a straight line. It looked like a polynomial regression will work since the graph has partial similarity with the graphs of odd powers of x (some graphs are provided in Fig. 1), but we don’t know the order yet. We can take advantage of linear regression with basis function to introduce polynomial features in the data. We will keep even orders of x in our initial hypothesis function too because we don’t know if the function depends on those too or not.

Normal equations were chosen over running gradient descent loops because of convenience in coding and leveraging matrix manipulation power of Numpy library in Python. Also, the size of the dataset is small, so normal equation will work fine. Below generalized code is written to calculate θ values for the hypothesis function, where the order of the polynomial can be changed easily by changing the value of the variable max_power.

