# Regression
This project demonstrates single variable regression with polynomial basis function (SingleVarRegression.ipynb) and multi-variable regression (MultipleFeatureRegression.ipynb)

# Single variable Regression:

Training data plot:

![alt text](https://github.com/Zobaer/Regression/blob/main/figs/train_data_plot.png)

From the plot training data, it was clear that linear regression will not work because the data is nowhere close to a straight line. It looked like a polynomial regression will work since the graph has partial similarity with the graphs of odd powers of x (some graphs are provided in Fig. 1), but we don’t know the order yet. We can take advantage of linear regression with basis function to introduce polynomial features in the data. We will keep even orders of x in our initial hypothesis function too because we don’t know if the function depends on those too or not.

Normal equations were chosen over running gradient descent loops because of convenience in coding and leveraging matrix manipulation power of Numpy library in Python. Also, the size of the dataset is small, so normal equation will work fine. Below generalized code is written to calculate θ values for the hypothesis function, where the order of the polynomial can be changed easily by changing the value of the variable max_power.

Using max_power = 5, we get nearly perfect match between training data and predicted data as shown in below figure:

![alt text](https://github.com/Zobaer/Regression/blob/main/figs/5th_order.png)

Test data plot:

![alt text](https://github.com/Zobaer/Regression/blob/main/figs/testdata.png)

The final polynomial after ignoring power of x with very small coefficient, the regression equation becomes:

y(x) = 6 + 10x -8x^3+x^5

Training and test errors calculated in code:
Train error (new):  7.761201853041103e-25
Test error (new):  3.686359336049432e-26



# Multiple feature Regression:

This is another linear regression problem, this time we have 11 features as seen from the training data. In the given matrix, last 10 rows were used as test data and the rest were used as training data. The last column was used as target value (price), the first column (house ID) was ignored because it is basically a serial number, not a feature. We used the same normal equations which we used in question 1 to find the equation that model the relationship. 

The results are:

Theta:  [20.04692057  2.48971532 22.76277055 -0.09954098  2.10212676  1.58283592 -8.39396231  7.31574099  0.05130734 -1.35776322 -0.99222408  5.86859344]

Predicted values (Evaluated hypothesis function) for training data:  [25.98288969 32.61373219 27.8745902  25.8563544  27.57561236 26.46236518
 29.06819455 29.84476567 84.16974735 82.3492236  33.2957776  30.13518358
 32.36481642 32.02255193 31.49634246 29.84476567 39.1265951  43.51649205]

Predicted values (Evaluated hypothesis function) for test data:  [46.08761514 27.38335221 36.05912813 48.06849685 50.29069254 41.16484288
 49.0898144  27.61588988 39.72356493 63.18392837]

Train error:  27.239758783632222
Test error:  586.3818075939682

So, from theta values the model equation becomes:

y(x)= 20.04692057 + 2.48971532x_1 + 22.76277055x_2 - 0.09954098x_3 + 2.10212676x_4 +1.58283592x_5 - 8.39396231x_6 + 7.31574099x_7 + 0.05130734x_8 - 1.35776322x_9 - 0.99222408x_10 + 5.86859344x_11

Where:
x1 = Local price
x2 = Bathrooms
x3 = Land area
x4 = Living area
x5 = Number of garages
x6 = Number of rooms
x7 = Number of bedrooms
x8 = Age of home
x9 = Construction
x10 = Architecture
x11 = Number of fire places
y = House price in $1000

To visualize the results, given data and predicted data are plotted in below figure (codes are given in the corresponding colab notebook file). In x-axis, just counting indices are used:

![alt text](https://github.com/Zobaer/Regression/blob/main/figs/multivar_training_plot.png)

![alt text](https://github.com/Zobaer/Regression/blob/main/figs/multivar_test_plot.png)

1)	The least square errors are:

Train error:  27.239758783632222

Test error:  586.3818075939682

The errors are not close to zero because the system has multiple variables and overall a very complex relationship. Nevertheless, the model fits the training data well enough and predicted data partially.

2)	In order to find the factor that have most (or least) effect on final value, we found maximum (or minimum) of absolute theta values using below code:

#Find the minimum of absolute values of theta, to find the feature that has least effect on prediction

#Find the max for maximum effect on prediction

min = np.min(np.abs(theta))

max = np.max(np.abs(theta))

print("min :", min)

print("max :", max)

The results are: 

min : 0.05130733993775749

max : 22.762770552300026

The max value is the co-efficient of x2 which is bathrooms which has the maximum impact (or maximum positive impact since the coefficient was originally positive) on house prices. But this value alone cannot be used to predict house prices, because this variable can have only a few values like 1, 1.5, 2, 2.5 etc. which will not be able to predict a wide range of prices of houses with multiple other more important features. 

After doing regression analysis (by adding below lines of code) with only the number of bathrooms and looking at prediction error, it seems that the prediction is not that worse than before, but from the test data plot with predicted data it is clear that the prediction is not at all good, we almost have binary data output (very high and very low as seen from the red points in the below graph).

#taking only number of bathrooms

x_train = x_train[:,1]

x_test = x_test[:,1]

Train error:  107.54094594594605

Test error:  316.1574242147554

![alt text](https://github.com/Zobaer/Regression/blob/main/figs/multi_one_var.png)

The minimum one among the negative theta values is -8.39396231 which is the coefficient of x6 (number of rooms). Hence this variable has the most negative impact. But this variable cannot be used alone to predict target value because with increasing number of rooms, the house prices are supposed to increase but here we are having a negative impact. Also, it has been tested with linear regression (by adding below lines of codes) that if we use only the number of rooms, then prediction error increases significantly:

#taking only number of rooms

x_train = x_train[:,5]

x_test = x_test[:,5]

Train error:  763.1753438113948

Test error:  379.77220734056147

3)	The minimum of absolute values of theta is 0.05130733993775749 which is the coefficient of x8 or age of home. Linear regression analysis using normal equations is run again excluding this variable using below additional line of codes to delete that column from dataset:

x_train = np.delete(x_train,7,1)

x_test = np.delete(x_test,7,1)

Full code is given in the colab notebook. The new results are given below:

Theta:  [22.50861666  2.54529555 20.62980335 -0.2760444   3.15104891  1.54142716
 -7.90310435  6.85630943 -1.21755292 -1.57170462  5.77285692]

Predicted values (Evaluated hypothesis function) for training data:  [26.26205711 32.06680453 28.46084351 25.7520209  27.7867659  25.6105424
 28.99768553 29.86482811 84.04814359 82.37477097 33.5004997  30.47345283
 32.02654717 32.22490515 31.35404816 29.86482811 39.10590457 43.82535174]

Predicted values (Evaluated hypothesis function) for test data:  [46.32645163 27.90881232 36.38126265 48.77401664 49.11113709 40.56059519
 48.07423846 28.78533353 40.53923691 63.17955503]

Train error:  28.21517251699303

Test error:  553.5627691257249

It is seen that the errors are remaining almost the same (train error is slightly increased while the test error is slightly decreased) with one variable (that has the least impact) removed. So, removing the feature that has the least impact (the feature is Age of home) will have negligible effect on predicting house prices.




