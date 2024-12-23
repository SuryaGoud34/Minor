#Perform and plot under-fitting and overfitting in a data set using Python


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

# True generating function
def true_gen(x):
    return np.sin(1.2 * x * np.pi)

# Generate data with noise
np.random.seed(42)
x = np.sort(np.random.rand(120))
y = true_gen(x) + 0.1 * np.random.randn(len(x))

# Split data into training and test sets
random_ind = np.random.choice(list(range(120)), size=120, replace=False)
xt = x[random_ind]
yt = y[random_ind]
train = xt[:int(0.7 * len(x))]
test = xt[int(0.7 * len(x)):]
y_train = yt[:int(0.7 * len(y))]
y_test = yt[int(0.7 * len(y)):]

# Model function for polynomial regression
def fit_poly(train, y_train, test, y_test, degrees, plot='train', return_scores=False):
    features = PolynomialFeatures(degree=degrees, include_bias=False)
    train = train.reshape((-1, 1))
    train_trans = features.fit_transform(train)

    model = LinearRegression()
    model.fit(train_trans, y_train)

    cross_valid = cross_val_score(model, train_trans, y_train, scoring='neg_mean_squared_error', cv=5)

    train_predictions = model.predict(train_trans)
    training_error = mean_squared_error(y_train, train_predictions)

    test = test.reshape((-1, 1))
    test_trans = features.fit_transform(test)
    test_predictions = model.predict(test_trans)
    testing_error = mean_squared_error(y_test, test_predictions)

    x_curve = np.linspace(0, 1, 100)
    x_curve = x_curve.reshape((-1, 1))
    x_curve_trans = features.fit_transform(x_curve)
    model_curve = model.predict(x_curve_trans)
    y_true_curve = true_gen(x_curve[:, 0])

    if plot == 'train':
        plt.plot(train[:, 0], y_train, 'ko', label='Observations')
        plt.plot(x_curve[:, 0], y_true_curve, linewidth=4, label='True Function')
        plt.plot(x_curve[:, 0], model_curve, linewidth=4, label='Model Function')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.ylim(-1, 1.5)
        plt.xlim(0, 1)
        plt.title(f'{degrees} Degree Model on Training Data')
        plt.show()
    elif plot == 'test':
        plt.plot(test, y_test, 'o', label='Test Observations')
        plt.plot(x_curve[:, 0], y_true_curve, 'b-', linewidth=2, label='True Function')
        plt.plot(test, test_predictions, 'ro', label='Test Predictions')
        plt.ylim(-1, 1.5)
        plt.xlim(0, 1)
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'{degrees} Degree Model on Testing Data')
        plt.show()

    if return_scores:
        return training_error, testing_error, -np.mean(cross_valid)

# Experimenting with different degrees
fit_poly(train, y_train, test, y_test, degrees=1, plot='train')
fit_poly(train, y_train, test, y_test, degrees=1, plot='test')

fit_poly(train, y_train, test, y_test, degrees=25, plot='train')
fit_poly(train, y_train, test, y_test, degrees=25, plot='test')

fit_poly(train, y_train, test, y_test, degrees=5, plot='train')
fit_poly(train, y_train, test, y_test, degrees=5, plot='test')

# Collecting errors for different degrees
degrees = [int(x) for x in np.linspace(1, 40, 40)]
results = pd.DataFrame(0, columns=['train_error', 'test_error'], index=degrees)
for degree in degrees:
    degree_results = fit_poly(train, y_train, test, y_test, degree, plot=False, return_scores=True)
    results.loc[degree, 'train_error'] = degree_results[0]
    results.loc[degree, 'test_error'] = degree_results[1]

# Plot the errors
plt.plot(results.index, results['train_error'], 'b-o', ms=6, label='Training Error')
plt.plot(results.index, results['test_error'], 'r-*', ms=6, label='Testing Error')
plt.legend(loc=2)
plt.xlabel('Degrees')
plt.ylabel('Mean Squared Error')
plt.title('Training and Testing Curves')
plt.ylim(0, 0.05)
plt.show()
