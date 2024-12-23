import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import matplotlib
import matplotlib.pyplot as plt

# %matplotlib inline
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['figure.titlesize'] = 16
matplotlib.rcParams['figure.figsize'] = [9, 7]

# Set the random seed for reproducible results
np.random.seed(42)

# "True" generating function representing a process in real life
def true_gen(x):
    return np.sin(1.2 * x * np.pi)

# x values and y value with a small amount of random noise
x = np.sort(np.random.rand(120))
y = true_gen(x) + 0.1 * np.random.randn(len(x))

# Random indices for creating training and testing sets
random_ind = np.random.choice(list(range(120)), size=120, replace=False)
xt = x[random_ind]
yt = y[random_ind]

# Training and testing observations (x values)
train = xt[:int(0.7 * len(x))]  # 70% train set
test = xt[int(0.7 * len(x)):]   # 30% test set
# y values
y_train = yt[:int(0.7 * len(y))]
y_test = yt[int(0.7 * len(y)):]

# Model the true curve
x_linspace = np.linspace(0, 1, 1000)
y_true = true_gen(x_linspace)

# Visualize observations and true curve
plt.plot(train, y_train, 'ko', label='Train')
plt.plot(test, y_test, 'ro', label='Test')
plt.plot(x_linspace, y_true, 'b-', linewidth=2, label='True Function')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Data')
plt.show()

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
        plt.title('{} Degree Model on Training Data'.format(degrees))
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
        plt.title('{} Degree Model on Testing Data'.format(degrees))
        plt.show()

    if return_scores:
        return training_error, testing_error, -np.mean(cross_valid)

# Degrees = 1 -> Underfitting
fit_poly(train, y_train, test, y_test, degrees=1, plot='train')
fit_poly(train, y_train, test, y_test, degrees=1, plot='test')

# Degrees = 25 -> Overfitting
fit_poly(train, y_train, test, y_test, plot='train', degrees=25)
fit_poly(train, y_train, test, y_test, degrees=25, plot='test')

# Degrees = 5 -> Balanced Model
fit_poly(train, y_train, test, y_test, plot='train', degrees=5)
fit_poly(train, y_train, test, y_test, degrees=5, plot='test')

# Collect training errors and test errors at different values of degrees
degrees = [int(x) for x in np.linspace(1, 40, 40)]
results = pd.DataFrame(0, columns=['train_error', 'test_error'], index=degrees)

for degree in degrees:
    degree_results = fit_poly(train, y_train, test, y_test, degree, plot=False, return_scores=True)
    results.loc[degree, 'train_error'] = degree_results[0]
    results.loc[degree, 'test_error'] = degree_results[1]

# Plot train errors and test errors w.r.t degrees to check when overfitting occurs
plt.plot(results.index, results['train_error'], 'b-o', ms=6, label='Training Error')
plt.plot(results.index, results['test_error'], 'r-*', ms=6, label='Testing Error')
plt.legend(loc=2)
plt.xlabel('Degrees')
plt.ylabel('Mean Squared Error')
plt.title('Training and Testing Curves')
plt.ylim(0, 0.05)
plt.show()

print('\nMinimum Training Error occurs at {} degrees.'.format(int(np.argmin(results['train_error']))))
print('Minimum Testing Error occurs at {} degrees.\n'.format(int(np.argmin(results['test_error']))))
