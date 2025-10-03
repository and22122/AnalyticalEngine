from sklearn.datasets import load_diabetes

def formatBunch(dataBunch) :
    """Takes a data bunch and formats it for linear regression."""
    X, y = dataBunch(return_X_y=True)
    return str(X) + '\n' + str(y)

print(formatBunch(load_diabetes))