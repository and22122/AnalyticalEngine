from sklearn.utils import Bunch

def formatBunch(dataBunch) :
    """Takes a data bunch and formats it for linear regression."""
    X, y = dataBunch(return_X_y=True)
    return str(X) + '\n' + str(y)
