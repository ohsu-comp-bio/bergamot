from sklearn.svm import LinearSVC
import json
import numpy

def _arrange_vector(data_X, d):
    """Private function. Arranges coefficients to match the data.
    
    Input:
      - data_X: an array of strings
      - d: a dictionary of {string: coefficient}

    Returns:
      - arranged_coef: an array of coefficients in the order of data_X
    """
    arranged_coef = []
    for x_var in data_X:
        coef = 0
        if x_var in d:
            coef = d[x_var]
        # If the coefficient doesn't exist in the dictionary, it is
        # set to zero.
        arranged_coef.append(coef)

    return arranged_coef


def from_json(self, data_X, json_file):
    """Returns a linearSVC regression model populated with the information
    from a json file.
    
    Input:
      - data_X: an array of strings
      - json_file: a json file with 'intercept' and 'coeff'

    Returns:
      - a scikit-learn model
    """
    model = self()
    
    with open(json_file, 'r') as fh:
        input_json = json.load(fh)

    intercept = 0
    coefs_dict = {}
    
    if input_json['intercept']:
        intercept = input_json['intercept']

    # 'coeff' saved as an array of dictionaries.    
    input_coeff = input_json['coeff']

    for i in range(len(input_coeff)):
        coefficient = input_coeff[i]['coeff']
        feature = input_coeff[i]['feature']
        coefs_dict[feature] = coefficient

    # scikit-learn linear SVC models store coefficients in an
    # array of arrays.
    # TODO: confirm this.
    arranged_coefs = [_arrange_vector(data_X, coefs_dict)]

    classes = [1.0, 0.0]
    
    model.intercept_ = float(intercept)
    model.coef_ = numpy.array(arranged_coefs)
    model.classes_ = numpy.array(classes)
    
    return model


LinearSVC.from_json = classmethod(from_json)
