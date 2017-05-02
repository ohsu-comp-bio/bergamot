import math

def logistic_function(t):
    """Returns the value after passing through logistic function.
    """
    return (1 / (1 + math.exp(-t)))

def dot_product(data, coef, intercept):
    """Returns dot products of the data and coefficients.
    
    Input
    ------------------------
    data: [pandas.DataFrame]
    coef: [array]
    intercept: [float]

    Returns
    ------------------------
    dot_products: [array of floats]
    """
    dot_products = []
        
    for row in range(len(data.index)):
        dot_product = intercept
        
        for i in range(len(data.columns)):
            dot_product += data.iloc[row, i] * coef[i]
        dot_products.append(dot_product)
            
    return dot_products
