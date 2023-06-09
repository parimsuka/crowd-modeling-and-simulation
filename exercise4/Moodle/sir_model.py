
def mu(b, I, mu0, mu1):
    """
    Recovery rate.
    
    Calculate the recovery rate based on the number of available beds and the number of infective persons.
    
    Parameters:
    -----------
    b : float
        Number of hospital beds per 10,000 persons.
    I : float
        Number of infective persons.
    mu0 : float
        Minimum recovery rate.
    mu1 : float
        Maximum recovery rate.

    Returns:
    --------
    mu : float
        Recovery rate.
    """
    # recovery rate, depends on mu0, mu1, b
    mu = mu0 + (mu1 - mu0) * (b/(I+b))
    return mu

def R0(beta, d, nu, mu1):
    """
    Basic reproduction number.
    
    Calculate the basic reproduction number of the SIR model.
    
    Parameters:
    -----------
    beta : float
        Average number of adequate contacts per unit time with infectious individuals.
    d : float
        Natural death rate.
    nu : float
        Disease-induced death rate.
    mu1 : float
        Maximum recovery rate.

    Returns:
    --------
    R0 : float
        Basic reproduction number.
    """
    return beta / (d + nu + mu1)

def h(I, mu0, mu1, beta, A, d, nu, b):
    """
    Indicator function for bifurcations.
    
    Calculate the indicator function value for bifurcations based on the number of infective persons.

    Parameters:
    -----------
    I : float
        Number of infective persons.
    mu0 : float
        Minimum recovery rate.
    mu1 : float
        Maximum recovery rate.
    beta : float
        Average number of adequate contacts per unit time with infectious individuals.
    A : float
        Recruitment rate of susceptibles (e.g. birth rate).
    d : float
        Natural death rate.
    nu : float
        Disease-induced death rate.
    b : float
        Hospital beds per 10,000 persons.

    Returns:
    --------
    res : float
        Indicator function value for bifurcations.
    """
    c0 = b**2 * d * A
    c1 = b * ((mu0-mu1+2*d) * A + (beta-nu)*b*d)
    c2 = (mu1-mu0)*b*nu + 2*b*d*(beta-nu)+d*A
    c3 = d*(beta-nu)
    res = c0 + c1 * I + c2 * I**2 + c3 * I**3
    return res
    

def model(t, y, mu0, mu1, beta, A, d, nu, b):
    """
    SIR model including hospitalization and natural death.
    
    Calculate the derivatives of the SIR model variables: susceptible (S), infective (I), and removed (R),
    with respect to time.

    Parameters:
    -----------
    t : float
        Current time.
    y : list
        List of current values of the variables [S, I, R].
    mu0 : float
        Minimum recovery rate.
    mu1 : float
        Maximum recovery rate.
    beta : float
        Average number of adequate contacts per unit time with infectious individuals.
    A : float
        Recruitment rate of susceptibles (e.g. birth rate).
    d : float
        Natural death rate.
    nu : float
        Disease-induced death rate.
    b : float
        Hospital beds per 10,000 persons.

    Returns:
    --------
    derivatives : list
        List of derivatives [dSdt, dIdt, dRdt] of the variables S, I, and R, respectively.
    """
    S, I, R = y
    m = mu(b, I, mu0, mu1)
    
    beta_SIR = (beta * S * I) / (S + I + R)
    
    dSdt = A - d * S - beta_SIR
    dIdt = -(d + nu) * I - m * I + beta_SIR
    dRdt = m * I - d * R
    
    return [dSdt, dIdt, dRdt]
