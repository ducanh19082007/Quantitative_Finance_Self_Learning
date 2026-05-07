import math

def future_value_discrete(PV : float, r: float, n: float):
    return PV*(1 + r)**n

def present_value_discrete(FV: float, r: float, n: float):
    return FV / (1 + r)**n

def future_value_continuous(PV: float, r: float, n: float):
    return PV * math.exp(r*n)

def present_value_continous(FV: float, r: float, n: float):
    return FV * math.exp( -r * n )

