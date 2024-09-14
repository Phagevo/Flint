import math

def kd(delta_G:float, temperature:float=298) -> float:
  """
  calculates the affinity constant depending on the binding free energy
  @param delta_G: the value of the binding free energy
  @param temperature: temperature, using Kelvin degrees
  @return: affinity constant (alias Kd)
  """
  
  return math.exp(delta_G / (8.314 * temperature))