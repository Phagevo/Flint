import math

def kd(delta_G:float, temperature:float=298.0) -> float:
  """
  calculates the affinity constant depending on the binding free energy
  @param delta_G (float): the value of the binding free energy (kcal/mol)
  @param temperature (float): temperature (kelvin)
  @return (float): affinity constant
  """
  
  R = 0.001987 # gaz constant
  return math.exp((-delta_G) / (R * temperature))