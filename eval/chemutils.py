import math

def kcalmol_to_jmol(x:float):
  """
  converts an energy from kcal/mol to J/mol
  @x (float): energy value to convert (kcal/mol)
  @return (float): converted energy value (J/mol)
  """

  return x * 4184.0


def kd(delta_G:float, temperature:float=298.0) -> float:
  """
  calculates the affinity constant depending on the binding free energy
  @param delta_G (float): the value of the binding free energy (kcal/mol)
  @param temperature (float): temperature (kelvin)
  @return (float): affinity constant
  """
  
  return math.exp((-1 * kcalmol_to_jmol(delta_G)) / (8.314 * temperature))