import os
from vina import Vina
from eval.chemutils import kd
import time

def docking(
  receptor_file:str, 
  ligand_file:str, 
  center:"tuple[float, float, float]"=(0,0,0), 
  box_size:"tuple[float, float, float]"=(20,20,20),
  n_dockings:int=64, 
  n_poses:int=32,
  score=False,
  write=False) -> "list[float] | float":

  """
  Docking simulation function : returns ...
  @param receptor_file: protein (pdbqt file)
  @param ligand_file: ligand (pdbqt file)
  @param center: docking window center
  @param box_size: docking window size
  @param n_dockings: number of docking simulations
  @param n_poses: number of pose attempts per simulation
  @param score (bool): wether or not the output should be a single score
  @param write (bool): wether or not the poses should be saved to a file
  @return (list[float] | float): a list of scores or a single score
  """

  receptor_name = os.path.splitext(receptor_file)[-1].split('.')[0]
  ligand_name = os.path.splitext(ligand_file)[-1].split('.')[0]

  #On initialise vina
  v = Vina(sf_name='vina', verbosity=1)
  v.set_receptor(receptor_file)
  v.set_ligand_from_file(ligand_file)

  #On pose la box de docking
  v.compute_vina_maps(center=center,box_size=box_size)

  # Score the current pose
  energy = v.score()
  print('Score before minimization: %.3f (kcal/mol)' % energy[0])

  # Minimized locally the current pose
  energy_minimized = v.optimize()
  print('Score after minimization : %.3f (kcal/mol)' % energy_minimized[0])
  # v.write_pose(f'{ligand_name}_minimized.pdbqt', overwrite=True)

  # Dock the ligand
  v.dock(exhaustiveness=n_dockings, n_poses=20)

  if write:
    v.write_poses(
      f'results/docked/{ligand_name}_docked_{time.time()}.pdbqt', 
      n_poses=n_poses)

  output = None

  # single hi-score or the list of all poses energies
  if not score:
    results = v.energies(n_poses=n_poses)
    output = [energies[0] for energies in results]
  else:
    output = v.energies(n_poses=1)[0][0]

  return output
