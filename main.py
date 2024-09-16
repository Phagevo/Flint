from eval.docking import docking
from eval.prepare import prepare
from eval.window import compute_box
from eval.chemutils import ligand_pdb

receptor_name = "xylS_m1_TPA"
ligand_name = "TPA"

receptor_path = f"data/mutants/raw/{receptor_name}.pdb"
ligand_path = f"data/ligands/raw/{ligand_name}.sdf"

# compute the sizes of the docking "box" / window
docking_box = compute_box(receptor_path, ligand_path, padding=10)
print(docking_box["center"])

# prepare source files to PDBQT files
receptor_path_preped = prepare(receptor_path)
ligand_path_preped = prepare(ligand_path)

docking(
  receptor_path_preped, 
  ligand_path_preped,
  center=docking_box["center"],
  box_size=docking_box["size"],
  n_dockings=40, 
  n_poses=20
)