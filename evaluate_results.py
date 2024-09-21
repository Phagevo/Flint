from eval.docking import docking
from eval.prepare import prepare
from eval.window import compute_box
import os
import csv
from eval.get_mutations import get_num_mutations


def get_energy(receptor_path, ligand_path):
  #receptor_name = "0_whole"
  #ligand_name = "0"

  # compute the sizes of the docking "box" / window
  docking_box = compute_box(receptor_path, ligand_path, padding=10)
  #print(docking_box["center"])

  # prepare source files to PDBQT files
  receptor_path_preped = prepare(receptor_path)
  ligand_path_preped = prepare(ligand_path)
  try:
    dic = docking(receptor_path_preped,
                     ligand_path_preped,
                     center=docking_box["center"],
                     box_size=docking_box["size"],
                     n_dockings=40,
                     n_poses=20)
    energy, dG =  dic["Kd"], dic["dG"]
    print("Kd:", energy)
    return energy, dG
  except Exception as e:
    print(f"[!]Error: {e}.")
    print("------")
    return -1


output_path = "./summary.tsv"
res_path = "./results/mutants/"

DATA = [["ID","SCORE" ,"KD", "NUM_MUTATIONS"]]

for mol in os.listdir(res_path):
  if mol.endswith(".sdf"):
    print("-------------------------")
    # We have a molecule that ends with sdf, which means that his receptors are also here
    prefix = mol[0:-4]
    for rec in os.listdir(res_path):
      if rec.startswith(prefix) and rec.endswith(".pdb"):
        # We have a receptor that starts with the prefix of the molecule
        print("Doing:", rec, "with ligand:", mol)
        energy,dG = get_energy(os.path.join(res_path, rec),
                            os.path.join(res_path, mol))
        
        numbers = "".join([s for s in rec.split() if s.isdigit()])
        print("Numbers:", numbers)
        num_mutations = get_num_mutations(
            os.path.join(res_path, rec),
            os.path.join(res_path, numbers + "_whole.pdb"))
        DATA.append([rec, dG,energy, num_mutations])
        if int(numbers) >= 3:
          break

# Write the list of lists into the TSV file
with open(output_path, 'w', newline='') as file:
  writer = csv.writer(file, delimiter='\t')
  writer.writerows(DATA)