from model.Model import Model

receptor_name = "xylS_m1_TPA"
ligand_name = "TPA"

receptor_path = f"data/mutants/raw/{receptor_name}.pdb"
ligand_path = f"data/ligands/raw/{ligand_name}.sdf"

tmpname = Model("./checkpoints/esm2_t33_650M_UR50D.pt", {
  "device": "cpu",
  "outputdir": "./results/mutants",
  "verbose": 2
})

# should then input data => Model.input(receptor_path, ligand_path)
# should then compute mutants one by one => Model.generate() * n
# should then log the results and write the summary file and PDBs => => Model.results()