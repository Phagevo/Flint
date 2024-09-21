from Bio import PDB


def get_sequence(structure):
  sequence = []
  for model in structure:
    for chain in model:
      for residue in chain:
        if PDB.is_aa(residue):
          sequence.append(residue.resname)
  return sequence


def get_num_mutations(protein1_path, protein2_path):
  parser = PDB.PDBParser()
  structure1 = parser.get_structure('protein1', protein1_path)
  structure2 = parser.get_structure('protein2', protein2_path)
  seq1 = get_sequence(structure1)
  seq2 = get_sequence(structure2)
  mutations = []
  for i, (res1, res2) in enumerate(zip(seq1, seq2)):
    if res1 != res2:
      mutations.append((i + 1, res1, res2))

  print(f"Number of mutations: {len(mutations)}")
  print("Mutations at positions:", mutations)
  return len(mutations)
