from Bio import PDB

def get_sequence(structure):
  """
  Loads a protein structure and returns the sequence of amino acids.
  @param structure (str): the protein structure.
  @return (sequence): the aa sequence of the protein.
  """
  sequence = []
  for model in structure:
    for chain in model:
      for residue in chain:
        if PDB.is_aa(residue):
          sequence.append(residue.resname)
  return sequence


def mutations(protein1_path, protein2_path):
  """
  Loads two protein paths and returns the number of mutations between them.
  @param protein1_path (str): the first protein path.
  @param protein2_path (str): the second protein path.
  @return (int): the number of mutations between the two proteins.
  """
  parser = PDB.PDBParser()
  structure1 = parser.get_structure('protein1', protein1_path)
  structure2 = parser.get_structure('protein2', protein2_path)
  seq1 = get_sequence(structure1)
  seq2 = get_sequence(structure2)
  mutations = []
  for i, (res1, res2) in enumerate(zip(seq1, seq2)):
    if res1 != res2:
      mutations.append((i + 1, res1, res2))
  return len(mutations)
