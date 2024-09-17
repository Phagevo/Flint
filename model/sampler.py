import torch
from .featurize import densify, featurize
from pocketgen.utils.protein_ligand import PDBProtein, parse_sdf_file
from pocketgen.utils.data import torchify_dict

def interaction(ligand_path: str, receptor_path: str) -> torch.Tensor:
  """
  Convert PDB and SDF files into a set of protein-ligand interaction features.
  @param ligand_path (str): path to the ligand SDF file.
  @param receptor_path (str): path to the receptor PDB file.
  @return (torch.Tensor): a data-dense feature tensor representing the interaction.
  """

  # Read and parse the mol (pdb / sdf) files
  pdb_block = open(receptor_path, 'r').read()
  protein = PDBProtein(pdb_block)
  ligand_dict = parse_sdf_file(ligand_path, feat=False)

  # select only the residues inside a radius around the ligand
  r10_index, r10_residues = protein.query_residues_ligand(ligand_dict, radius=10, selected_residue=None, return_mask=False)
  full_seq_index, full_seq_residues = protein.query_residues_ligand(ligand_dict, radius=3.5, selected_residue=r10_residues, return_mask=False)

  # define pocket from the (r < 10) residues
  pocket = PDBProtein(protein.residues_to_pdb_block(r10_residues))
  pocket_dict = pocket.to_dict_atom()
  residue_dict = pocket.to_dict_residue()

  # define the scope of protein_edit_residue (sould be of type torch.Tensor[bool])
  _, residue_dict['protein_edit_residue'] = pocket.query_residues_ligand(ligand_dict)

  full_seq_index.sort()
  r10_index.sort()

  # defensive assertions
  assert len(r10_index) == len(r10_residues)
  assert residue_dict['protein_edit_residue'].sum() == len(full_seq_index)
  assert len(residue_dict['protein_edit_residue']) == len(r10_index)

  data = featurize(
    protein_dict=torchify_dict(pocket_dict),
    ligand_dict=torchify_dict(ligand_dict),
    residue_dict=torchify_dict(residue_dict),
    seq=''.join(residue_dict['seq']),
    full_seq_index=full_seq_index,
    r10_index=r10_index
  )

  # Add metadata
  data.update({
    'protein_filename': receptor_path,
    'ligand_filename': ligand_path
  })

  return densify(data)