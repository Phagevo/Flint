from PocketGen.utils.transforms import FeaturizeProteinAtom, FeaturizeLigandAtom
from torch_geometric.transforms import Compose
import torch

def densify(features:dict) -> torch.Tensor:
  """
  Transforms a set of human-level features to a dense data tensor.
  @param data (dict): a feature-dict returned by featurize()
  @return (torch.Tensor): a dense-data torch tensor representing features.
  """

  return Compose([
    FeaturizeProteinAtom(),
    FeaturizeLigandAtom(),
  ])(features)


def featurize(
  protein_dict={}, 
  ligand_dict={}, 
  residue_dict={}, 
  seq=None, 
  full_seq_index=None,
  r10_index=None) -> dict:

  """
  Transforms molecule interaction data into a feature 
  dict that is interpretable by the densify function.
  @param protein_dict (dict): a dictionary representation of the receptor
  @param ligand_dict (dict): a dictionary representation of the ligand
  @param residue_dict (dict): a dictionary representation of the residue
  @param seq (str): #################
  @param full_seq_index (torch.Tensor): #################
  @param r10_index (torch.Tensor): indexes of the residues (r < 10 around ligand)
  @return (dict): a feature dictionnary
  """

  # concatenates the first 3 dicts (prot, lig and residue)
  features = dict({f"protein_{k}":v for k,v in protein_dict.items()}, 
    **{f"ligand_{k}":v for k,v in ligand_dict.items()})
  features.update(residue_dict)

  # adds keys for simple variables
  features.update({
    'full_seq_idx': full_seq_index,
    'r10_idx': r10_index,
    'seq': seq
  })

  return features