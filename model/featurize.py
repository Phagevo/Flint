from pocketgen.utils.transforms import FeaturizeProteinAtom, FeaturizeLigandAtom
from torch_geometric.transforms import Compose
import torch

def densify(data:dict) -> torch.Tensor:
  """
  Transforms a set of human-level features to a dense data tensor.
  @param data: a feature-dict returned by featurize()
  @return: a dense-data torch tensor representing features.
  """

  return Compose([
    FeaturizeProteinAtom(), # issue : star import
    FeaturizeLigandAtom(), # issue : star import
  ])(data)


def featurize(
  protein_dict={}, 
  ligand_dict={}, 
  residue_dict={}, 
  seq=None, 
  full_seq_idx=None,
  r10_idx=None) -> dict:

  """
  Transforms a 3-uplet of molecule dicts into a features 
  dict that is interpretable by the densify function.
  @param protein_dict: #################
  @param ligand_dict: #################
  @param residue_dict: #################
  @param seq: #################
  @param full_seq_idx: #################
  @param r10_idx: #################
  @return: #################
  """

  # concatenate the first 3 dicts (prot, lig and residue)
  features = dict({f"p_{k}":v for k,v in protein_dict.items()}, 
    **{f"l_{k}":v for k,v in ligand_dict.items()})
  features.update(residue_dict)

  # add keys for simple variables
  features.update({
    'full_seq_idx': full_seq_idx,
    'r10_idx': r10_idx,
    'seq': seq
  })

  return features