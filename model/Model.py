import esm
import torch
from torch.utils.data import DataLoader

from pocketgen.models.PD import Pocket_Design_new
from pocketgen.utils.misc import seed_all, load_config
from pocketgen.utils.transforms import FeaturizeProteinAtom, FeaturizeLigandAtom
from pocketgen.utils.data import collate_mols_block
from functools import partial

from .sampler import interaction

class Model:
  def __init__(self, checkpoint_path:str, verbose:int=1, device="cuda:0") -> "Model":
    """
    The mutant generation model constructor. This method does the setup of 
    torch and CUDA environment, loads the checkpoint and then returns a PocketGen 
    instance using the weights from checkpoints and the parameters retrieved.
    @param checkpoint_path (str): Path to checkpoint (.pt) file for PocketGen.
    @param verbose (int): 0 for quiet, 1 for necessary information and 2 for debug.
    @return (Model): the instance of Model, for chainability purposes.
    """

    # setup global class variables
    self.verbose = verbose
    self.pwd = "./"
    self.output_path = "./results"
    self.mutants = []
    self.config = load_config('./pocketgen/configs/train_model.yml')
    self.device = device

    if self.verbose > 0:
      print('__PJNAME__ setup started, please wait.')
    if self.verbose == 2:
      print('Now initializing pytorch and CUDA environment :')

    # clean cache and setting the libs seeds
    torch.cuda.empty_cache()
    seed_all(2089)

    if self.verbose == 2:
      print('\tpytorch and CUDA initialized correctly.')
      print('Now retrieving alphabet from fair-ESM :')

    # set ESM2 alphabet as the usual alphabet
    pretrained_model, self.alphabet = esm.pretrained.load_model_and_alphabet_hub('esm2_t33_650M_UR50D')
    del pretrained_model # ESM2 pretrained_model that we don't need here is deleted from memory

    if self.verbose == 2:
      print('\tESM alphabet successfully loaded.')
      print('Now building PocketGen model :')

    # set the model and load the checkpoint from .pt file
    self.checkpoint = torch.load(checkpoint_path, map_location=self.device)

    if self.verbose == 2:
      print('\tcheckpoint successfully created.')

    self.model = Pocket_Design_new(
      self.config.model,
      protein_atom_feature_dim=FeaturizeProteinAtom().feature_dim,
      ligand_atom_feature_dim=FeaturizeLigandAtom().feature_dim,
      device=self.device
    )

    if self.verbose == 2:
      print("\tPocketGen model well instanciated.")

    self.model = self.model.to(self.device)

    if self.verbose == 2:
      print('\tPocketGen model sent to selected device.')

    self.model.load_state_dict(self.checkpoint['model'])

    if self.verbose == 2:
      print('\tcheckpoint loaded into PocketGen.\n')
      print('End of setup, model can now be used.\n\n\n')

    return self
  

  def input(self, receptor_path:str, ligand_path:str) -> "Model":
    """
    Loads a protein receptor and a ligand from files and store it in 
    a data-loader, useable by the model when generating mutants.
    @param ligand_path (str): path to the ligand SDF file.
    @param receptor_path (str): path to the receptor PDB file.
    @return (Model): the instance of Model, for chainability purposes.
    """

    if self.verbose == 2:
      print('Now parsing data from receptor and ligand :')
    
    # get dense features from receptor-ligand interaction
    features = interaction(receptor_path, ligand_path)

    if self.verbose == 2:
      print('\tsuccessfully parsed interaction features.\n')
      print('Now building the pytorch dataloader :')

    self.loader = DataLoader(
      [features for _ in range(8)], # 8 * features for batching reasons
      batch_size=4, 
      shuffle=False,
      num_workers=self.config.train.num_workers,
      collate_fn=partial(
        collate_mols_block, 
        batch_converter=self.alphabet.get_batch_converter()
      )
    )

    if self.verbose == 2:
      print('\tpytorch dataloader built correctly.')

  
  def generate(self):
    pass
    
  
  def results(self):
    pass