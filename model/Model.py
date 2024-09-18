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
  def __init__(self, checkpoint_path:str, args):
    """
    The mutant generation model constructor. This method does the setup of 
    torch and CUDA environment, loads the checkpoint and then returns a PocketGen 
    instance using the weights from checkpoints and the parameters retrieved.
    @param checkpoint_path (str): Path to checkpoint (.pt) file for PocketGen.
    @param verbose (int): 0 for quiet, 1 for necessary information and 2 for debug.
    """

    # setup global class variables
    self.verbose = args["verbose"]
    self.device = args["device"]
    self.outputdir = args["output"]
    self.mutants = []
    self.config = load_config('./pocketgen/configs/train_model.yml')
    
    if self.verbose > 0:
      print('__PJNAME__ setup started, please wait.')
    if self.verbose == 2:
      print('Now initializing pytorch and CUDA environment :')

    # cleans cache and sets the libs seeds
    torch.cuda.empty_cache()
    seed_all(2089)

    if self.verbose == 2:
      print('\tpytorch and CUDA initialized correctly.')
      print('Now retrieving alphabet from fair-ESM :')

    # sets ESM2 alphabet as the usual alphabet
    pretrained_model, self.alphabet = esm.pretrained.load_model_and_alphabet_hub('esm2_t33_650M_UR50D')
    del pretrained_model # ESM2 pretrained_model that we don't need here is deleted from memory

    if self.verbose == 2:
      print('\tESM alphabet successfully loaded.')
      print('Now building PocketGen model :')

    # get the model checkpoint from .pt file
    self.checkpoint = torch.load(checkpoint_path, map_location=self.device)

    if self.verbose == 2:
      print('\tcheckpoint successfully created.')

    # instanciate PocketGen model for pocket design
    self.model = Pocket_Design_new(
      self.config.model,
      protein_atom_feature_dim=FeaturizeProteinAtom().feature_dim,
      ligand_atom_feature_dim=FeaturizeLigandAtom().feature_dim,
      device=self.device
    )

    if self.verbose == 2:
      print("\tPocketGen model well instanciated.")

    # send model to selected device
    self.model = self.model.to(self.device)

    if self.verbose == 2:
      print('\tPocketGen model sent to selected device.')

    # load current saved checkpoint into model
    self.model.load_state_dict(self.checkpoint['model'])

    if self.verbose == 2:
      print('\tcheckpoint loaded into PocketGen.')
      print('End of setup, model can now be used.\n\n')
  

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

    # initialize the data loader (including batch converter)
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