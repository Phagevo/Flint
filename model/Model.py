import esm
import torch
from pocketgen.models.PD import Pocket_Design_new
from pocketgen.utils.misc import seed_all, load_config
from pocketgen.utils.transforms import FeaturizeProteinAtom, FeaturizeLigandAtom

class Model:
  def __init__(self, checkpoint_path:str, verbose:int=1) -> "Model":
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

    if self.verbose > 0:
      print('__PJNAME__ setup started, please wait.')
    if self.verbose == 2:
      print('Now initializing pytorch and CUDA environment :')

    # clean cache and setting the libs seeds
    torch.cuda.empty_cache()
    seed_all(2089)
    self.device = torch.device('cpu') # for GPU : "cuda:0"

    if self.verbose == 2:
      print('\ttorch and CUDA initialized correctly.\nNow retrieving alphabet from fair-ESM :')

    # set ESM2 alphabet as the usual alphabet
    _, self.alphabet = esm.pretrained.load_model_and_alphabet_hub('esm2_t33_650M_UR50D')
    del _ # ESM2 pretrained_model that we don't need here is deleted from memory

    if self.verbose == 2:
      print('\tESM alphabet successfully loaded.\nNow building PocketGen model :')

    # set the model and load the checkpoint from .pt file
    self.checkpoint = torch.load(checkpoint_path, map_location=self.device)

    if self.verbose == 2:
      print('\tcheckpoint successfully created.')

    self.model = Pocket_Design_new(
      load_config('./pocketgen/configs/train_model.yml').model,
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
      print('\tcheckpoint loaded into PocketGen.\nEnd of setup, model can now be used.\n\n\n')

    return self
  

  def input(self, receptor_path, ligand_path):
    pass


  def generate(self):
    pass

  
  def results(self):
    pass