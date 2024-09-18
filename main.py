from model.Model import Model
import argparse
import torch

# if called from command line
if __name__ == "__main__":
  torch.set_warn_always(False)
  parser = argparse.ArgumentParser()

  parser.add_argument("-d", "--device", type=str, default="cuda:0", help="Set the device (cpu or cuda:0)")
  parser.add_argument("-o", "--output", type=str, default="./results/mutants", help="Set the path for the output directory (defaults to ./results/mutants)")
  parser.add_argument("-v", "--verbose", type=int, choices=[0, 1, 2], default=1, help="Set the verbosity between 0 and 2")
  parser.add_argument("--receptor", type=str, required=True, help="Set the receptor filepath")
  parser.add_argument("--ligand", type=str, required=True, help="Set the ligand filepath")

  # parse arguments
  args = parser.parse_args()
  
  # instantiates the model with args
  tmpname = Model("./checkpoints/checkpoint.pt", {
    "device": args.device,
    "output": args.output,
    "verbose": args.verbose
  })
  
  # pass molecule files to the model 
  tmpname.input(args.receptor, args.ligand)

  # begin the inference / generate mutants
  tmpname.generate()

  # should then log the results and write the summary file and PDBs => => Model.results()