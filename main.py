from model.Model import Model
import argparse

# if called from command line
if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument("-d", "--device", type=str, default="cuda:0", help="Set the device (cpu or cuda:0)")
  parser.add_argument("-o", "--output", type=str, default="./results/mutants", help="Set the path for the output directory (defaults to ./results/mutants)")
  parser.add_argument("-v", "--verbose", type=int, choices=[0, 1, 2], default=1, help="Set the verbosity between 0 and 2")
  parser.add_argument("--receptor", type=str, required=True, help="Set the receptor filepath")
  parser.add_argument("--ligand", type=str, required=True, help="Set the ligand filepath")

  # parse arguments
  args = parser.parse_args()
  
  # instantiates the model with args
  tmpname = Model("./checkpoints/esm2_t33_650M_UR50D.pt", {
    "device": args.device,
    "output": args.output,
    "verbose": args.verbose
  })
  
  # pass mol files to the model 
  tmpname.input(args.receptor, args.ligand)

# should then compute mutants one by one => Model.generate() * n
# should then log the results and write the summary file and PDBs => => Model.results()