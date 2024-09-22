# 🚀 Flint : AI-powered protein receptor mutation

A computational tool designed to generate protein receptor mutants that allows enhanced binding specificity towards a target ligand, based on [PocketGen]. While generating a set of mutated receptor structures, Flint evaluates their affinity with the ligand using [AutoDock Vina]. By default, our custom `__MODELNAME__` pre-trained model checkpoint is used for the generation. The key point is that the docking simulation was embedded as the scoring function during the learning transfer, making it the target of the gradient descent (see [Model](https://2024.igem.wiki/evry-paris-saclay/model) article on the wiki).

## Outputs and expected results

If executed correctly, Flint will generate a set of unique mutated receptor proteins in PDB format, designed to maximize the binding affinity for the ligand ; and a summary file containing, for each receptor :
   - The corresponding docking score, affinity constant and number of mutations compared to the original receptor.
   - Additional information about the receptor-ligand interaction (e.g. ligand position, residues involved).
   - The sequence of mutations that leads to its creation, starting from the original receptor.

## Getting started with Flint
Make sure your machine meets all [requirements](https://github.com/Phagevo/Flint/wiki/Requirements) and has [git](https://git-scm.com/) and [anaconda](https://www.anaconda.com/) installed.
```bash
git clone https://github.com/Phagevo/Flint.git
cd Flint
git clone https://github.com/Phagevo/PocketGen.git
```
Install the environment and dependencies using the conda config file :
```bash
conda env create -f env.yaml
conda activate flint
```
If you intend to build environment without conda, keep in mind that installing [AutoDock Vina] from `pip` or any other package manager is deprecated. Besides, to run the project from Windows, follow [this](https://github.com/Phagevo/Flint/wiki/Running-Flint-in-Windows) tutorial.
```yaml
├── PocketGen # can be cloned from PocketGen repository
├── checkpoints
│   ├── __MODELNAME__.pt # flint custom & fine-tuned model
│   └── pocketgen.pt # the pocketgen pre-trained model
│
├── eval
├── model
└── main.py
```
This (above) is what should ressemble your working directory after installing Flint. The `checkpoints` folder should be created manually and is necessary for the program to run, and you can find the [PocketGen] checkpoint file on the official repository.

## Usage from command line
```bash
python main.py --receptor <receptor.pdb> --ligand <ligand.sdf>
```
- `<receptor.pdb>`: Path to the input protein receptor file in PDB format.
- `<ligand.sdf>`: Path to the input ligand file in SDF format.

Additional parameters can be found by running `python main.py --help`.

[AutoDock Vina]: https://github.com/ccsb-scripps/AutoDock-Vina
[PocketGen]: https://github.com/zaixizhang/PocketGen
