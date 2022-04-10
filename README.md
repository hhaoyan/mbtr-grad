## Gradient domain many-body tensor representation (MBTR)

This code extends the many-body tensor representation (MBTR) to include derivatives so that one could perform gradient-domain machine-learning.

### sGDML

We demonstrate its ability using the MD17 (http://quantum-machine.org/gdml/) datasets to train symmetric gradient domain machine learning (sGDML) models.

#### Prepare datasets

Download MD17 dataset:

```bash
wget -nc http://quantum-machine.org/gdml/data/npz/benzene2017_dft.npz -O notebooks/datasets/md17/benzene2017_dft.npz
wget -nc http://quantum-machine.org/gdml/data/npz/uracil_dft.npz -O notebooks/datasets/md17/uracil_dft.npz
wget -nc http://quantum-machine.org/gdml/data/npz/naphthalene_dft.npz -O notebooks/datasets/md17/naphthalene_dft.npz
wget -nc http://quantum-machine.org/gdml/data/npz/aspirin_dft.npz -O notebooks/datasets/md17/aspirin_dft.npz
wget -nc http://quantum-machine.org/gdml/data/npz/salicylic_dft.npz -O notebooks/datasets/md17/salicylic_dft.npz
wget -nc http://quantum-machine.org/gdml/data/npz/malonaldehyde_dft.npz -O notebooks/datasets/md17/malonaldehyde_dft.npz
wget -nc http://quantum-machine.org/gdml/data/npz/ethanol_dft.npz -O notebooks/datasets/md17/ethanol_dft.npz
wget -nc http://quantum-machine.org/gdml/data/npz/toluene_dft.npz -O notebooks/datasets/md17/toluene_dft.npz
```

#### Running CM experiments

We also include the code for training sGDML models using the Coulomb matrix (CM) representation.

```bash
./script_sgdml_cm_lr.py --output ./tmp/sgdml/cm
```

### Running MBTR experiments

```bash
./script_sgdml_mbtr2_lr.py --use_gpu --output ./tmp/sgdml/cm
```

### Precomputed results data

```bash
TODO
```
