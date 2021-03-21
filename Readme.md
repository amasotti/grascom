# grascom

**Gra**dient **S**ymbolic **Com**putation (grascom)

---

This project is a Python3 implementation of the GSC (Smolensky & Legendre 2016, Cho, Goldrick & Smolensky 2020).
The main inspiration for the code comes from the MATLAB program LDNet (v. 1.5). Main differences of grascom compared with LDNet:

- It uses the Pytorch library to deal with high-dimensional tensors
- It lacks the sequential mode and 3d Animation (for now)

The idea is to use the GSC to model linguistic phenomena such as allomorphy or sandhi (external and internal), where we could assume that much of what at the surface appears to be chaotic, is just the product of gradient blends of multiple underlying representations.
See the Jupyter-notebook for a concrete instantiation.

---

**disclaimer** : The project is a work in progress in a early phase.
There are for sure many parts that could be improved or present some bugs. Feel free to fork and use this project and open an issue if you find bugs or have some suggestions.
Any help would be greatly appreciated.

## Structure

```
    grascom
    │   .gitignore
    │   LICENSE
    │   main.py
    │   Readme.md
    │   requirements.txt
    │   setup.py
    │   Structure.md
    │
    ├───data
    │    │   Constraints.py # to be deleted
    │    │   initialized_mats.mat
    │    │   inputs.xlsx
    │    │   inp_pandas.csv
    │    │   summary.txt
    │    │
    │    └─── plots
    │
    └───src
        │   Constraints.py # to be deleted
        │
        ├───classes
        │      bindings.py
        │      Bowl.py
        │      Grammar.py
        │      utilFunc.py
        │
        ├───gsc
        │      gsc_network.py
        │      plotting.py

```

---

- `data/` : This folder contains the external inputs/training data and is used to save logs and summaries.
  - `data/inp.csv` : contains the training data. The first col (id) separates different inputs, same index = same word. The other columns give activation values for each filler in the word for each role in the grammar. The csv is read into the grammar using the `pandas` library.
  - `initialized_mats.mat` : a backup file created after the general setup and before the training routine begins. This file stores in binary MATLAB-like format all the weights, biases and Lotka-Volterra matrices
    initialized after the external inputs have been read. The `.mat` file can be loaded in Python using the [scipy library](https://docs.scipy.org/doc/scipy/reference/io.html).
- `src/` :
  - `classes/` : A collection of Python classes, to process single components of the GSC model. These include the following classes: Roles, Fillers, Bindings, HarmonicGrammar, Bowl.
  - `utilfunc.py` : A couple of general purpose auxiliary functions.
  - `gsc/`: The folder contains the main class of the project: the GSCNet
  - `main.py`: the entry point for simulations. Here the user can declare a set of fillers and roles, (optionally) together with filler symmetries
    and Harmonic Constraints. These variables are used to initialize and run the network.

## Requirements

see `requirement.txt`

**N.B**: The last stable version of PyTorch (1.8.0) is required since only this version supports some advanced algebric operations (eigenvalues computation, Kronecker product etc..)

## References

---

- Smolensky, Cho & Mathis - [Optimization and Quantization in Gradient Symbolic Systems](https://onlinelibrary.wiley.com/doi/pdf/10.1111/cogs.12047) [SEMINAL PAPER]

### ToDO

<em>See under "Issues".</em>

- Create a JupyterNotebook (better for visualizing)
- Start documentation
- Improve plots, add other plotting functions
- Move the tensors to CUDA to improve speed.
- Animation axes are not implemented here. MATLAB functions for this:

## Next Project

I've just started an attempt to implement the same GSC as R package. By now the project proceeds slowly, mostly due to my unsufficient knowledges of the details of OOP in R. If anyone wants to contribute, just mail me. I would be glad to share this task!
