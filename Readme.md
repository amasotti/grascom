# grascom

Playground **Gra**dient **S**ymbolic **Com**putation (grascom)

---

This project is a Python3 implementation of the GSC (Smolensky & Legendre 2016, Cho, Goldrick & Smolensky 2014,2020), a psychogically plausible, _continuous-state, continuous-time stochastic dynamical-system model of input processing_.
The main interest for GSC comes from the fact that it makes possible to join classical symbolic cognitive theory with neural computation.

The main inspiration for the code comes from the MATLAB program LDNet (v. 1.5), of which this repo is a python translation.
Main differences of grascom compared with LDNet:

- It uses the Pytorch library to deal with high-dimensional tensors
- It lacks the sequential mode and 3d Animation (for now)
- Plotting using [seaborn](https://seaborn.pydata.org)

LDNet was originally developed to model phenomena such as voicing neutralization or twister production.
The idea here is to use the GSC to model linguistic phenomena such as allomorphy or sandhi (external and internal), where we could assume that much of chaotic surface forms are just the consequence of blends of multiple underlying representations, each partially activated.
See the Jupyter-notebook for a concrete instantiation.
At moment very simplistic positional roles were used. I'm thinking about implementing some kind of [_span_ (or _brick_) _roles_](https://doi.org/10.31234/osf.io/utcgv).

- `TPR_example` : How to represent symbolic structures using the Tensor Product Representation
- `example` : GSC run on a toy-example (Sanskrit allomorphy: _bhud_ vs _budh_ aka Grassmann's Law)


Optimal weights (for constraints) and activation values (for inputs) can be found using either mathematical Optimization or a simple learning algorithm (as suggested in E. Rosen (2019)). See the folder `Optimization` for a simple implementation.

---

**disclaimer** : The project is a work in progress in a early phase and was created with the main goal to practice with the GSC framework and develop an intuition, how the GSC works.
There are for sure many parts that could be improved or present some bugs. Feel free to use this project, open an issue or make suggestions. Any help would be greatly appreciated.

## Structure

```
grascom
│   LICENSE
│   main.py
│   Readme.md
│   requirements.txt
│
├───data
│   │   full_traces.mat
│   │   full_traces.pt
│   │   initialized_mats.mat
│   │   inp2.csv
│   │   inp_pandas.csv
│   │   params.mat
│   │   params.pt
│   │   stimuli_summary.txt
│   │   summary.txt
│   │
│   └───plots
│           harmonized_input.png
│           Input_gradient_activation.png
│           plotsFrequency_ep1_stimulus_0.png
│           plotsHarmonies_ep1_stimulus_0.png
│           random_activation.png
├───docs
│       LDnet1.5 Manual.pdf
│       Readme.md
│
├───examples
│       example.ipynb
│       tableaux.xlsx
│       TPR_example.ipynb
│
├───Optimization
│       optimizeWA..pdf
│       optimizeWA.jl
│
└───src
    │
    ├───classes
    │      bindings.py
    │      Bowl.py
    │      Grammar.py
    │      utilFunc.py
    └───gsc
         gsc_network.py
         plotting.py


```

---

- `data/` : This folder contains the external inputs, logs and summaries.
  - `data/inp.csv` : contains the training data. The first col (id) separates different inputs, same index = same word. The other columns give the activation values for each filler in the word for each role in the grammar. The csv is read into the grammar using the `pandas` library.
  - `*.mat`, `*.pt` : backup files in Matlab and Pytorch format.
- `src/` :
  - `classes/` : A collection of Python classes, to process single components of the GSC model. These include the following classes: Roles, Fillers, Bindings, HarmonicGrammar, Bowl.
  - `utilfunc.py` : A couple of general purpose auxiliary functions.
  - `gsc/`: The folder contains the main class of the project: the GSCNet and the class Plot for plotting the traces
  - `main.py`: the entry point for simulations. Here the user can declare a set of fillers and roles, (optionally) together with filler symmetries
    and Harmonic Constraints. These variables are used to initialize and run the network.
  - Notebooks : Example and visualizations.

## Requirements

see `requirement.txt`

**N.B**: Use the last stable version of PyTorch (1.8.0) since only this version supports some advanced linalg operations (eigenvalues computation, Kronecker product etc..)

## References

- Cho, Mathis & Smolensky (2014) - [Optimization and Quantization in Gradient Symbolic Systems](https://onlinelibrary.wiley.com/doi/pdf/10.1111/cogs.12047)
- Cho, Goldrick & Smolensky (2017) - [Incremental parsing in a continuous dynamical system](https://faculty.wcas.northwestern.edu/matt-goldrick/publications/pdfs/GSC_LV_final3.pdf)
- Cho, Goldrick & Smolensky (2020) .[Parallel parsing in a Gradient Symbolic Computation parser](https://psyarxiv.com/utcgv/download/?format=pdf)

### ToDO

<em>See under "Issues".</em>

- <del>Create a JupyterNotebook (better for visualizing)</del>
- Implement Span Roles/brick roles to represent recursive structure?
- Implement quantization policies (different bowl_strengths for different timesteps)
- Start documentation
- Improve plots, add other plotting functions
- Test more complex input data
- Move the tensors to CUDA to improve speed.
- 3d animation / Visualize Quantization
- Check clamping techniques (constant value? Projection matrix? input activations?)

## Next Project(s)

- GSC-learning implementation
  - Possible starting points:
    - [GSR Learning](https://github.com/clairemoorecantwell/GSR_Learning) by C. Cantwell

# Interesting resources

- [RNN with TPR](https://github.com/tommccoy1/tpdn)
- [Universal linguistic inductive biases](https://github.com/tommccoy1/meta-learning-linguistic-biases)
- [TPR units](https://github.com/shuaitang/TPRU)
- [TPR modelling for polysynthetic languages](https://github.com/neural-polysynthetic-language-modelling/tpr)
- [RNN and LSTM Optimization](https://github.com/neural-polysynthetic-language-modelling/awd-lstm-lm)
- [TPR caption](https://github.com/ggeorgea/TPRcaption)

* Goldrick's works:
  - [Verb-particle replication](https://osf.io/6v3r9/)
  - [Phonetic variation as symptom of phonological planning scope](https://osf.io/uge8x/)
