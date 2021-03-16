# **Gra**dient **S**ymbolic **Com**putation (grascom)

_a cool description here_

---

**disclaimer** : The project is a work in progress and there are for sure many parts that could be improved or present some bugs. Feel free to fork and use this project and open an issue if you find bugs or have some suggestions. Any help would be greatly appreciated.

## Structure
---

+ Roles: Since this network learns Allomorphy and thus process strings, the roles correspond to the positional roles of each sement in a string. At moment the grammar accepts 5 roles (for tokens up to 5 letters long). But this can easily be updated using the `read_from_Input` method in the `Bindings` class.

+ `data/inp.csv` : contains the training data. The first col (id) separates different inputs, same index = same word. The other columns give activation values for each filler in the word for each role in the grammar. The csv is read into the grammar using the `pandas` library.



## Requirements

see `requirement.txt`

## References

---

- Smolensky, Cho & Mathis - [Optimization and Quantization in Gradient Symbolic Systems](https://onlinelibrary.wiley.com/doi/pdf/10.1111/cogs.12047) [SEMINAL PAPER]

### ToDO

---

1. Issue 1 : Pytorch
   I wonder which parts of this nn can be implemented using directly Pytorch functionalities or modules.
   At the end this is nothing else but a nn that performs symbolic tensor computation. Is there a way to simplify the code?

2. Issue 2 : Lambda
   While training the GSC should balance quantization (transform continuous vectors and blends into discrete units) and optimization (that works on continuous space). This is done by balancing the two processes through lambda-diffusion:
   In pseudo-code the dynamics `D` at any given training step `t` is equal to:

`D(t) = lambda_t * Optimization + (1-lambda_t) * Quantization`

lambda progressively decreases from 1 to 0, starting with pure Optimization and landing at pure Quantization. The important issue here is how we calculate lambda.

The original code by Goldrick et al. uses this formula (in `src/settings_checker.py`):

```matlab
eigMin = min(real(eig(net.Wc)));                 % Find min eigenvalue of c-space weight matrix.
lambda = 1/(1+4*abs(eigMin-domain.q));

```

which I translated in Python as follows:

```python
import numpy as np

min_eigenvalue = np.min(np.real(np.linalg.eigvals(self.Wc)))
l = 1 / (1 + 4*np.abs(min_eigenvalue - self.domain.q))

```

where `self.Wc` is the weight matrix and `self.domain.q` is the bowl-parameter q.

The authors state that this formula

> THIS LOGIC MAY NOT BE CORRECT SINCE THE DRIFT TERM OF THE DIFFUSION PROCESS IS NOT EQUAL TO THE PARTIAL DERIVATIVES IN C-SPACE.

I've left the formula as in the original lacking a better idea, but I would be glad if anyone could help to improve this.

3. Move the tensors to CUDA to improve speed.

#### Next Project

I've just started an attempt to implement the same GSC as R package. By now the project proceeds slowly, mostly due to my unsufficient knowledges of the details of OOP in R. If anyone wants to contribute, just mail me. I would be glad to share this task!
