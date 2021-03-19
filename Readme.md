# **Gra**dient **S**ymbolic **Com**putation (grascom)

_a cool description here_

The main inspiration for this project comes from the MATLAB program LDNet (v. 1.5). The actual version is however not a simple translation in Python. Main differences of grascom compared with LDNet:

- It uses the Pytorch library to deal with high-dimensional tensors
- It lacks the sequential mode
- It's modified (in part heavily) to deal with allomorphy in Sanskrit, whereas LDNet dealt with Tree Structures and Twister simulations.

---

**disclaimer** : The project is a work in progress and there are for sure many parts that could be improved or present some bugs. Feel free to fork and use this project and open an issue if you find bugs or have some suggestions. Any help would be greatly appreciated.

## Structure

---

- Roles: Since this network learns Allomorphy and thus process strings, the roles correspond to the positional roles of each sement in a string. At moment the grammar accepts 5 roles (for tokens up to 5 letters long). But this can easily be updated using the `read_from_Input` method in the `Bindings` class.

- `data/inp.csv` : contains the training data. The first col (id) separates different inputs, same index = same word. The other columns give activation values for each filler in the word for each role in the grammar. The csv is read into the grammar using the `pandas` library.

## Requirements

see `requirement.txt`

## References

---

- Smolensky, Cho & Mathis - [Optimization and Quantization in Gradient Symbolic Systems](https://onlinelibrary.wiley.com/doi/pdf/10.1111/cogs.12047) [SEMINAL PAPER]

### ToDO

---

Have a look at the issues tab.

3. Move the tensors to CUDA to improve speed.

4. Animation axes are not implemented here. MATLAB functions for this:

   setupAnimationAxes (< SetupLDNet)
   settings.animation_axes
   initAnimationForRun(settings) (< rundLDNet on Stimulus)

## Next Project

I've just started an attempt to implement the same GSC as R package. By now the project proceeds slowly, mostly due to my unsufficient knowledges of the details of OOP in R. If anyone wants to contribute, just mail me. I would be glad to share this task!
