# # Playground and testing
# from src.classes.Grammar import Grammar
# from src.gsc.gsc_network import Net
# fillers = ["bh", "b", "u", "d", "dh"]
# roles = ["ons1", "ons2", "nuc", "coda1", "coda2"]

# G = Grammar(fillers, roles)
# # scons = [("bh/r1", -2), ("b/r1", -2), ("d/r3", 1), ("dh/r3", -3)]
# # cons = [("bh/r1", "b/r1", -10), ("d/r3", "dh/r3", -10),
# #         ("bh/r1", "dh/r3", -10)]


# N = Net(G)
from src.classes.utilFunc import fixed_dotProduct_matrix


b = fixed_dotProduct_matrix(2, 3, 0)
print(b)
