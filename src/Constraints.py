"""
Set of OT constraints 

Test class 
This functions will be replaced by the Harmony constraints.

__date__ : Februar 201
__author__ : A. Masotti
"""
import re
import torch
from collections import Counter


class Constraints(object):
    def __init__(self, inp1, inp2, out1, out2=None):
        self.inp1 = inp1
        self.inp2 = inp2
        self.out1 = out1
        self.out2 = out2
        self.cons_viol = {"no_coda": 0,
                          "dep": 0,
                          "max": 0}

    def nocoda(self):
        violations = 0
        if not re.search('^.*([aeiouywAEIOUYW]|aj|aw)$', self.out1):
            violations += 1
        if self.out2 is not None:
            if not re.search('^.*([aeiouywAEIOUYW]|aj|aw)$', self.out2):
                violations += 1
        return violations

    def dep(self):
        violations = 0
        len_inp = len(self.inp1) + len(self.inp2)
        len_out = len(self.out1) + len(self.out2)
        if len_out > len_inp:
            violations = len_out - len_inp
        return violations

    def maxC(self):
        violations = 0
        len_inp = len(self.inp1) + len(self.inp2)
        len_out = len(self.out1) + len(self.out2)
        if len_inp > len_out:
            violations = len_inp - len_out
        return violations

    def maxDep(self):
        viol_dep = 0
        viol_max = 0
        inp_segments = Counter("".join([self.inp1, self.inp2]))
        out_segments = Counter("".join([self.out1, self.out2]))
        if len(inp_segments) == len(out_segments):
            for seg, count in inp_segments.items():
                if count < out_segments[seg]:
                    viol_dep += 1
                elif count > out_segments[seg]:
                    viol_max += 1
        # TODO: not only the length, but also how many have been added for key not present in one counter
        elif len(inp_segments) < len(out_segments):
            viol_dep += len(out_segments) - len(inp_segments)
        elif len(inp_segments) > len(out_segments):
            viol_max += len(inp_segments) - len(out_segments)
        return viol_dep, viol_max

    # TODO: Create a Candidate class and evaluate there

    def calc_h(self):
        self.cons_viol['no_coda'] = self.nocoda() * c_weights['no_coda']
        self.cons_viol['dep'], self.cons_viol['max'] = self.maxDep()
        self.cons_viol['dep'] *= c_weights['dep']
        self.cons_viol['max'] *= c_weights['max']
        # self.cons_viol['dep'] = self.dep() * c_weights['dep']
        #self.cons_viol["max"] = self.maxC() * c_weights['max']

        H = sum(self.cons_viol.values())
        print(f"The candidate {self.out1, self.out2} has Harmony : {H}")
        return H, self.cons_viol

    def __repr__(self):
        if self.word2 is not None:
            return f"Constraint checking for '{self.word1}' '{self.word2}'"
        else:
            return f"Constraint checking for '{self.word1}'"


def winner_harmony(inps, winner):
    con = Constraints(inps[0], inps[1], winner[0], winner[1])
    _, cons_viol = con.calc_h()
    return cons_viol


def evaluator(inps, outs, true_cand=None):
    """
    Evaluate a set of candidates
    and update the weights to favour the winner
    """

    viol_optimal = winner_harmony(inps, true_cand)

    # Make all candidates unprobable
    H = torch.ones(len(outs)) * -1e6
    errors = True
    candidates_violations = []
    while errors:
        for n, o in enumerate(outs):
            con = Constraints(inps[0], inps[1], o[0], o[1])
            H[n], cons_violations = con.calc_h()
            candidates_violations.append(cons_violations)
        winner = int(H.argmax())
        print(f"The winner is cand number {winner} --> {outs[winner]}")
        if outs[winner] == true_cand:
            print("Right winner")
            errors = False
        else:
            print("Wrong candidate selected: Update weights")
            update_weights(viol_optimal, candidates_violations[winner])
    # TODO: Case of multiple winner
    max_harm = H.max()
    indices = torch.where(H == max_harm)[0].tolist()
    winner_repr = '\n'.join([str(outs[i]) for i in indices])
    print(f"The winner is/are :\n {winner_repr}")
    return winner, H


def update_weights(violations_optimal, violations_winner, eta=0.5):
    """Weights update #FIXME: it doesn't work

    Strengthen the constraints that favours the winner and
    weaken the constraints that favours other candidates
    """
    for c in violations_optimal.keys():
        if violations_optimal[c] == violations_winner[c]:
            continue
        if violations_winner[c] > violations_optimal[c]:
            c_weights[c] += abs(c_weights[c]) * eta
        else:
            c_weights[c] -= abs(c_weights[c]) * eta
    print(f"New weights: {c_weights}")


if __name__ == '__main__':
    c_weights = {"no_coda": -.3, "dep": -1, "max": -4}
    inps = ["abr", "ata"]
    outs = [["abr", "ata"], ["abra", "ata"], [
        "abara", "ata"], ["ab", "ata"], ["aba", "ata"], ["abi", "ata"]]

    true_cand = ["abara", "ata"]

    print("OPTIMAL :", true_cand)
    winner, hlist = evaluator(inps, outs, true_cand)
    print(f"Winner: {winner}")
    print(f"harmony: {hlist}")
    print(c_weights)
