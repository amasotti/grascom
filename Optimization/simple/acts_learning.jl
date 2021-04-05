include("symbolic_repr.jl")


# Fillers list
# For learning
α = 0.1


#---
# For buddha
bu = symbolic_repr.filler("bu",true, false,α+.2) # model higher frequency of bu
bhu = symbolic_repr.filler("bhu",true, true,α)
d = symbolic_repr.filler("d",true, false,α)
dh = symbolic_repr.filler("dh", true, true,α)
t = symbolic_repr.filler("t", false, false,α)
dhi = symbolic_repr.filler("dhi", false, true,α)
di = symbolic_repr.filler("di", false, false,α)
ti = symbolic_repr.filler("ti", false, false,α)
dha = symbolic_repr.filler("dha", true, true,α)
ta = symbolic_repr.filler("ta", false, false,α)


#---
# bodhami
bo = symbolic_repr.filler("bo",true, false,α)
bho = symbolic_repr.filler("bho",true, true,α)
da = symbolic_repr.filler("da",true, false,α)
mi = symbolic_repr.filler("mi",true,false,1) # discrete
# Define possible fillers for possible roles

#---
# bhotsyati
syati = symbolic_repr.filler("syati",false,false,1) # discrete
#---

# Define possible fillers for possible roles
initial = [bo, bho]
final = [d, dh,t,dhi,ti,di]
suffix = [syati]
should_win = "bhotsyati"

# Constraints
MAX = 1
DEP = -1
VOICE = -1
LICENSE = -1


candidates = symbolic_repr.build_candidates(initial, final, suffix)

"""Update constraints weights"""
function update_constraints(winner::String, optimal::String, violations::Vector)
    global MAX
    global DEP
    global VOICE
    global LICENSE
    #Update Constraints
    ## Find indices
    optViol_idx = findall(x -> x["candidate"] == optimal, violations)
    winViol_idx = findall(x -> x["candidate"] == winner, violations)
    ## Find dictionaries entries
    optViol = violations[optViol_idx[1]]
    winViol = violations[winViol_idx[1]]
    # Improve constraints
    optViol["MaxIO"] > winViol["MaxIO"] ? MAX -= .5 : MAX += .5
    optViol["Dep"] > winViol["Dep"] ? DEP += .5 : DEP -= .5
    optViol["Voice"] > winViol["Voice"] ? VOICE += .5 : VOICE -= .5
    optViol["License"] > winViol["License"] ? LICENSE += .5 : LICENSE -= .5
end




# Learning algorithm
error = true # control var
while error
# Calculate harmony
global violations = []
for c in candidates
    # Initialize violations
    cand_viols = Dict("candidate" => c.repr,
                      "MaxIO" => 0.0,
                      "Dep" => 0.0,
                      "Voice" => 0.0,
                      "License" => 0.0)

    #maxIO
    maxio = MAX * (c.root_initial.act + c.root_final.act + c.suffix.act)
    cand_viols["MaxIO"] = maxio

    dep = DEP * (3 - (c.root_initial.act + c.root_final.act + c.suffix.act))
    cand_viols["Dep"] = dep

    harm = maxio + dep

    # Add voice violations
    if c.root_final.voiced == !c.suffix.voiced
        cand_viols["Voice"] += VOICE
        harm += VOICE
    end # voice violations

    # Add License(lar) violations
    if c.root_final.aspirated && !c.suffix.voiced
        if c.root_final ∉ [dhi, ti, di]
            cand_viols["License"] += LICENSE
            #display("$(c) has License violations")
            harm += LICENSE
        end
    end

    # Add epenthesis violations
    if c.root_final in [dhi, ti, di]
        cand_viols["Dep"] += DEP
        harm += DEP
    end # end epenthesis check

    # Add candidate and harmony to tableau/candidates
    c.harmony = harm
    push!(violations, cand_viols)
end # candidates loop

# Find winner
winner, maxh = symbolic_repr.maxHarm(candidates)

# Define optimal candidate
optimal_index = findall(x -> x.repr == should_win, candidates)
optimal = candidates[optimal_index][1]

# case 1 : more than one winner:
indices_winners = findall(x -> x.harmony == maxh, candidates)

if length(indices_winners) > 1
    # increase act optimal
    symbolic_repr.increase_activations(optimal)

    # remove optimal candidate
    indices_winners = filter(x -> x ≠ optimal_index, indices_winners)

    for c in candidates[indices_winners]
        symbolic_repr.decrease_activations(c)
    end # decrease suboptimal

    # Update constraints weights
    update_constraints(winner, optimal.repr,violations)

# Case 2 : unique winner but not the right one
elseif winner != optimal.repr
    win_idx = findall(x -> x.repr == winner, candidates)
    symbolic_repr.decrease_activations(candidates[win_idx[1]])
    symbolic_repr.increase_activations(optimal)

    #Update Constraints
    update_constraints(winner, optimal.repr,violations)

else
    error = false
    display("The winner is : $(winner)")
    break
end # check condition
end # while loop


#display(candidates)
open("results_$(should_win).txt", "w") do f
for i in candidates
    println(f,i)
end
end

# Violations
open("violations_$(should_win).txt", "w") do f
for cand in violations
    println(f,"--------------------------------------------")
    for (key, value) in cand
    println(f, "$(key) = $(value)")
    end
    println(f,"--------------------------------------------")
end
end


# Violations
open("Constraints_weights_$(should_win).txt", "w") do f
println(f, "MAX : $(MAX)")
println(f, "DEP : $(DEP)")
println(f, "License : $(LICENSE)")
println(f, "Voice Contrast : $(VOICE)")
end
