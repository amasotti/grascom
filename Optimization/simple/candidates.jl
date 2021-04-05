using Base
mutable struct filler
    sym::String
    voiced::Bool
    aspirated::Bool
    act::Float64
end


α = 0.2
bu = filler("bu",true, false,α)
bhu = filler("bhu",true, true,α+.3)
d = filler("d",true, false,α)
dh = filler("dh", true, true,α+.3)
t = filler("t", false, false,α)
dha = filler("dha", true, true,α)
ta = filler("ta", false, false,α+.3)
dhi = filler("dhi", true, true,α)
di = filler("di", true, false,α)
ti = filler("ti", false, false,α)

initial = [bu, bhu]
final = [d, dh,t,dhi,ti,di]
suffix = [dha, ta]

MAX = 1
DEP = -2
VOICE = -2
LAZY = -3


function concatFiller(x::filler, y::filler, z::filler)
    return x.sym * y.sym * z.sym
end

"""Build all possible combinations

Each candidate will be a Vector with the following elements

initial, final, suffix, string_repr, placeholder for harmony

"""
global candidates = []
for c_i ∈ initial
    for c_f ∈ final
        for c_s ∈ suffix
            push!(candidates, [c_i, c_f, c_s, concatFiller(c_i, c_f, c_s),-Inf])
        end
    end
end



"""Find the candidate with the highest harmony

Return a tuple winner, maxh where

__winner__ : the winner
__maxh__ : harmony of the winner
"""
function maxHarm(candidates::Vector{Any})
    max_h = -Inf
    winner = []
    for el in candidates
        if el[5] > max_h
            max_h = el[5]
            winner = el[1:3]
        end
    end
    return (winner, max_h)
end


# Learning algorithm
error = true # control var

while error
    # Calculate harmony
    for c in candidates
        harm = MAX * (c[1].act + c[2].act + c[3].act) +
            DEP * (3 - c[1].act - c[3].act - c[3].act)

        # Add voice violations
        if c[2].voiced == !c[3].voiced
            harm += VOICE
        end # voice violations

        # Add lazyness violations
        asp = false
        for subs in c[1:3]
            if subs.aspirated
                if asp
                    harm += LAZY
                end # Lazyness
                asp = true
            end
        end # Lazy violations

        # Add epenthesis violations
        if c[2].sym == "dhi" || c[2].sym == "ti" || c[2].sym == "di"
            harm += DEP
        end # epenthesis

        # Add candidate and harmony to tableau/candidates
        c[5] = harm
    end # candidates loop

    # Find winner
    winner, maxh = maxHarm(candidates)

    # Define optimal candidate
    optimal_index = findall(x -> x[4] == concatFiller(bu, d, dha), candidates)
    optimal = candidates[optimal_index][1]

    # case 1 : more than one winner:
    indices_winners = findall(x -> x[5] == maxh, candidates)

    if length(indices_winners) > 1
        for f in optimal[1:3]
            f.act += 0.02
        end  # increase act optimal

        # remove optimal candidate
        indices_winners = filter(x -> x ≠ optimal_index, indices_winners)
        for c in candidates[indices_winners]
            for f in c[1:3]
                f.act > 0.05 ? f.act -= 0.02 : nothing
            end
        end # decrease suboptimal
    elseif winner != optimal[1:3]
        for c in winner
            if c.act >= 0.05
                c.act -= 0.02
            end # decrease act wrong winner
        end # loop over winner
        for f in optimal[1:3]
            f.act += 0.02
        end  # increase act optimal
        display(optimal)
    else
        error = false
        display("The winner is : $(winner)")
        break
    end # check condition
end # while loop


#display(candidates)
open("results.txt", "w") do f
    for i in candidates
        println(f,i)
    end
end
