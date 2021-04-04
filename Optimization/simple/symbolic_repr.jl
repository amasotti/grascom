module symbolic_repr
export filler, output, maxHarm, concatFiller,build_candidates
mutable struct filler
    sym::String
    voiced::Bool
    aspirated::Bool
    act::Float64
end

Base.show(io::IO, x::filler) = print(io, "$(x.sym) : $(x.act)")

mutable struct output
    root_initial::filler
    root_final::filler
    suffix::filler
    repr::String
    harmony::Float64
end

Base.show(io::IO, x::output) = print(io, "$(x.repr) : harmony: $(x.harmony) ($(x.root_initial.sym) : $(x.root_initial.act), $(x.root_final.sym) : $(x.root_final.act), $(x.suffix.sym) : $(x.suffix.act))")

"""Join representations of two or three fillers into a string"""
function concatFiller(x::filler, y::filler, z::filler=nothing)
    if z == nothing
        repr = x.sym * y.sym
    else
        repr = x.sym * y.sym * z.sym
    end
    return repr
end


"""Build all possible combinations

Each candidate will be a Vector with the following elements

initial, final, suffix, string_repr, placeholder for harmony

"""
function build_candidates(initial::Vector{filler}, medial::Vector{filler}, suffix::Vector{filler})
    candidates = []
    for c_i ∈ initial
        for c_f ∈ medial
            for c_s ∈ suffix
                cand = output(c_i, c_f, c_s, concatFiller(c_i, c_f, c_s),-Inf)
                push!(candidates, cand)
            end
        end
    end
    return candidates
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
        if el.harmony > max_h
            max_h = el.harmony
            winner = el.repr
        end
    end
    return (winner, max_h)
end

"""Decrease gradient activations for wrong winners"""
function decrease_activations(x::output)
    x.root_initial.act > 0.05 ? x.root_initial.act -= 0.02 : nothing
    x.root_final.act > 0.05 ? x.root_final.act -= 0.02 : nothing
    x.suffix.act > 0.05 ? x.suffix.act -= 0.02 : nothing
end

"""Increase activations for the winner"""
function increase_activations(x::output)
    x.root_initial.act += 0.02
    x.root_final.act += 0.02
    x.suffix.act += 0.02
end


end # module
