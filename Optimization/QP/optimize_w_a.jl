using JuMP, Ipopt, Juniper
using LinearAlgebra # for the dot product
optimizer = Juniper.Optimizer
nl_solver = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)

model = Model(optimizer_with_attributes(optimizer, "nl_solver" => nl_solver))

# model = Model(with_optimizer(Ipopt.Optimizer))
# set_optimizer_attribute(model, "max_cpu_time", 60.0)
# set_optimizer_attribute(display_verblevel=1,limits_gap=0.05)
# set_optimizer_attribute(model, print_level=1)

# Declare variables
a = 1
b = -1

@variable(model, -5 <= M <= -1, start = b,integer = true)
@variable(model, -5 <= D <= -1, start = b,integer = true)
@variable(model, -5 <= A <= -1, start = b,integer = true)
@variable(model, -5 <= Li <= -1, start = b,integer = true)
@variable(model, -5 <= Coda <= -1, start = b,integer = true)
@variable(model, -5 <= L <= -1, start = b,integer = true)
@variable(model, -5 <= Id <= -3, start = 2 * b,integer = true)

@variable(model, 0.01 <= α <= 0.9, start = 0.2)
@variable(model, 0.01 <= β <= 0.9, start = 0.9)
@variable(model, 0.01 <= γ <= 0.9, start = 0.4)
@variable(model, 0.01 <= δ <= 0.5, start = 0.5)
@variable(model, 0.5 <= ϵ <= 0.9, start = 0.7)

# δ = .3
# ϵ = .8

# Objective function : Harmony of the winner
@objective(model, Min, M * (α + γ + δ) + D * (2 - β - ϵ) + Id)

# Winner Harmony
winner = @expression(model, M * (α + γ + δ) + D * (2 - β - ϵ) + Id)

bhudta = @expression(model, winner - (M * (β + γ + δ) + D * (2 - α - ϵ) + A))
@constraint(model, bhudta >= a)

budhta = @expression(model, winner - (M * (α + γ + δ) + D * (2 - β - ϵ) + A + Li))
@constraint(model, budhta >= a)

budhita = @expression(model, winner - (M * (α + γ + δ) + D * (3 - β - ϵ)))
@constraint(model, budhita >= a)


# Fake constraints -> realistisch
# @constraint(model, δ <= ϵ - 0.10)
# @constraint(model, α <= β - 0.10)
# @constraint(model, γ <= β - 0.10)


display(model)

# Optimize / Solve
optimize!(model)

# Printing logic
println(JuMP.objective_value(model))
println(JuMP.termination_status(model))
println(JuMP.primal_status(model))
println("Max = ", value(M))
println("Dep = ", value(D))
println("Assimilate = ", value(A))
println("License = ", value(Li))
println("Coda = ", value(Coda))
println("Lazy = ", value(L))
println("Ident lar = ", value(Id))
println("bhud = ", value(α))
println("budh = ", value(β))
println("bhut = ", value(γ))
println("dha = ", value(δ)) 
println("ta = ", value(ϵ))

