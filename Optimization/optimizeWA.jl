### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ 85e3f1ee-9158-11eb-2c00-672502866677
begin 
using Pkg
	Pkg.add("JuMP")
	Pkg.add("Juniper")
	Pkg.add("Ipopt") # Juniper + Ipopt = solver for Quadratic optimization problems like this
using JuMP, Ipopt, Juniper
end

# ╔═╡ 2cfafa80-9157-11eb-35c2-51be3c25e814
html"""<style>
main {
    max-width: 1500px;
}"""

# ╔═╡ feb87f00-914f-11eb-2784-8f05d5919eba
md"""# Optimal Weights and Activation Values"""

# ╔═╡ 22f58180-9153-11eb-00c6-6586404966cc
md"""__Data:__  In Sanskrit we observe an alternation between several allomorphic roots for one and the same verb. Let's take _budh_ 'awake, know, $\ldots$' as a concrete example

+ bodh-ati (3sg. pres.)
+ bhot-sya-ti (3sg. fut.)
+ bhut (Noun, Nom.)
+ bhud-bhis (Noun, Instr.)
+ bud-dha (Past participle)

GSC offers a possible solution to this puzzle if we assume that the input for morphological derivations is a blend of different allomorphs with some partial activation values (see GSCNet).

But how do we find suitable values for these activations and how do we find the right constraint weights in a Gradient Harmonic Grammar?
"""



# ╔═╡ 13e8db00-9154-11eb-187a-0d4cd645dcaa
md"""The problem can be phrased as a mathematical optimization problem: We have a function:

$$H_{winner}(x) = c^Tx$$

expressing the Harmony (sum of violations) of the winner, that we want to maximize. 
Our $x$s are the Harmonic constraints and the coefficients $c$ are the violations for the winning candidate.
This expression is subject to some constraints: for each other candidate it should hold that 

$$H_{winner} > H_{candidate_i} $$ $$ \forall cand_i \in Tableaux$$

We also want the weights to be negative (these are penalties):

$$[x_i, x_2, x_3 ... x_n] \lt 0$$

and we know that the activation values are between 0 and 1: 0 meaning inactive and 1 meaning totally active (discrete).

$$1 \leq [\alpha .... \omega] \geq 0$$

"""

# ╔═╡ 65cb0890-9153-11eb-25b4-b19428afad2c
md""" ### Example: the past participle
__Inputs__:
root allomorphs:

+ (bhud)$^\alpha$
+ (budh)$^\beta$
+ (bhut)$^\gamma$

suffix allomorphs:

+ (-ta)$^\delta$
+ (-dha)$^\epsilon$

__Constraints__:
(all of them $\lt 0$)

+ Max : anti-deletion constraint
+ Dep : anti-epenthesis constraint
+ Voice : Avoid voice contrast ([+ voice][-voice] is bad)
+ License(lar) : Aspirated segments cannot occur before voiceless segments. Also avoid two aspirated next to each other.
+ Lazy : Reduce articulatory effort (avoid aspirated sounds)
+ *h]_W:Aspirated are not allowed in word final position
+ Id(lar) : Aspirated in the input == Aspirated in the output

"""

# ╔═╡ 1f066d00-9153-11eb-00ee-b760c15b6b67
md""" __Toy-example__

| $root_{(\alpha + \beta + \gamma)} + suff_{(\delta + \epsilon)}$ | Max   |   Dep | Voice | Lic | h]W | Lazy | Id(Lar) | Harmony |
|:---:|:---:|:---:|:-----:|:---:|:---:|:---:|:---:|:---:|
|bhud-ta | $\beta + \gamma + \epsilon$ | $(2-\alpha - \delta)$ | 1 | 0 | 0 | 0 | 0 | $M(\beta + \gamma + \epsilon) + D(2-\alpha - \delta) + Voice$ |
|:) bud-dha | $\alpha + \gamma + \delta$ | $(2-\beta - \epsilon)$ | 0 | 0 | 0 | 0 | 1 | $M(\alpha + \gamma + \epsilon) + D(2-\beta - \epsilon) + Id(lar)$ |
|budh-i-ta | $\alpha + \gamma + \epsilon$ | $(3-\beta - \delta)$ | 0 | 0 | 0 | 0 | 1 | $M(\alpha + \gamma + \epsilon) + D(3-\beta - \delta)$ |
| ... |... |... |... |... |... |... |... |... |

Notice that candidate 3 (*budh-i-ta*, built on the model of *kridh-i-ta*) has an extra epenthesis, hence the violation for dep (**3** - $\beta - \delta$)

Let's now solve this problem using the _JuMP_ library for mathematical optimization.
The expression we want to maximize is the Harmony of the winner (candidate 2):

$$max: M(\alpha + \gamma + \epsilon) + D(2-\beta - \epsilon) + Id(lar)$$

subject to:

$M(\alpha + \gamma + \epsilon) + D(2-\beta - \epsilon) + Id(lar) > M(\beta + \gamma + \epsilon) + D(2-\alpha - \delta) + Voice$
$M(\alpha + \gamma + \epsilon) + D(2-\beta - \epsilon) + Id(lar) > M(\alpha + \gamma + \epsilon) + D(3-\beta - \delta)$
$0 \lt [\alpha, \beta, \gamma, \delta, \epsilon] \lt 1$
$M, D, Id(lar), Voice, Lic, h]W < 0$

"""

# ╔═╡ 79976060-915a-11eb-325f-916ca3326950
# Initialize Model
begin
	optimizer = Juniper.Optimizer
	nl_solver = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
	model = Model(optimizer_with_attributes(optimizer, "nl_solver" => nl_solver))
end


# ╔═╡ 75b962e0-915a-11eb-1b24-5d35a2b179e2
begin
	# Declare variables
	a = 1 # The difference between winner and losers should be greater than a
	b = -1 # Start value for the constraint weights
end

# ╔═╡ 4d9bf512-915b-11eb-2fee-e9a1ff31c9b7
begin
	unregister(model, :M)
	unregister(model, :D)
	unregister(model, :A)
	unregister(model, :L)
	unregister(model, :Li)
	unregister(model, :Coda)
	unregister(model, :Id)
	unregister(model, :α)
	unregister(model, :β)
	unregister(model, :γ)
	unregister(model, :δ)
	unregister(model, :ϵ)
end

# ╔═╡ 1bf6e310-9153-11eb-2fe5-5b4d277cff3a
begin
	@variable(model,  -10 <= Ma <= -1, start = b,integer = true) # MaxIO
	@variable(model,  -10 <= De <= -1, start = b,integer = true) # DepIO
	@variable(model,  -10 <= V <= -1, start = b,integer = true) # Assimilate == Voice
	@variable(model,  -10 <= Lic <= -1, start = b,integer = true) # License(lar)
	@variable(model, -10 <= hW <= -1, start = b,integer = true) # h]W
	@variable(model, -10 <= La <= -1, start = b,integer = true) #Lazy
	@variable(model, -10 <= IdL <= -3, start = 2 * b,integer = true) # Ident(lar)
	@variable(model, 0.1 <= α <= 0.9, start = 0.2) # bhud
	@variable(model, 0.1 <= β <= 0.9, start = 0.9) # budh
	@variable(model, 0.1 <= γ <= 0.9, start = 0.4) # bhut
	@variable(model, 0.1 <= δ <= 0.5, start = 0.5) # dha
	@variable(model, 0.5 <= ϵ <= 0.9, start = 0.7) # ta
end

# ╔═╡ 6d36f150-915f-11eb-252a-17639f320ed2
@objective(model, Max, Ma * (α + γ + δ) + De * (2 - β - ϵ) + IdL)

# ╔═╡ 900302e0-915b-11eb-3b97-855f0f06e4ba
# Objective function : Harmony of the winner
"""@objective(model, Max, Ma * (α + γ + δ) + De * (2 - β - ϵ) + IdL)"""

# ╔═╡ 8856e662-915b-11eb-1903-37e8e9ef1ed0
# Constraints
begin
	# Winner Harmony
	winner = @expression(model, Ma * (α + γ + δ) + De * (2 - β - ϵ) + IdL)

	bhudta = @expression(model, winner - (Ma * (β + γ + δ) + De * (2 - α - ϵ) + V))
	@constraint(model, bhudta >= a)

	budhta = @expression(model, winner - (Ma * (α + γ + δ) + De * (2 - β - ϵ) + V + Lic))
	@constraint(model, budhta >= a)

	budhita = @expression(model, winner - (Ma * (α + γ + δ) + De * (3 - β - ϵ)))
	@constraint(model, budhita >= a)
end

# ╔═╡ 6774d1e0-915c-11eb-1070-e98f93030697
"""$model"""

# ╔═╡ 7d023ed0-915c-11eb-25dc-e538e4c8f878
# Optimize / Solve
optimize!(model)

# ╔═╡ 7ae6d8e0-915c-11eb-39a3-bbfbfb998e5a
# Printing logic
# Objective value (Harmony of the winner after the optimization)
JuMP.objective_value(model)

# ╔═╡ 830bb260-915d-11eb-1afa-ffdeaa98ad5a
# Solver status
JuMP.primal_status(model)

# ╔═╡ 80785d50-915d-11eb-1441-2f013038b3e0
JuMP.termination_status(model)

# ╔═╡ 745246e0-915c-11eb-14f6-593066f40ef8
# MaxIO
value(Ma)

# ╔═╡ 6ecf24e0-915c-11eb-28cc-7d5e29ba29f1
#Dep
value(De)

# ╔═╡ b747204e-915d-11eb-1d2b-b77c4386f7f5
# Voice
value(V)

# ╔═╡ c1afebce-915d-11eb-3ba2-71fa0cc58aa1
# Coda
value(hW)

# ╔═╡ c20dea50-915d-11eb-2046-bb9095f0f080
value(α)

# ╔═╡ c22f06e0-915d-11eb-22da-3d03814961cd
value(β)

# ╔═╡ fcbe5450-915d-11eb-1080-c91bfcde29b5
value(γ)

# ╔═╡ 0187ea52-915e-11eb-2475-11a14ba0bc20
value(δ)

# ╔═╡ 062ce150-915e-11eb-1d7b-7558b1067064
value(ϵ)

# ╔═╡ ff36a752-915d-11eb-0835-33c93eaa8233
md"Of course this was only a toy-example. JuMP evaluated the expressions for just two candidates and some value were arbitrarly fixed but I think it gives the idea, how Mathematical Optimization (Quadratic Programming in this case) can be used to search suitable values for the constraints and the activation values"

# ╔═╡ Cell order:
# ╟─2cfafa80-9157-11eb-35c2-51be3c25e814
# ╟─feb87f00-914f-11eb-2784-8f05d5919eba
# ╟─22f58180-9153-11eb-00c6-6586404966cc
# ╟─13e8db00-9154-11eb-187a-0d4cd645dcaa
# ╟─65cb0890-9153-11eb-25b4-b19428afad2c
# ╟─1f066d00-9153-11eb-00ee-b760c15b6b67
# ╟─85e3f1ee-9158-11eb-2c00-672502866677
# ╠═79976060-915a-11eb-325f-916ca3326950
# ╠═75b962e0-915a-11eb-1b24-5d35a2b179e2
# ╟─4d9bf512-915b-11eb-2fee-e9a1ff31c9b7
# ╠═1bf6e310-9153-11eb-2fe5-5b4d277cff3a
# ╠═6d36f150-915f-11eb-252a-17639f320ed2
# ╠═900302e0-915b-11eb-3b97-855f0f06e4ba
# ╠═8856e662-915b-11eb-1903-37e8e9ef1ed0
# ╠═6774d1e0-915c-11eb-1070-e98f93030697
# ╠═7d023ed0-915c-11eb-25dc-e538e4c8f878
# ╠═7ae6d8e0-915c-11eb-39a3-bbfbfb998e5a
# ╠═830bb260-915d-11eb-1afa-ffdeaa98ad5a
# ╠═80785d50-915d-11eb-1441-2f013038b3e0
# ╠═745246e0-915c-11eb-14f6-593066f40ef8
# ╠═6ecf24e0-915c-11eb-28cc-7d5e29ba29f1
# ╠═b747204e-915d-11eb-1d2b-b77c4386f7f5
# ╠═c1afebce-915d-11eb-3ba2-71fa0cc58aa1
# ╠═c20dea50-915d-11eb-2046-bb9095f0f080
# ╠═c22f06e0-915d-11eb-22da-3d03814961cd
# ╠═fcbe5450-915d-11eb-1080-c91bfcde29b5
# ╠═0187ea52-915e-11eb-2475-11a14ba0bc20
# ╠═062ce150-915e-11eb-1d7b-7558b1067064
# ╟─ff36a752-915d-11eb-0835-33c93eaa8233
