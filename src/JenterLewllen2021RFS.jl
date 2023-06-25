"""Replication of Figure 2 in Jenter and Lewellen (2021, RFS)"""

module JenterLewllen2021RFS

using Random
using StableRNGs
using Distributions
using CategoricalArrays
using GLM
using DataFrames
using Optim
using Plots

# Parameters
const RANDOM_SEED = 1284
const ALPHA_1 = -1.4
const BETA_1 = -1.6
const BETA_2 = -0.4
const X_MEAN = 0.1
const X_SD = 0.3

const SAMPLE_SIZE = 1_000_000  # CEO-years


"""
    estimate_decile_probit(y, deciles)

Estimate the simple probit model for the probability of CEO turnover
as a function of performance deciles.
See equation (4) in the paper.

# Arguments
- `y::Vector{Int}`: (0, 1)-valued vector of CEO turnover or not
- `deciles::Vector{Int}`: Vector of performance deciles

# Returns
- `model::GLM.GlmResp{Array{Float64, 1}, Binomial{Float64}, ProbitLink}`: The estimated model
"""
function estimate_decile_probit(y, deciles)
    
    data = DataFrame(y=y, deciles=deciles)
    model = glm(@formula(y ~ deciles), data, Binomial(), ProbitLink(), contrasts=Dict(:deciles => DummyCoding()))

    @info "Estimated coefficients of the simple probit model" model
    
    return model
end


"""
    two_probit_model(x, alpha_1, beta_1, beta_2)

Compute the probability of CEO turnover based on the 'two-probit' model. See equation (5) in the paper.

# Arguments
- `x::Vector{Float64}`: Vector of CEO performances
- `alpha_1::Float64`: Intercept of the probability of other turnover
- `beta_1::Float64`: Intercept of the probability of performance turnover
- `beta_2::Float64`: Slope of the probability of performance turnover

# Returns
- `prob_perf::Vector{Float64}`: Vector of the probability of CEO turnover due to performance
- `prob_other::Float64`: Scalar of the probability of CEO turnover which is unrelated to firm performance
- `prob_turn::Vector{Float64}`: Vector of the probability of CEO turnover
"""
function two_probit_model(x, alpha_1, beta_1, beta_2)
    prob_perf = cdf.(Normal(0,1), beta_1 .+ beta_2 .* x)
    prob_other = cdf(Normal(0,1), alpha_1)
    prob_turn = prob_other .+ (1 - prob_other) .* prob_perf

    return (prob_perf, prob_other, prob_turn)
end

"""
    log_likelihood(params, y, x)

Compute the log-likelihood function of the two-probit model.
"""
function log_likelihood(params, y, x)
    alpha_1, beta_1, beta_2 = params
    _, _, prob_turn = two_probit_model(x, alpha_1, beta_1, beta_2)
    return sum(y .* log.(prob_turn) .+ (1 .- y) .* log.(1 .- prob_turn))
end


"""
    plot_figure(df_summary)

Plot Figure 2 in Jenter and Lewellen (2021, RFS).

# Arguments
- `df_summary::DataFrame`: Summary statistics of the two-probit model and the simple probit model

# Returns
- `p::Plots.Plot`: The plot
"""
function plot_figure(df_summary)
    p = plot(
        df_summary.deciles,
        df_summary.actual_prob_turn_mean,
        label="Actual Total turnover",
        linestyle=:dash,
        linewidth=3,
        color=:gray
    )

    plot!(
        df_summary.deciles,
        df_summary.actual_prob_other_mean,
        label="Actual \"Other\" turnover",
        linestyle=:solid,
        marker=:diamond,
        linewidth=2,
        color=:gray
    )

    plot!(
        df_summary.deciles,
        df_summary.actual_prob_perf_mean,
        label="Actual \"Perf.-ind.\" turnover: Actual",
        linestyle=:solid,
        linewidth=2,
        color=:gray
    )

    plot!(
        df_summary.deciles,
        df_summary.two_probit_prob_perf_mean,
        label="Perf.-ind. turnover: two-probit model",
        linestyle=:solid,
        marker=:star,
        markersize=8,
        markerstrokecolor=:blue,
        linewidth=2,
        color=:blue
    )

    plot!(
        df_summary.deciles,
        df_summary.simple_probit_prob_perf_mean,
        label="Perf.-ind. turnover: Probit with decile dummies",
        linestyle=:dot,
        linewidth=2,
        color=:red
    )

    # Styles
    plot!(
        xlabel="Performance decile",
        ylabel="Turnover Probability",
        ylim=(0, 0.25),
        legend=:outertopright,
        grid=(:y, :dot, 1, 0.9),
        title="Figure 2 in Jenter and Lewellen (2021; RFS)",
        size=(800, 600),
    )

    return p
end


function main()
    # Check the parameters
    @info "Parameters" ALPHA_1 BETA_1 BETA_2 X_MEAN X_SD SAMPLE_SIZE RANDOM_SEED

    # Setup
    rng = StableRNG(RANDOM_SEED)

    performances = rand(rng, Normal(X_MEAN, X_SD), SAMPLE_SIZE)
    deciles = cut(performances, 10, labels=1:10)
    
    # Actual (true) probabilities
    prob_perf, prob_other, prob_turn = two_probit_model(performances, ALPHA_1, BETA_1, BETA_2)

    # Simulate CEO turnover; 1 if turnover, 0 if not
    turnover_or_not = rand.(rng, Bernoulli.(prob_turn)) # (0, 1)-valued vector

    # Estimate the simple probit model
    simple_probit_model = estimate_decile_probit(turnover_or_not, deciles)
    simple_probit_coefs = coef(simple_probit_model)
    
    simple_probit_prob_turn = predict(simple_probit_model)
    simple_probit_prob_other = cdf(Normal(0,1), simple_probit_coefs[1] + simple_probit_coefs[10])
    simple_probit_prob_perf = max.(simple_probit_prob_turn .- simple_probit_prob_other, 0) ./ (1 - simple_probit_prob_other)  # See equation (3)


    # Estimate the two-probit model
    initial_parameter = [0.0, 0.0, 0.0]
    two_probit_result = optimize(params -> -log_likelihood(params, turnover_or_not, performances), initial_parameter, BFGS())
    @info "Optimization result" two_probit_result    
    estimated_parameters = Optim.minimizer(two_probit_result)
    @info "Estimated parameters of the two-probit model" estimated_parameters
    
    two_probit_prob_perf, two_probit_prob_other, two_probit_prob_turn = two_probit_model(performances, estimated_parameters[1], estimated_parameters[2], estimated_parameters[3])

    # Combine the result into a DataFrame
    df = DataFrame(
        performances=performances,
        deciles=deciles,
        turnover_or_not=turnover_or_not,
        actual_prob_turn=prob_turn,
        actual_prob_other=prob_other,
        actual_prob_perf=prob_perf,
        simple_probit_prob_turn=simple_probit_prob_turn,
        simple_probit_prob_other=simple_probit_prob_other,
        simple_probit_prob_perf=simple_probit_prob_perf,
        two_probit_prob_turn=two_probit_prob_turn,
        two_probit_prob_other=two_probit_prob_other,
        two_probit_prob_perf=two_probit_prob_perf
    )

    # Summarize the result grouped by deciles
    df_summary = combine(
        groupby(df, :deciles),
        :turnover_or_not => mean,
        :actual_prob_turn => mean,
        :actual_prob_other => mean,
        :actual_prob_perf => mean,
        :simple_probit_prob_turn => mean,
        :simple_probit_prob_other => mean,
        :simple_probit_prob_perf => mean,
        :two_probit_prob_turn => mean,
        :two_probit_prob_other => mean,
        :two_probit_prob_perf => mean,
    )
    
    # Plot the result
    p = plot_figure(df_summary)
    output_path = savefig(p, "figure2_jenter_lewellen_2021_rfs.png")
    @info "Saved the plot to $(output_path)" 
end

end # module JenterLewllen2021RFS

if abspath(PROGRAM_FILE) == @__FILE__
    JenterLewllen2021RFS.main()
end
