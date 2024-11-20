#packages
using Plots, Statistics, Printf, ProgressMeter, Distributions, StatsPlots, StatsBase, MCMCDiagnosticTools, Random

#remove randomization by setting a seed
Random.seed!(123) 

#function to generate a realization of the discrete-time Lotka-Volterra model: population based implementation

# Generate data by simulating the Lotka-Volterra model and recording the observed population sizes
# The arguments to the functions are:
#   - res: The Sim structure to store the simulation results
#   - prey0: The initial number of prey
#   - pred0: The initial number of predators
#   - λ_breed: The breeding rate of the prey
#   - λ_interaction: The probability that a predator catches a prey
#   - λ_death: The death rate of the predators
#   - max_time: The maximum number of time steps to simulate
#   - d_start: The time step at which to start observing populations
#   - d_end: The time step at which to stop observing populations

function lv_gen(prey0::Int, pred0::Int, λ_breed::Float64, λ_interaction::Float64, λ_death::Float64, max_time::Int)
    
    # set the initial condition of the two populations
    t = 0
    n_prey = [prey0] # initialize the prey population array with the initial population size
    n_pred = [pred0] # initialize the predator population array with the initial population size

    while t < max_time - 1
        
        t += 1 # increase time step by 1
        
        # check if both prey and predator populations are non-zero
        if(n_prey[t] > 0 && n_pred[t] > 0)
            
            prey_reproduced = rand(Binomial(n_prey[t],λ_breed)) # extract a realization of the number of prey reproduced
            predator_died = rand(Binomial(n_pred[t],λ_death))   
            predator_reproduced = rand(Binomial(n_prey[t], 1-(1-λ_interaction)^n_pred[t]))
            
            # update the prey and predator populations using the formulas from the Lotka-Volterra model
            push!(n_prey, n_prey[t] + prey_reproduced - predator_reproduced)
            push!(n_pred, n_pred[t] - predator_died + predator_reproduced)
            
        # Check if there are preys but no predators
        elseif(n_prey[t]>0 && n_pred[t]==0)
            
            prey_reproduced = rand(Binomial(n_prey[t],λ_breed))           
            
            # Add the new preys and update the number of predators
            push!(n_prey,n_prey[t]+prey_reproduced)
            push!(n_pred,0)
        
                # If the predator population is greater than 0 but prey population is 0, 
        # then the predators will die as there is no prey available for them to consume.
        # The following block of code updates the predator and prey populations.
        # The new predator population will be the original predator population minus the 
        # number of predators that died due to lack of prey (aux_d). The new prey population 
        # will be 0 because there are no prey available.
        # The new prey reproduction and predator reproduction values will be 0.
        elseif(n_pred[t]>0 && n_prey[t]==0)
            
            predator_died = rand(Binomial(n_pred[t],λ_death))
            
            push!(n_pred,n_pred[t]-predator_died)
            push!(n_prey,0)
    
        # If both the predator and prey populations are 0, then there is no interaction possible 
        # between the two populations, so both the predator and prey populations remain 0.
        # The new prey reproduction and predator reproduction values will be 0.
        elseif(n_pred[t]==0 && n_prey[t]==0)
            push!(n_pred,0)
            push!(n_prey,0)
    # The function returns the updated predator and prey population values.
        end
    end
    return(n_prey=n_prey,n_pred=n_pred)
end



struct Sim{T}
    n_prey::Vector{T}     # prey population over time
    n_pred::Vector{T}     # predator population over time
    n_preynew::Vector{T}  # new prey population each time step
    n_prednew::Vector{T}  # new predator population each time step
    obs_prey::Vector{T}   # observed prey population at specified times
    obs_pred::Vector{T}   # observed predator population at specified times
    
    # here we initialize the structure
    
    function Sim(T::Int)
        return new{Int}(zeros(Int, T), zeros(Int, T), zeros(Int, T), zeros(Int, T), zeros(Int, T),zeros(Int, T))
    end
end

#function where we introduce partial observations
# We add argument p_obs: The probability of observing a population size at each time step
function lv_gen_and_obs!(res::Sim, prey0::Int, pred0::Int, λ_breed::Float64, λ_interaction::Float64, λ_death::Float64, max_time::Int, p_obs::Float64, d_start::Int, d_end::Int)
    
    # Set initial conditions
    t = 1
    n_prey = prey0
    n_pred = pred0
    res.n_prey[1] = prey0        # Set the initial number of preys
    res.n_pred[1] = pred0        # Set the initial number of predators
    res.n_preynew[1] = prey0         # Initialize the number of new preys at time t
    res.n_prednew[1] = pred0         # Initialize the number of new predators at time t
    
    # Run the simulation
    while t < max_time 
        t+=1
        
        # Generate the new population size based on breeding, interaction and death rates
        new_prey=rand(Binomial(n_prey,λ_breed))
        d_pred=rand(Binomial(n_pred,λ_death))
        new_pred = rand(Binomial(n_prey, 1-(1-λ_interaction)^n_pred))
        
        # Update the population sizes by accounting for the new populations and deaths
        prey = n_prey + new_prey - new_pred
        n_prey = max(0,prey)
        pred = n_pred + new_pred - d_pred
        n_pred = max(0,pred)
        
        # Get the observed population size, if observation window is active
        if(t >= d_start && t <= d_end)
            res.obs_prey[t] = rand(Binomial(new_prey,p_obs))     # Observed number of prey
            res.obs_pred[t] = rand(Binomial(new_pred,p_obs))     # Observed number of predators
        end
        
        # Append the population sizes to the result struct
        res.n_prey[t] = n_prey
        res.n_pred[t] = n_pred
        res.n_preynew[t] = new_prey
        res.n_prednew[t] = new_pred
    end
end

# generate a sample
function gen_sample!(res, prey0, pred0, λ_breed, λ_interaction, λ_death, max_time, p_obs, d_start, d_end, obs0_prey, obs0_pred, ϵ)
    count = 0
    # Generate new observation and store in `res`
    lv_gen_and_obs!(res, prey0, pred0, λ_breed, λ_interaction, λ_death, max_time, p_obs, d_start, d_end)
    # Extract new observations of prey and predator from `res`
    obs1_prey = res.obs_prey
    obs1_pred = res.obs_pred
    d = 0.0
    # Check if the difference between new and old observations is less than the threshold ϵ
    if dist(obs1_prey, obs0_prey) <= ϵ && dist(obs1_pred, obs0_pred) <= ϵ
        count += 1
        # Compute distance between the two observations
        d = dist(obs1_prey, obs0_prey) + dist(obs1_pred, obs0_pred)
    end
    # end
    # Return the count and the distance
    return count, d
end
