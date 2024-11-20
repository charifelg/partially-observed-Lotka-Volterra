function randomwalk_sequential(r1,r2)
    x = r1*(0.5 - rand())
    y = r2*(0.5 - rand())
    z = r1*(0.5 - rand())
    x,y,z
end

function proposal_sequential(b, i, d, r1, r2, bmin, bmax, imin, imax, dmin, dmax)
    # This function proposes a new set of parameter values for the model using a sequential Monte Carlo method.
    # It generates a new set of parameter values by randomly walking from the old parameter values, subject to certain constraints.
    # Input:
    # b, i, d: Old parameter values for birth rate, interaction rate, and death rate, respectively
    # r1, r2: Step size in x and z dimensions, and y dimension, respectively, for the random walk
    # bmin, bmax, imin, imax, dmin, dmax: Minimum and maximum allowed parameter values for birth rate, interaction rate, and death rate, respectively
    # Output:
    # Three values, representing the proposed new parameter values for birth rate, interaction rate, and death rate, respectively
    
    bnew = b # initialize new birth rate to old value
    inew = i # initialize new interaction rate to old value
    dnew = d # initialize new death rate to old value
    flag = 0 # initialize flag variable to 0
    
    # Loop until flag is set to 1 (i.e., until a valid set of parameter values is generated)
    while flag == 0
        db, di, dd = randomwalk_sequential(r1,r2) # generate random walk in three dimensions
        bnew = b + db # propose new birth rate value by adding step in x dimension
        inew = i + di # propose new interaction rate value by adding step in y dimension
        dnew = d + dd # propose new death rate value by adding step in z dimension
        flag = 1 # assume new parameter values are valid unless proven otherwise
        if bnew < bmin || bnew > bmax || inew < imin || inew > imax || dnew < dmin || dnew > dmax
            flag = 0 # if any of the proposed parameter values are outside the allowed range, set flag to 0 to indicate invalid parameter values
        end
    end
    
    return bnew, inew, dnew # return proposed new parameter values
end

function reweight(w0, θpop, θi, θsize, radius1, radius2)
    # `reweight` function takes in a set of weights `w0`, a population of parameter sets `θpop`, 
    # the current parameter set `θi`, the size of the population `θsize`, and two radii values `radius1` and `radius2`.
    
    s = 0 # Initialize the sum of weights

    # Iterate over all parameter sets in the population
    for i in 1:θsize
        # Check if the difference between each parameter value of the current parameter set `θi`
        # and the corresponding parameter value of the parameter set in the population `θpop[i]` 
        # is less than the radius values
        if ((abs(θpop[i, 1] - θi[1]) < radius1/2) && (abs(θpop[i, 2] - θi[2]) < radius2/2) && (abs(θpop[i, 3] - θi[3]) < radius1/2))
            s += w0[i] # Add the weight of the parameter set in the population to the sum of weights
        end
    end

    # Return the reciprocal of the sum of weights
    1.0 / s
end


# Create a simulation object with the maximum time
res2 = Sim(max_time)

# Set the observation probability
p_obs = 0.9

# Run the Lotka-Volterra model with the specified parameters and observation settings
lv_gen_and_obs!(res2, prey0, pred0, λ_breed, λ_interaction, λ_death, max_time, p_obs, d_start, d_end)

# Set the value of ϵ
ϵ = 10

# Create a simulation object to hold simulation data
res_inf = Sim(max_time)

# Population parameters and initialization
θsize = 2000
bmin = 0.0
bmax = 1.0
imin = 0.0
imax = 0.05
dmin = 0.0
dmax = 1.0
θpop = zeros(θsize, 3)
θindex=[i for i in 1:θsize]

# Generate initial population
# Assign random values to θpop[i,2] (immigration rate) and θpop[i,3] (predation rate) for each i in 1:θsize
# Assign a random value to θpop[i,1] (birth rate) for each i in 1:θsize
[θpop[i,2] = imin + rand() * (imax - imin) for i in 1:θsize]
[θpop[i,1] = rand() for i in 1:θsize]
[θpop[i,3] = rand() for i in 1:θsize]

# Set up initial weights
wθ = ones(θsize) / θsize

# Set up old weights and old population
wθ_old = wθ / sum(wθ)
θpop_old = θpop

# Set up other variables for algorithm
old_num_success = 1
num_repetition_per_pair = 1
radius1 = 0.1
radius2 = 0.0025
num_trial = 1000

#Set a variable that counts number of accepted values
acceptances_smc = 0

# Perform the genetic algorithm
@showprogress for i in 1:num_trial
    # Select a random individual from the previous population
    for j in 1:θsize
        ind = sample(θindex, Weights(wθ_old))
        
        # Propose a new individual
        θnew = proposal_sequential(θpop_old[ind,1], θpop_old[ind,2], θpop_old[ind,3], radius1, radius2, bmin, bmax, imin, imax, dmin, dmax)
        
        # Evaluate the performance of the new individual
        num_success, _ = gen_samples!(res_inf, prey0, pred0, θnew[1], θnew[2], θnew[3], max_time, p_obs, d_start, d_end, num_repetition_per_pair, obs0_prey_smc, obs0_pred_smc, ϵ)
        
        # Update the population based on the performance of the new individual
        if rand() < num_success / old_num_success
            θpop[ind,1] = θnew[1]
            θpop[ind,2] = θnew[2]
            θpop[ind,3] = θnew[3]
            wθ[ind] = reweight(wθ_old, θpop_old, θpop[ind,1:3], θsize, radius1, radius2)
            old_num_success = num_success
            acceptances_smc+=num_success
        end
    end
    
    # Decrease the value of ϵ
    ϵ *= 0.999
    
    # Update the weights and population
    wθ_old = wθ / sum(wθ)
    θpop_old = θpop
end
