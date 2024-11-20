# Function to generate random walk
function randomwalk(r)
    # Generate a random displacement in x and y direction within the square of linear size r
    x = r*(0.5 - rand())
    y = r*(0.5 - rand())
    # Return the tuple of x and y coordinates
    x, y
end

# Function to propose a new point for Metropolis-Hastings algorithm
function proposal(b, d, r, bmin, bmax, dmin, dmax)
    # Generate random walk displacement in b and d directions
    db, dd = randomwalk(r)
    # Calculate the new values for b and d based on the displacements
    bnew = b + db
    dnew = d + dd
    # Keep generating new points until they lie within the given bounds
    while bnew < bmin || bnew > bmax || dnew < dmin || dnew > dmax
        db, dd = randomwalk(r)
        bnew = b + db
        dnew = d + dd
    end
    # Return the tuple of proposed values for b and d
    return bnew, dnew
end

# Create empty arrays to store MCMC samples
all_λ_breed_mcmc1 = []
all_λ_death_mcmc1 = []

# Set parameters for MCMC sampling
res_inf=Sim(max_time)
r = 0.15
ϵ = 50.0
old_num_success = 1
num_repetition_per_pair = 10
distances = zeros(Float64, num_repetition_per_pair)'
λ_breed_max = 1
λ_death_max = 1
l_breed_old = 0.5
l_death_old = 0.5
num_trial = 10000

# set a flag to control when to start saving the values of lambda breed and lambda death (after burn-in)
flag = 0

# set a burn-in value
burn_in = 0

# Run the MCMC sampling for `num_trial` iterations
@showprogress for i in 1:num_trial
    # Propose a new value for λ_breed and λ_death using the `proposal` function
    l_breed, l_death = proposal(l_breed_old, l_death_old, r, 0.0, λ_breed_max, 0.0, λ_death_max)
    dis = zeros(Float64, num_repetition_per_pair)
    num_success = 0
    # Repeat the generation and observation process `num_repetition_per_pair` times
    for i=1:num_repetition_per_pair
        n, d = gen_sample!(res_inf, prey0, pred0, l_breed, λ_interaction, l_death, max_time, p_obs, d_start, d_end, obs0_prey, obs0_pred, ϵ)
        dis[i] = d
        num_success+=n
    end
    # Use Metropolis-Hastings to accept or reject the new proposal
    if rand() < num_success/old_num_success
        # If the flag is greater than or equal to the burn-in value, record the values of lambda breed and lambda death
        if(flag>=burn_in)
            append!(all_λ_breed_mcmc1, l_breed)
            append!(all_λ_death_mcmc1, l_death)
        end

        # update the number of successful iterations, the values of lambda breed and lambda death and record distances
        old_num_success = num_success
        l_breed_old = l_breed
        l_death_old = l_death
        distances = vcat(distances, dis')

        # decrease the value of the tolerance
        ϵ *= 0.999

        # increment the flag
        flag+=1
    end
end
