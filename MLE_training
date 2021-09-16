# Load X and y variable
using JLD


# Load initial probabilities and transition probabilities of Markov chain
data = load("rain.jld")
X = data["X"]
(n,d) = size(X)


# Split into a training and validation set
splitNdx = Int(ceil(n/2))
trainNdx = 1:splitNdx
validNdx = splitNdx+1:n
Xtrain = X[trainNdx,:]
Xvalid = X[validNdx,:]
nTrain = length(trainNdx)
nValid = length(validNdx)


# a function that will find the initial prob of rain and not rain given a dataset
# returns init p(rain), init p(not rain) in this order
function find_MLE_initial_prob(X)


    # Step 1: Initialize variables to hold init p(rain) and p(not rain) values
    init_p_rain = 0
    init_p_not_rain = 0

    # Step 2: Obtain # of months (ie. n_train or number of rows)
    (n_train,d_train) = size(X)
    @show(n_train, d_train)

    # Step 3: Obtain the number of times it rained by summing down all the rows of column 1
    # Data is binary with 1 = rain. Therefore, summing over rows will count number of times it rained on the first day of month
    sum_rain = 0
    for i in 1:n_train
        sum_rain = sum_rain + X[i,1]
    end 

    # Step 4: Compute the initial probabilities
    init_p_rain = sum_rain / n_train
    init_p_not_rain = 1 - init_p_rain

    # Step 5: Sanity checks - printing values to ensure that probabilities are < 1 and sum to 1
    @show(init_p_rain)
    @show(init_p_not_rain)
    @show(((n_train - sum_rain)/n_train))       # an alternative way to calculate init p(not rain)

    return init_p_rain, init_p_not_rain


end

# Finds the following transition probabilities 
#   1. Rain to rain
#   2. Rain to no rain
#   3. No rain to no rain
#   4. No rain to rain 

function find_MLE_transition_prob(X)

    # Step 1: Initialize variables to hold transition probabilities
    tp_rain_to_RAIN = 0
    tp_rain_to_NOTRAIN = 0
    tp_notrain_to_notrain = 0
    tp_notrain_to_rain = 0

    # Step 2: Find number of days (d_train)
    (n_train,d_train) = size(X)
    @show(n_train, d_train)

    # Step 3: Calculate transition from rain to any state
    tp_rain_to_RAIN, tp_rain_to_NOTRAIN = find_tp_from_rain_state(X, n_train, d_train)

    # Step 4: Sanity Checks to ensure probabilities < 1, sum to 1
    @show(tp_rain_to_RAIN)
    @show(tp_rain_to_NOTRAIN)
    @show (1 - tp_rain_to_RAIN)

    # Step 5: Calculate transition from not_rain to any state
    tp_notrain_to_notrain, tp_notrain_to_rain = find_tp_from_notrain_state(X, n_train, d_train)

    # Step 6: Same sanity checks as step 4
    @show(tp_notrain_to_notrain)
    @show(tp_notrain_to_rain)
    @show(1 - tp_notrain_to_notrain)

    return tp_rain_to_RAIN, tp_rain_to_NOTRAIN, tp_notrain_to_notrain, tp_notrain_to_rain
    
end

# Helper function to find the transition probabilities of 
#  - 1. rain to rain
#  - 2. rain to not rain 
# ^ in the above order 
function find_tp_from_rain_state(X, n_train, d_train)

    # Step 3: Initialize variables to hold counts
    count_rain_to_rain = 0
    count_rain_to_notrain = 0
    count_rain_to_anything = 0
    
    # Step 4: Compute p(x_j = rain | x_{j-1} = rain)
    
    for i in 1:n_train
        # prev_index = j-1 feature
        for prev_index in 1:(d_train-1)
            if ( X[i, prev_index] == 1 )
                count_rain_to_anything = count_rain_to_anything + 1
    
                if ( X[i, prev_index + 1] == 1 )
                    count_rain_to_rain = count_rain_to_rain + 1
                else 
                    count_rain_to_notrain = count_rain_to_notrain + 1
                end
            end 
        end
    end

    tp_rain_to_RAIN = count_rain_to_rain / count_rain_to_anything
    tp_rain_to_NOTRAIN = count_rain_to_notrain / count_rain_to_anything

    return tp_rain_to_RAIN, tp_rain_to_NOTRAIN

end

# Helper function to find the transition probabilities of 
#  - 1. not rain to not rain
#  - 2. not rain to rain 
# ^ in the above order 
function find_tp_from_notrain_state(X, n_train, d_train)

    # Step 3: Initialize variables to hold counts
    count_NOT_TO_RAIN = 0
    count_NOT_TO_NOT = 0
    count_NOT_TO_ANYTHING = 0
    
    # Step 4: Compute 
    #   p(x_j = rain | x_{j-1} = not_rain)
    #   p(x_j = not_rain | x_{j-1} = not_rain)
    
    for i in 1:n_train
        # prev_index = j-1 feature
        # Prev state = not rain
        for prev_index in 1:(d_train-1)
            if ( X[i, prev_index] == 0 )
                count_NOT_TO_ANYTHING = count_NOT_TO_ANYTHING + 1
                # Current state = not rain
                if ( X[i, prev_index + 1] == 0 )
                    count_NOT_TO_NOT = count_NOT_TO_NOT + 1
                else 
                    count_NOT_TO_RAIN = count_NOT_TO_RAIN + 1
                end 
            end 
        end
    end

    tp_NOT_TO_RAIN = count_NOT_TO_RAIN / count_NOT_TO_ANYTHING
    tp_NOT_TO_NOT = count_NOT_TO_NOT / count_NOT_TO_ANYTHING

    return tp_NOT_TO_NOT, tp_NOT_TO_RAIN

end




p_rain, p_not_rain = find_MLE_initial_prob(Xtrain)
tp_R_R, tp_R_NR, tp_NR_NR, tp_NR_R = find_MLE_transition_prob(Xtrain)



# Measure test set NLL
NLL = 0
for i in 1:nValid

    # Initial probabilities
    if (Xvalid[i, 1] == 0)
        global NLL = NLL + log(p_not_rain)
    else 
        global NLL = NLL + log(p_rain)
    end 

    # Transition probabilities 
    for prev_index in 1:( d-1 )

        # Prev state = not rain
        if (Xvalid[i, prev_index] == 0)
            # Current state = not rain 
            if (Xvalid[i, prev_index + 1] == 0)
                global NLL = NLL + log(tp_NR_NR)
            else 
                global NLL = NLL + log(tp_NR_R)
            end

        # Prev state = rain
        else
            # Current state = rain
            if (Xvalid[i, prev_index + 1] == 1)
                global NLL = NLL + log(tp_R_R)
            else 
                global NLL = NLL + log(tp_R_NR)
            end 
        end

    end
    
end
global NLL = NLL * -1 
@show NLL