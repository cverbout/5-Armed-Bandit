import numpy as np 
import matplotlib.pyplot as plt

def banditProblem(k, t, epsilon, runTotal, variance): 
    ### INTIALIZE RESULT ARRAYS ###
    totalAvgRewardEgreedy = np.zeros((1, t))[0]
    totalAvgRewardGreedy = np.zeros((1, t))[0]
    
    totalAvgOptActEgreedy = np.zeros((1, t))[0]
    totalAvgOptActGreedy = np.zeros((1, t))[0]
    
    ### RUN GREEDY ALGORITHMS ###
    runCount = 0
    while runCount < runTotal:
        # INDEPENDENT TESTBED PER RUN
        testBed = np.random.normal(0, variance, size=(1,k))[0]
        
        ### EPSILON GREEDY ###
        avgRewardEgreedy, averageOptActEgreedy = greedy(k, t, epsilon, testBed, variance)
        totalAvgRewardEgreedy += avgRewardEgreedy
        totalAvgOptActEgreedy += averageOptActEgreedy
        
        ### GREEDY ###
        avgRewardGreedy, averageOptActGreedy = greedy(k, t, 0, testBed, variance)
        totalAvgRewardGreedy += avgRewardGreedy
        totalAvgOptActGreedy += averageOptActGreedy
        
        runCount += 1
    
    ### PREPARE DATA FOR PLOT ###    
    totalAvgRewardEgreedy /= runTotal
    totalAvgRewardGreedy /= runTotal
    totalAvgOptActEgreedy /= runTotal
    totalAvgOptActGreedy /= runTotal
    
    ### PLOT RESULTS ###
    
    figure, (ax1, ax2) = plt.subplots(2, 1) 
    figure.suptitle((str(k) + " Armed Bandit: Greedy vs \u03B5-Greedy"))
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Average Reward")
    ax1.plot(range(0,t), totalAvgRewardEgreedy, label="\u03B5 = " + str(epsilon))
    ax1.plot(range(0,t), totalAvgRewardGreedy, label="\u03B5 = 0")
    ax1.legend()
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Optimal Action %")
    ax2.plot(range(0,t), totalAvgOptActEgreedy, label="\u03B5 = " + str(epsilon))
    ax2.plot(range(0,t), totalAvgOptActGreedy, label="\u03B5 = 0")
    ax2.legend()
    plt.show()
    
def greedy(k, t, epsilon, testBed, variance):
    ### INITIALIZATIONS ###
    rewardTracker = np.zeros((1,t))[0]
    optimalActionTracker = np.zeros((1,t))[0]
    estimatedActionValues = np.zeros((1,k))[0]
    actionSelectionCount = np.zeros((1,k))[0]
    
    ### RUN FOR T STEPS ###
    for i in range(0,t):
        p = np.random.rand()
        ### Choose action based on epsilon and random number selection for probability - Break ties randomly ###
        action = np.random.randint(0, k) if (p > 1 - epsilon) else greedyActionBreakTies(estimatedActionValues)
        ### Reward based on action ###
        reward = np.random.normal(testBed[action], variance) 
        ### Update action count ###
        actionSelectionCount[action] += 1
        ### Update estimated value for action ###
        estimatedActionValues[action] += (1/actionSelectionCount[action]) * (reward - estimatedActionValues[action])
        ### Data Collection for result graph ###
        rewardTracker[i] = reward
        optimalActionTracker[i] += 1 if action == np.argmax(testBed) else 0
        
    return rewardTracker, optimalActionTracker

def greedyActionBreakTies(estimatedActionValues):
    maxValue = np.max(estimatedActionValues)
    occurrences = np.where(estimatedActionValues == maxValue)[0]
    action = np.argmax(estimatedActionValues)
    if len(occurrences) > 1:
        action = np.random.choice(occurrences)
    return action
        
    
    

    