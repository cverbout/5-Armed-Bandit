# Reinforcement Learning: 5-Armed Bandit Problem

## Overview

This project explores the application of reinforcement learning techniques, specifically focusing on the 5-armed bandit problem. It contrasts two primary action selection algorithms: Greedy and ε-Greedy, providing insights into their performance and effectiveness in decision-making scenarios.

## Features

- **Two Action Selection Algorithms**: Implements both Greedy and ε-Greedy algorithms, allowing for a comparative analysis of their performance.
- **Customizable Parameters**: Flexibility in setting key parameters like the number of arms (k), time steps (t), ε value, total runs, and variance for reward distribution.
- **Performance Metrics**: Tracks and compares the average reward and optimal action percentage across both algorithms.
- **Visualization**: Includes plots for average rewards and optimal action percentages, offering a clear visual representation of the algorithms' performance over time.

## Technical Stack

- **Python**: Core programming language.
- **NumPy**: Utilized for efficient numerical computations.
- **Matplotlib**: Employed for generating insightful visualizations of the algorithm's performance.

## Program Structure

The project consists of two main Python files:

- `bandit.py`: Contains the core logic for the 5-armed bandit problem, implementing both Greedy and ε-Greedy algorithms and generating performance metrics.
- `main.py`: Serves as the entry point for the program. It imports the functions from `bandit.py` and sets the key parameters for the simulation.

## Running the Program

To run the program, follow these steps:

1. Ensure you have Python installed on your system.
2. Install the required dependencies, NumPy and Matplotlib, using pip: ```pip install numpy matplotlib```
3. Run `main.py` from your terminal or command line:
This will execute the bandit problem simulation with the predefined parameters (ARMS, STEPS, EPSILON, RUNS, VARIANCE) and display the results.

## Results and Insights

The project provides valuable insights into how the Greedy and ε-Greedy algorithms perform under various conditions. Initial findings suggest that the ε-Greedy algorithm, with a balance of exploration and exploitation, tends to outperform the purely Greedy approach in diverse scenarios.
