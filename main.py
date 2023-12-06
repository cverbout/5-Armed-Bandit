import bandit as b
ARMS = 5
STEPS = 100
EPSILON = .4
RUNS = 200
VARIANCE = 1

def main():
    print("this is main")
    b.banditProblem(ARMS, STEPS, EPSILON, RUNS, VARIANCE)
    
if __name__ == "__main__":
    main()