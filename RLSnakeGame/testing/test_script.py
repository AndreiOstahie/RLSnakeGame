import itertools
import subprocess
import csv

# Define ranges for each parameter
LR_values = [0.001, 0.01, 0.1]
DISCOUNT_RATE_values = [0.9, 0.95, 0.99]
EXPLORATION_VAL_values = [50, 75, 100]
HIDDEN_SIZE_values = [128, 256, 512]


# Generate all combinations of parameter values
parameter_combinations = itertools.product(LR_values, DISCOUNT_RATE_values, EXPLORATION_VAL_values, HIDDEN_SIZE_values)

# Open CSV file for writing
with open('results_rewardFoodDistance.csv', 'w', newline='') as csvfile:
    fieldnames = ['LR', 'DISCOUNT_RATE', 'EXPLORATION_VAL', 'HIDDEN_SIZE', 'Highscore', 'Average_Score']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # Write header
    writer.writeheader()

    count = 0

    # Run snake agent with each combination of parameters
    for params in parameter_combinations:
        # Construct the command to run program with specific parameter values
        command = f"python agent_testing.py --LR {params[0]} --DISCOUNT_RATE {params[1]} --EXPLORATION_VAL {params[2]} --HIDDEN_SIZE {params[3]}"

        # Execute the command using subprocess
        result = subprocess.run(command, shell=True, capture_output=True, text=True)

        # Parse the output to extract highscore and average score
        output_lines = result.stdout.split('\n')
        highscore = output_lines[-3].split()[-1]  # Assuming highscore is printed before the last line
        avg_score = output_lines[-2].split()[-1]  # Assuming average score is printed before the second last line

        print("highscore")
        print(highscore)
        print("avg_score")
        print(avg_score)

        # Write results to CSV
        writer.writerow({
            'LR': params[0],
            'DISCOUNT_RATE': params[1],
            'EXPLORATION_VAL': params[2],
            'HIDDEN_SIZE': params[3],
            'Highscore': highscore,
            'Average_Score': avg_score
        })


        count += 1
        print("Progress")
        print(count)
        print(len(LR_values) * len(DISCOUNT_RATE_values) * len(EXPLORATION_VAL_values) * len(HIDDEN_SIZE_values))
