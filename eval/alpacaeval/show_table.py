import json
import pandas as pd
import os
from pathlib import Path

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)
    
    
def calculate_win_rates(annotations_file):
    # Load the JSON data
    data = load_json(annotations_file)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Function to check if example is valid (has logprob in preference_raw_completion)
    def is_valid_example(row):
        try:
            return 'logprobs' in row['preference_raw_completion']
        except:
            return False
    
    # Add valid flag
    df['is_valid'] = df.apply(is_valid_example, axis=1)
    
    # Group by dataset and calculate win rates
    results = []
    valid_df = df[df['is_valid']]
    
    # Calculate per-dataset statistics
    for dataset, group in valid_df.groupby('dataset'):
        total = len(group)
        wins = len(group[group['preference'] >= 1.5])
        win_rate = wins / total if total > 0 else 0
        results.append({
            'dataset': dataset,
            'win_rate': win_rate,
            'wins': wins,
            'total': total
        })
    
    # Add "all" row with aggregate statistics
    total_all = len(valid_df)
    wins_all = len(valid_df[valid_df['preference'] >= 1.5])
    win_rate_all = wins_all / total_all if total_all > 0 else 0
    results.append({
        'dataset': 'all',
        'win_rate': win_rate_all,
        'wins': wins_all,
        'total': total_all
    })
    
    return pd.DataFrame(results)
def process_results_directory(base_dir):
    # Find all directories containing '4o-mini'
    results = []
    base_path = Path(base_dir)
    
    for path in base_path.rglob('*4o-mini*/annotations.json'):
        print(path)
        # Calculate win rates for this annotations file
        win_rates_df = calculate_win_rates(str(path))
        print(win_rates_df)
        results.append(win_rates_df)
    
    return results

def merge_results_vertically(base_dir):
    base_path = Path(base_dir)
    csv_files = list(base_path.rglob('*4o-mini*/leaderboard.csv'))
    
    # Check if any CSV files were found
    if not csv_files:
        print(f"No CSV files found in {base_dir}")
        return None
        
    return pd.read_csv(csv_files[0])

# Example usage:
if __name__ == "__main__":
    base_list = [
      "",
      ""
    ]
    
    all_results = []
    
    for base_dir in base_list:
        result = merge_results_vertically(base_dir)
        if result is not None:
            all_results.append(result)
            print(f"Added results from {base_dir}")
        else:
            print(f"Skipping {base_dir} - no valid data found")
    
    # 合并所有结果
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        final_df = final_df[['Unnamed: 0', 'win_rate', 'standard_error', 'avg_length', 'n_total', 'length_controlled_winrate', 'lc_standard_error']]
        output_path = "./leaderboard.csv"
        final_df.to_csv(output_path, index=False)
        print(f"Saved combined results to {output_path}")
    
    





