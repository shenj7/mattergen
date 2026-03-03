import argparse
import json
import matplotlib.pyplot as plt
import os

def main():
    parser = argparse.ArgumentParser(description="Plot mean_reward vs epoch from metrics.json")
    parser.add_argument("filepath", type=str, help="Path to metrics.json")
    parser.add_argument("--output", type=str, default=None, help="Output image path (defaults to same dir as metrics.json)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.filepath):
        print(f"Error: File '{args.filepath}' does not exist.")
        return
        
    with open(args.filepath, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: Failed to parse '{args.filepath}'. Ensure it is valid JSON.")
            return
            
    if not isinstance(data, list):
        print("Error: Expected a JSON array of metric objects.")
        return
        
    epochs = []
    mean_rewards = []
    
    for entry in data:
        if "epoch" in entry and "mean_reward" in entry:
            epochs.append(entry["epoch"])
            mean_rewards.append(entry["mean_reward"])
            
    if not epochs:
        print("Error: No 'epoch' or 'mean_reward' data found.")
        return
        
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, mean_rewards, marker='o', linestyle='-', markersize=4)
    plt.title("Mean Reward vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Reward")
    plt.grid(True)
    
    output_path = args.output
    if output_path is None:
        # Default to saving in the same directory as the input file
        base_dir = os.path.dirname(os.path.abspath(args.filepath))
        output_path = os.path.join(base_dir, "mean_reward_plot.png")
        
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to '{output_path}'")

if __name__ == "__main__":
    main()
