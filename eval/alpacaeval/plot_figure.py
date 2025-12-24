import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the data
df = pd.read_csv('/home/zhe/eval/alpacaeval/leaderboard.csv')
df = df[['name', 'length_controlled_winrate']]

# Extract model names and iteration numbers
df['model_type'] = df['name'].apply(lambda x: x.split('-')[-1])
df['model_type'] = df['model_type'].apply(lambda x: '_'.join(x.split('_')[:-1]))
df['iteration'] = df['name'].apply(lambda x: int(x.split('_')[-1]))

# Set figure style and background
plt.style.use('default')
plt.figure(figsize=(16, 12), facecolor='white')  # Increased figure size
ax = plt.gca()
ax.set_facecolor('white')

# Create grouped bar plot
bar_width = 0.15
opacity = 0.8

model_types = ['evol', 'autoevol', 'codeclm_iter', 'tag_instruct','tag_instruct_reward']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

for idx, model in enumerate(model_types):
    # Use exact match instead of overlapping matches
    model_data = df[df['model_type'] == model]
    plt.bar(
        [x + idx*bar_width - bar_width*2 for x in model_data['iteration']], 
        model_data['length_controlled_winrate'],
        bar_width,
        alpha=opacity,
        color=colors[idx],
        label=model
    )
    print(f"{model}: {model_data['length_controlled_winrate'].values}")
    
# Customize plot
plt.xlabel('Iteration', fontsize=20, fontweight='bold')  # Increased font size
plt.ylabel('Win Rate (%)', fontsize=20, fontweight='bold')  # Increased font size
plt.title('Model Performance Across Iterations', fontsize=24, fontweight='bold', pad=20)  # Increased font size
plt.xticks([r for r in range(5)], ['1', '2', '3', '4', '5'], fontsize=20)  # Increased font size
plt.yticks(fontsize=16)  # Increased font size

# Add grid and border
plt.grid(True, linestyle='--', alpha=0.3)
ax.spines['top'].set_color('black')
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['top'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)

# Place legend in the upper left corner with a box
plt.legend(loc='upper left', frameon=True, edgecolor='black', fontsize=20)  # Increased font size

# Add text annotation for units
plt.text(0.02, -0.15, 'Note: Win rate is length-controlled and measured in percentage points', 
         transform=ax.transAxes, fontsize=20, style='italic')  # Increased font size

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()