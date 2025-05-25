from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import os

run_dir = 'tensorboard/DQN_2'

# Load the logs from the selected run directory
event_acc = EventAccumulator(run_dir)
event_acc.Reload()

# Extract all scalar tags
scalar_tags = event_acc.Tags()['scalars']

# Create an output directory for plots
output_dir = "best_param/DQN/tensorboard_plots_DQN"
os.makedirs(output_dir, exist_ok=True)

# Plot and save each scalar metric separately
for tag in scalar_tags:
    scalar_events = event_acc.Scalars(tag)
    steps = [e.step for e in scalar_events]
    values = [e.value for e in scalar_events]

    plt.figure(figsize=(10, 4))
    plt.plot(steps, values)
    plt.title(tag)
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.grid(True)

    # Sanitize filename
    safe_tag = tag.replace("/", "_")
    plt.savefig(os.path.join(output_dir, f"{safe_tag}.png"))
    plt.close()

# List saved files
os.listdir(output_dir)
