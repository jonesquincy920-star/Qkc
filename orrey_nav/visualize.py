import time

import matplotlib.pyplot as plt
import torch
from IPython.display import clear_output

from .brain import encode_to_brain

HOPS = 9  # diameter of the 9D hypercube
GRID_SHAPE = (16, 32)  # 512 nodes reshaped for display


def visualize_thought_process(input_data, model, adjacency, shells, device):
    """Propagates a stimulus through the brain and renders a heatmap per hop."""
    current_state = encode_to_brain(input_data, shells, device)
    history = []

    with torch.no_grad():
        x = current_state
        for hop in range(HOPS + 1):
            energy = x[0].abs().mean(dim=1).cpu().numpy()
            history.append(energy)
            if hop < HOPS:
                x = model(x, adjacency)

    for hop, energy_map in enumerate(history):
        clear_output(wait=True)
        grid = energy_map.reshape(GRID_SHAPE)
        plt.figure(figsize=(10, 6))
        plt.imshow(grid, cmap="magma")
        plt.title(f"Orrery Engine Propagation - Hop: {hop}")
        plt.colorbar(label="Activity")
        plt.show()
        time.sleep(0.4)
