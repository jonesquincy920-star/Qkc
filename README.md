# Orrey Nav

Orrey Nav (the **Orrery Deep Space Navigator**) is a research-simulation
toolkit built around the **Enneract Brain**: a neural model that propagates
signals across a 9-dimensional hypercube graph (512 nodes, 9-hop diameter).

## Components

- `orrey_nav/topology.py` — builds the 9D hypercube graph, its adjacency
  matrix, and node "shells" (grouped by Hamming weight).
- `orrey_nav/brain.py` — `EnneractBrain`, the graph-propagation model, and
  `encode_to_brain`, which maps a 512-element input vector onto brain nodes.
- `orrey_nav/visualize.py` — `visualize_thought_process`, which renders a
  heatmap of neural activity as a stimulus propagates through the brain.
- `orrey_nav/assistant.py` — `HypercubeAssistant`, a navigation/research
  wrapper that scans stellar sectors, computes astrometric courses, and logs
  telemetry for reporting.

## Usage

```bash
pip install -r requirements.txt
python main.py
```

This runs a demo: scanning a stellar sector (Messier 81), calculating an
astrometric course, and printing the resulting science log.
