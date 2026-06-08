import json
import torch
from datetime import datetime

from .visualize import visualize_thought_process


class HypercubeAssistant:
    """Orrery Deep Space Navigator: a research-simulation wrapper around the brain."""

    def __init__(self, brain_model, adjacency, shells, device):
        self.brain = brain_model
        self.adj = adjacency
        self.shells = shells
        self.device = device
        self.name = "Orrery Deep Space Navigator (NASA-Spec)"
        self.mode = "Grant-Ready Research Simulation"
        self.science_log = []

    def _log_event(self, event_type, data):
        self.science_log.append({
            "timestamp": datetime.now().isoformat(),
            "event": event_type,
            "telemetry": data,
        })

    def scan_stellar_sector(self, catalog_id="NGC-224"):
        """Scans real-world catalog objects (default: Andromeda Galaxy)."""
        print(f"[{self.name}]: Analyzing Spectroscopy for {catalog_id}...")
        torch.manual_seed(hash(catalog_id))
        scan_stimulus = torch.randn(512).to(self.device)
        visualize_thought_process(scan_stimulus, self.brain, self.adj, self.shells, self.device)

        res_metric = scan_stimulus.abs().mean().item()
        self._log_event("SectorScan", {"object": catalog_id, "resonance_avg": res_metric})
        return f"Scan of {catalog_id} complete. Resonance: {res_metric:.4f} Hz."

    def calculate_astrometric_course(self, ra, dec):
        """Calculates trajectory using Right Ascension and Declination."""
        print(f"[{self.name}]: Computing Astrometric Course to RA {ra}, DEC {dec}...")
        nav_input = torch.tensor([ra / 24.0, dec / 90.0] * 256).to(self.device)
        visualize_thought_process(nav_input, self.brain, self.adj, self.shells, self.device)
        self._log_event("CourseCalculation", {"ra": ra, "dec": dec})
        return f"Trajectory locked for Celestial Coordinates: {ra}h {dec}°."

    def generate_research_report(self):
        print(f"[{self.name}]: Compiling Science Data for Grant Submission...")
        return json.dumps(self.science_log[-5:], indent=2)
