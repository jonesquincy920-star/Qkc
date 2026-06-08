import torch

from orrey_nav import EnneractBrain, HypercubeAssistant, build_topology


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, adjacency, shells = build_topology(device)
    model = EnneractBrain().to(device)
    model.eval()

    assistant = HypercubeAssistant(model, adjacency, shells, device)
    print(assistant.scan_stellar_sector("Messier 81"))
    print(assistant.calculate_astrometric_course(9.93, 69.07))
    print("\n--- LATEST SCIENCE LOG ---")
    print(assistant.generate_research_report())


if __name__ == "__main__":
    main()
