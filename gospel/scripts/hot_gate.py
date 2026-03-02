from __future__ import annotations

import time
from enum import Enum
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

import numpy as np
import typer
from graphix import command
from graphix.rng import ensure_rng
from tqdm import tqdm
from veriphix.client import Client, Secrets
from veriphix.trappifiedCanvas import TrappifiedCanvas

from gospel.brickwork_state_transpiler import (
    ConstructionOrder,
    generate_random_pauli_pattern,
    get_bipartite_coloring,
)
from gospel.noise_models.faulty_gate_noise_model import FaultyCZNoiseModel
from gospel.scripts.qasm2brickwork_state import (
    draw_brickwork_state_colormap_from_pattern,
)
from gospel.stim_pauli_preprocessing import (
    StimBackend,
)

if TYPE_CHECKING:
    from graphix.pattern import Pattern
    from numpy.random import Generator


class Method(Enum):
    StimBackend = "stim-backend"
    DensityMatrix = "density-matrix"
    StimShots = "stim-shots"


def perform_simulation(
    pattern: Pattern,
    #method: Method = StimBackend,
    method: Method = Method.StimBackend,
    depol_prob: float = 0.0,
    chosen_edges: frozenset[frozenset[int]] | None = None,
    shots: int = 1,
    rng: Generator | None = None,
) -> list[dict[int, int]]:
    # NOTE data validation? nqubits, nlayers larger than 0, p between 0 and 1,n shots int >0

    rng = ensure_rng(rng)

    # for order in (ConstructionOrder.Canonical, ConstructionOrder.Deviant):

    # dummy computation
    # only canonical ordering

    # Initialize secrets and client
    secrets = Secrets(r=False, a=False, theta=False)
    client = Client(pattern=pattern, secrets=secrets)

    # Get bipartite coloring and create test runs
    colours = get_bipartite_coloring(pattern)
    test_runs = client.create_test_runs(manual_colouring=colours)

    # Define noise model
    # don't reinitialise it since has its own randomness

    # noise_model = UncorrelatedDepolarisingNoiseModel(entanglement_error_prob=depol_prob)

    # specific to 7 qubits and brick depth 2 instance

    noise_model = FaultyCZNoiseModel(
        entanglement_error_prob=depol_prob,
        edges=pattern.edges,
        chosen_edges=chosen_edges,
    )

    # noise_model = DepolarisingNoiseModel(entanglement_error_prob = 0.001)

    results_table = []
    n_failures = 0

    if method == Method.StimBackend:
        for i in tqdm(range(shots)):  # noqa: B007
            # reinitialise the backend!
            backend = StimBackend()
            # generate trappiefied canvas (input state is refreshed)

            run = TrappifiedCanvas(test_runs[rng.integers(len(test_runs))], rng=rng)

            # Delegate the test run to the client
            trap_outcomes = client.delegate_test_run(  # no noise model, things go wrong
                backend=backend, run=run, noise_model=noise_model
            )

            # Create a result dictionary (trap -> outcome)
            result = {
                trap: outcome for (trap,), outcome in zip(run.traps_list, trap_outcomes)
            }

            results_table.append(result)

            # Print pass/fail based on the sum of the trap outcomes
            if sum(trap_outcomes) != 0:
                n_failures += 1
                # print(f"Iteration {i}: ❌ Trap round failed", flush=True)
            else:
                pass
                # print(f"Iteration {i}: ✅ Trap round passed", flush=True)
    elif method in {Method.DensityMatrix, Method.StimShots}:
        raise NotImplementedError

    # Final report after completing the test rounds
    print(
        f"Final result: {n_failures}/{shots} failed rounds",
        flush=True,
    )
    print("-" * 50, flush=True)
    return results_table


def compute_failure_probabilities(
    results_table: list[dict[int, int]],
) -> dict[int, float]:
    """Compute failure probabilities from simulation results."""
    if not results_table:
        return {}
    
    # Get all unique trap nodes that appear in any result
    all_traps = set()
    for results in results_table:
        all_traps.update(results.keys())
    
    # Initialize counters for all traps
    occurrences = {trap: 0 for trap in all_traps}
    occurrences_one = {trap: 0 for trap in all_traps}
    
    # Count occurrences and failures
    for results in results_table:
        for trap, result_val in results.items():
            occurrences[trap] = occurrences.get(trap, 0) + 1
            if result_val == 1:
                occurrences_one[trap] = occurrences_one.get(trap, 0) + 1
    
    return {trap: occurrences_one[trap] / occurrences[trap] for trap in all_traps}

def plot_heatmap(
    pattern: Pattern,
    data: dict[int, float],
    target: Path,
    heavy_edges: set[tuple[int, int]] | None = None,
) -> None:
    print(f"Plotting heatmap to: {target.absolute()}")
    print(f"Data points: {len(data)}")
    
    # Ensure we have data for all nodes
    if not data:
        print("Warning: No failure data to plot. Creating default heatmap with zeros.")
        from gospel.scripts.qasm2brickwork_state import OpenGraph
        graph = OpenGraph.from_pattern(pattern)
        data = {node: 0.0 for node in graph.inside.nodes()}
    
    try:
        # Import matplotlib with non-interactive backend
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        
        # Ensure the directory exists
        target.parent.mkdir(parents=True, exist_ok=True)
        
        # Call the original function
        draw_brickwork_state_colormap_from_pattern(
            pattern=pattern, target=target, failure_probas=data, heavy_edges=heavy_edges
        )
        
        # Explicitly close all figures to ensure file is written
        plt.close('all')
        
        # Verify the file was created
        if target.exists():
            file_size = target.stat().st_size
            print(f"✓ File created successfully: {target.absolute()} ({file_size} bytes)")
            
            # Basic file validation
            if file_size < 100:  # PNG files should be at least 100 bytes
                print("⚠ Warning: File seems very small, might be corrupted")
            else:
                print("✓ File size looks good")
        else:
            print("❌ Error: File was not created!")
            
    except Exception as e:
        print(f"❌ Error creating heatmap: {e}")
        import traceback
        traceback.print_exc()
        
        # Create a simple fallback plot
        create_fallback_plot(target, str(e))

def create_fallback_plot(target: Path, error_msg: str = "") -> None:
    """Create a simple fallback plot when the main one fails."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 8))
        plt.text(0.5, 0.7, "Brickwork State Heatmap", 
                ha='center', va='center', fontsize=20, weight='bold')
        plt.text(0.5, 0.5, "Original plot failed", 
                ha='center', va='center', fontsize=16, style='italic')
        if error_msg:
            plt.text(0.5, 0.4, f"Error: {error_msg[:100]}...", 
                    ha='center', va='center', fontsize=10, wrap=True)
        plt.text(0.5, 0.2, f"File: {target.name}", 
                ha='center', va='center', fontsize=12)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(target, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Created fallback plot at: {target.absolute()}")
    except Exception as e:
        print(f"❌ Even fallback plot failed: {e}")

# specific for nqubits = 7 and nlayers = 2
CHOSEN_EDGES = frozenset(
    frozenset(edge)
    for edge in [(0, 5), (7, 12), (27, 32), (33, 34)]
)

def test_matplotlib_output(target: Path):
    """Test if matplotlib can create and save a simple plot"""
    try:
        plt.figure(figsize=(8, 6))
        plt.plot([1, 2, 3], [1, 4, 9])
        plt.title("Test Plot")
        plt.savefig(target, dpi=300, bbox_inches='tight')
        plt.close()
        
        if target.exists():
            file_size = target.stat().st_size
            print(f"Test plot created successfully: {target} ({file_size} bytes)")
            return True
        else:
            print("Test plot failed - file not created")
            return False
    except Exception as e:
        print(f"Test plot failed with error: {e}")
        return False


def cli(
    order: ConstructionOrder,
    target: Path,
    nqubits: int = 5,
    nlayers: int = 2,
    seed: int = 12345,
    depol_prob: float = 0.5,
    shots: int = 10000,
) -> None:
    # Create the directory first
    target.parent.mkdir(parents=True, exist_ok=True)
    
    # Test matplotlib first
    test_target = target.parent / "test_plot.png"
    print("Testing matplotlib output...")
    if not test_matplotlib_output(test_target):
        print("Matplotlib test failed! The issue is with matplotlib configuration.")
        return
    
    # Rest of your existing code...
    # initialising pattern
    rng = np.random.default_rng(seed)

    pattern = generate_random_pauli_pattern(
        nqubits=nqubits, nlayers=nlayers, order=order, rng=rng
    )
    # Add measurement commands to the output nodes
    for onode in pattern.output_nodes:
        pattern.add(command.M(node=onode))

    print("Starting simulation...")
    start = time.time()
    results_table = perform_simulation(
        pattern, depol_prob=depol_prob, shots=shots, chosen_edges=CHOSEN_EDGES
    )

    print(f"Simulation finished in {time.time() - start:.4f} seconds.")

    print("Computing failure probabilities...")
    failure_probas = compute_failure_probabilities(results_table)

    print("Plotting the heatmap...")
    plot_heatmap(pattern, failure_probas, target)
    print("Done!")

if __name__ == "__main__":
    typer.run(cli)
