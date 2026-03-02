from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import LogNorm
from graphix.opengraph import OpenGraph
from tqdm import tqdm

from gospel.brickwork_state_transpiler import (
    get_node_positions,
    layers_to_measurement_table,
    transpile,
    transpile_to_layers,
)
from gospel.scripts.qasm_parser import read_qasm

if TYPE_CHECKING:
    from graphix import Circuit, Pattern


def format_angle(angle: float) -> str:
    """
    Converts an angle in radians to a string representation as a multiple of π.
    """
    # If the angle is effectively zero, return "0"
    if abs(angle) < 1e-12:
        return "0"

    # Convert angle/π to a Fraction, limiting the denominator for a neat representation.
    frac = Fraction(angle).limit_denominator(1000)
    num, den = frac.numerator, frac.denominator

    # Determine the sign and work with absolute value for formatting.
    sign = "-" if num < 0 else ""
    num = abs(num)

    # When denominator is 1, we don't need to show it.
    if den == 1:
        if num == 1:
            return f"{sign}π"
        return f"{sign}{num}π"
    if num == 1:
        return f"{sign}π/{den}"
    return f"{sign}{num}π/{den}"


def draw_brickwork_state_pattern(pattern: Pattern, target: Path) -> None:
    graph = OpenGraph.from_pattern(pattern)
    pos = get_node_positions(pattern, reverse_qubit_order=True)
    labels = {
        node: format_angle(measurement.angle)
        for node, measurement in graph.measurements.items()
    }
    plt.figure(
        figsize=(max(x for x, y in pos.values()), max(y for x, y in pos.values()))
    )
    nx.draw(
        graph.inside,
        pos,
        labels=labels,
        node_size=1000,
        node_color="white",
        edgecolors="black",
        font_size=9,
    )
    plt.savefig(target, format="svg")
    plt.close()


def draw_brickwork_state_colormap(
    circuit: Circuit, target: Path, failure_probas: dict[int, float]
) -> None:
    """Draw the brickwork state with trap failure probability drawn as the node color.
    Heavily redundant since we have already gone through the transpilation step.

    Parameters
    ----------
    circuit : Circuit

    target : Path
        where to save the figure
    failure_probas : dict[int, float]
        dictionary of failure probability (value) by node (key)
    """

    pattern = transpile(circuit)
    graph = OpenGraph.from_pattern(pattern)
    pos = get_node_positions(pattern, reverse_qubit_order=True)
    labels = {node: node for node in graph.inside.nodes()}
    colors = [failure_probas[node] for node in graph.inside.nodes()]

    plt.figure(
        figsize=(max(x for x, y in pos.values()), max(y for x, y in pos.values()))
    )
    nx.draw_networkx_edges(graph.inside, pos, edge_color="black")
    # false error: Argument "node_color" to "draw_networkx_nodes" has incompatible type "list[float]"; expected "str"  [arg-type]
    # false error: Module has no attribute "jet"  [attr-defined]
    nc = nx.draw_networkx_nodes(
        graph.inside,
        pos,
        nodelist=graph.inside.nodes,
        label=labels,
        node_color=colors,  # type: ignore[arg-type]
        node_size=1000,
        #cmap=plt.cm.jet,  # type: ignore[attr-defined]
        cmap="magma",
        vmin=1e-4,
        vmax=1,
    )
    plt.colorbar(nc)
    plt.axis("off")
    plt.savefig(target, format="svg")
    plt.close()


"""def draw_brickwork_state_colormap_from_pattern(
    pattern: Pattern,
    target: Path,
    failure_probas: dict[int, float],
    heavy_edges: set[tuple[int, int]] | None = None,
) -> None:
"""
"""Draw the brickwork state with trap failure probability drawn as the node color.
    Heavily redundant since we have already gone through the transpilation step.

    Parameters
    ----------
    circuit : Circuit

    target : Path
        where to save the figure
    failure_probas : dict[int, float]
        dictionary of failure probability (value) by node (key)
    """

"""graph = OpenGraph.from_pattern(pattern)
    pos = get_node_positions(pattern, reverse_qubit_order=True)
    labels = {node: node for node in graph.inside.nodes()}
    colors = [failure_probas[node] for node in graph.inside.nodes()]

    plt.figure(
        figsize=(max(x for x, y in pos.values()), max(y for x, y in pos.values()))
    )

    nx.draw_networkx_edges(graph.inside, pos, edge_color="black")

    # heavy edge overlay if provided
    if heavy_edges is not None:
        filtered_nodes: set[int] = set()

        for edge in heavy_edges:
            filtered_nodes.update(edge)  # should work
        filtered_pos = {i: j for i, j in pos.items() if i in filtered_nodes}

        heavy_graph = nx.Graph(heavy_edges)

        nx.draw_networkx_edges(heavy_graph, filtered_pos, edge_color="red", width=5)
    # false error: Argument "node_color" to "draw_networkx_nodes" has incompatible type "list[float]"; expected "str"  [arg-type]
    # false error: Module has no attribute "jet"  [attr-defined]
    nc = nx.draw_networkx_nodes(
        graph.inside,
        pos,
        nodelist=graph.inside.nodes,
        label=labels,
        node_color=colors,  # type: ignore[arg-type]
        node_size=1000,
        #cmap=plt.cm.jet,  # type: ignore[attr-defined]
        cmap="magma",
        vmin=0,
        vmax=1,
    )
    plt.colorbar(nc)
    plt.axis("off")
    plt.savefig(target, format="svg")
    plt.close()
    """

"""graph = OpenGraph.from_pattern(pattern)
    pos = get_node_positions(pattern, reverse_qubit_order=True)
    labels = {node: node for node in graph.inside.nodes()}

    # Collect values for all nodes (default to 0.0 if missing)
    values = np.array([failure_probas.get(node, 0.0) for node in graph.inside.nodes()])
    nonzero = values[values > 0]

    # Choose a log color scale that is FIXED across experiments
    # This makes runs with depol_prob = 1e-2 and 1e-1 directly comparable.
    if nonzero.size == 0:
        # Degenerate case: everything is zero. Fall back to linear.
        norm = None
        cmap = "magma"
        colors = values
        tick_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        tick_labels = [str(t) for t in tick_values]
    else:
        # Log scale from 1e-5 up to 1 (or you can stop at 1e-1 if you know it never gets higher)
        vmin = 1e-5
        vmax = 1.0

        norm = LogNorm(vmin=vmin, vmax=vmax)

        # Mask zeros so they use the "bad" color (no error)
        colors = np.where(values == 0, np.nan, values)

        cmap = plt.get_cmap("magma").copy()
        cmap.set_bad("lightgray")  # zero-probability nodes

        # Ticks at orders of magnitude
        tick_values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
        tick_labels = ["1e-5", "1e-4", "1e-3", "1e-2", "1e-1", "1"]

    plt.figure(
        figsize=(max(x for x, y in pos.values()), max(y for x, y in pos.values()))
    )

    nx.draw_networkx_edges(graph.inside, pos, edge_color="black")

    # heavy edge overlay if provided (unchanged)
    if heavy_edges is not None:
        filtered_nodes: set[int] = set()
        for edge in heavy_edges:
            filtered_nodes.update(edge)
        filtered_pos = {i: j for i, j in pos.items() if i in filtered_nodes}
        heavy_graph = nx.Graph(heavy_edges)
        nx.draw_networkx_edges(heavy_graph, filtered_pos, edge_color="red", width=5)

    # Draw nodes with chosen colormap / norm
    if norm is None:
        nc = nx.draw_networkx_nodes(
            graph.inside,
            pos,
            nodelist=graph.inside.nodes,
            label=labels,
            node_color=colors,
            node_size=1000,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
        )
    else:
        nc = nx.draw_networkx_nodes(
            graph.inside,
            pos,
            nodelist=graph.inside.nodes,
            label=labels,
            node_color=colors,
            node_size=1000,
            cmap=cmap,
            norm=norm,
        )

    cbar = plt.colorbar(nc)
    cbar.set_ticks(tick_values)
    cbar.set_ticklabels(tick_labels)
    cbar.set_label("Trap failure probability")

    plt.axis("off")
    plt.savefig(target, format="svg")
    plt.close()
"""

def draw_brickwork_state_colormap_from_pattern(
    pattern: Pattern,
    target: Path,
    failure_probas: dict[int, float],
    heavy_edges: set[tuple[int, int]] | None = None,
) -> None:
    """Draw the brickwork state with trap failure probability as node color."""

    graph = OpenGraph.from_pattern(pattern)
    pos = get_node_positions(pattern, reverse_qubit_order=True)

    nodes = list(graph.inside.nodes())
    labels = {node: node for node in nodes}

    # Get failure probabilities for all nodes (default 0.0)
    values = np.array([failure_probas.get(node, 0.0) for node in nodes])
    nonzero = values[values > 0]

    # Set up figure
    plt.figure(
        figsize=(max(x for x, y in pos.values()), max(y for x, y in pos.values()))
    )

    # Base edges
    nx.draw_networkx_edges(graph.inside, pos, edge_color="black")

    # Heavy edge overlay if provided
    if heavy_edges is not None:
        filtered_nodes: set[int] = set()
        for edge in heavy_edges:
            filtered_nodes.update(edge)
        filtered_pos = {i: j for i, j in pos.items() if i in filtered_nodes}
        heavy_graph = nx.Graph(heavy_edges)
        nx.draw_networkx_edges(heavy_graph, filtered_pos, edge_color="red", width=5)

    # If everything is zero, just draw a boring linear colormap and bail
    if nonzero.size == 0:
        nc = nx.draw_networkx_nodes(
            graph.inside,
            pos,
            nodelist=nodes,
            label=labels,
            node_color=values,
            node_size=1000,
            cmap="magma",
            vmin=1e-4,
            vmax=1.0,
        )
        cbar = plt.colorbar(nc)
        cbar.set_label("Trap failure probability")
        plt.axis("off")
        plt.savefig(target, format="svg")
        plt.close()
        return

    # ---- LOG SCALE BRANCH ----

    # Fix the log range across experiments so 1e-2 vs 1e-1 is comparable.
    vmin = 1e-4
    vmax = 1.0  # if your probs never reach 1, you can set this to 1e-1

    norm = LogNorm(vmin=vmin, vmax=vmax)

    # Split nodes into zero / nonzero
    zero_nodes = [n for n, v in zip(nodes, values) if v == 0.0]
    nonzero_nodes = [n for n, v in zip(nodes, values) if v > 0.0]
    nonzero_values = np.array([v for v in values if v > 0.0])

    # Map nonzero probabilities to [0, 1] using the log norm
    mapped_values = norm(nonzero_values)  # all between 0 and 1

    # First draw zero-probability nodes in a neutral color
    if zero_nodes:
        nx.draw_networkx_nodes(
            graph.inside,
            pos,
            nodelist=zero_nodes,
            node_color="black",  # "no error" nodes
            node_size=1000,
        )

    # Draw nonzero nodes with mapped values and a colormap
    nc = nx.draw_networkx_nodes(
        graph.inside,
        pos,
        nodelist=nonzero_nodes,
        label=labels,
        node_color=mapped_values,  # already in [0, 1]
        node_size=1000,
        cmap="magma",
        vmin=1e-4,
        vmax=1.0,
    )

    # Colorbar: positions are in [0,1], labels are the real values
    cbar = plt.colorbar(nc)
    tick_values = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    tick_positions = norm(tick_values)  # also in [0, 1]

    cbar.set_ticks(tick_positions)
    cbar.set_ticklabels(["0", "1e-3", "1e-2", "1e-1", "1.0"])
    cbar.set_label("Trap failure probability")

    plt.axis("off")
    plt.savefig(target, format="svg")
    plt.close()



def convert_circuit_directory_to_brickwork_state_svg(
    path_circuits: Path, path_brickwork_state_svg: Path
) -> None:
    path_brickwork_state_svg.mkdir()
    for path_circuit in tqdm(list(path_circuits.glob("*.qasm"))):
        with Path(path_circuit).open() as f:
            circuit = read_qasm(f)
            pattern = transpile(circuit)
            target = (path_brickwork_state_svg / path_circuit.name).with_suffix(".svg")
            draw_brickwork_state_pattern(pattern, target)


def convert_circuit_directory_to_brickwork_state_table(
    path_circuits: Path, path_brickwork_state_table: Path
) -> None:
    path_brickwork_state_table.mkdir()
    for path_circuit in tqdm(list(path_circuits.glob("*.qasm"))):
        with Path(path_circuit).open() as f:
            circuit = read_qasm(f)
            layers = transpile_to_layers(circuit)
            table_float = layers_to_measurement_table(layers)
            table_str = [
                [format_angle(angle / math.pi) for angle in column]
                for column in table_float
            ]
            target = (path_brickwork_state_table / path_circuit.name).with_suffix(
                ".json"
            )
            with target.open("w") as f_target:
                json.dump(table_str, f_target)


if __name__ == "__main__":
    convert_circuit_directory_to_brickwork_state_table(
        Path("pages/circuits"), Path("pages/brickwork_state_table")
    )
