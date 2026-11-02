"""Run the XCAT simulation, reconstructions and comparison figures."""
from workflow import (
    generate_xcat, simulate_simind, prepare_data,
    reconstruct,
    evaluate_results, plot_slices, plot_metrics,
)


if __name__ == '__main__':
    gates = (8, 16, 32, 64, 128, 256)

    generate_xcat()
    simulate_simind()
    for gate_count in gates:
        prepare_data(gate_count)  # Bin the projections for both gating schemes and calibrate the gain.
        reconstruct(gate_count, "dip", "static")
        reconstruct(gate_count, "dip", "dynamic")
        reconstruct(gate_count, "osem", "static")
        reconstruct(gate_count, "osem", "dynamic")
        reconstruct(gate_count, "bsrem", "static")
        reconstruct(gate_count, "bsrem", "dynamic")
        evaluate_results(gate_count)  # Compute image metrics and extract axial slices.
    plot_slices(gates)
    plot_metrics(gates)
