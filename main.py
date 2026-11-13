"""Run the XCAT simulation, reconstructions and comparison figures."""
from workflow import (
    generate_xcat, simulate_simind, prepare_data,
    reconstruct,
    evaluate_results, plot_slices, plot_metrics,
)


if __name__ == '__main__':
    gates = (8, 16, 32, 64, 128, 256)
    static_dip_gates = (8, 16, 32)
    dynamic_dip_gates = (8, 16)

    generate_xcat()
    simulate_simind()
    for gate_count in gates:
        prepare_data(gate_count)  # Bin the projections for both gating schemes and calibrate the gain.
        reconstruct(gate_count, "td_dip", "static")
        reconstruct(gate_count, "td_dip", "dynamic")
        if gate_count in static_dip_gates:
            reconstruct(gate_count, "dip", "static")
        if gate_count in dynamic_dip_gates:
            reconstruct(gate_count, "dip", "dynamic")
        reconstruct(gate_count, "osem", "static")
        reconstruct(gate_count, "osem", "dynamic")
        reconstruct(gate_count, "bsrem", "static")
        reconstruct(gate_count, "bsrem", "dynamic")
        dip_modes = tuple(
            mode for mode, coverage in (
                ("static", static_dip_gates), ("dynamic", dynamic_dip_gates)
            ) if gate_count in coverage
        )
        evaluate_results(gate_count, dip_modes=dip_modes)  # Compute metrics and extract axial slices.
    plot_slices(gates)
    plot_metrics(gates)
