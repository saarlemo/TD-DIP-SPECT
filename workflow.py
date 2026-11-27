"""Connect the project scripts and their output paths."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'reconstructions' / 'data'
RESULTS = ROOT / 'results'


def run(script, *arguments):
    subprocess.run([sys.executable, str(ROOT / script), *(str(a) for a in arguments)], check=True)


def generate_xcat():
    run('xcat/simulate_xcat.py')


def simulate_simind():
    run('simind/simulate_simind.py')


def prepare_data(gates):
    run('reconstructions/prepare_data.py', '--gates', gates)


def reconstruct(gates, method, mode):
    study = DATA / f'{gates}gates' / mode / f'gates_{gates}'
    output = RESULTS / mode / f'gates_{gates}' / method
    run(f'reconstructions/{mode}_{method}.py', study, output, '--gates', gates)


def evaluate_results(gates):
    study = DATA / f'{gates}gates' / 'static' / f'gates_{gates}'
    for mode in ('static', 'dynamic'):
        result = RESULTS / mode / f'gates_{gates}'
        osem_visualization = result / 'osem' / 'best_visualization.h5'
        bsrem_visualization = result / 'bsrem' / 'best_visualization.npy'
        dip_visualization = result / 'dip' / 'best_visualization.h5'
        td_dip_visualization = result / 'td_dip' / f'{mode}_td_dip_best_visualization.pt'
        arguments = [
            '--study', study, '--mode', mode,
            '--osem', osem_visualization, '--bsrem', bsrem_visualization,
            '--dip', dip_visualization,
            '--td-dip', td_dip_visualization, '--masks',
            RESULTS / 'static' / f'gates_{gates}' / 'evaluation' / 'truth_masks.npz',
            '--output', result / 'evaluation',
        ]
        run('plotting/evaluate.py', *arguments)


def plot_slices(gates):
    for mode in ('static', 'dynamic'):
        evaluations = [RESULTS / mode / f'gates_{n}' / 'evaluation' for n in gates]
        run('plotting/plot.py', *evaluations, '--output', RESULTS / 'figures' / mode)
