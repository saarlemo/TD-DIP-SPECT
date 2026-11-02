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
    method = 'td_dip' if method == 'dip' else method
    study = DATA / f'{gates}gates' / mode / f'gates_{gates}'
    output = RESULTS / mode / f'gates_{gates}' / method
    run(f'reconstructions/{mode}_{method}.py', study, output, '--gates', gates)


def evaluate_results(gates):
    study = DATA / f'{gates}gates' / 'static' / f'gates_{gates}'
    for mode, steps in (('static', 3000), ('dynamic', 100000)):
        result = RESULTS / mode / f'gates_{gates}'
        checkpoint = result / 'td_dip' / f'{mode}_td_dip_step{steps:06d}.pt'
        run('plotting/evaluate.py', '--study', study, '--mode', mode,
            '--osem', result / 'osem', '--bsrem', result / 'bsrem',
            '--dip', checkpoint, '--masks',
            RESULTS / 'static' / f'gates_{gates}' / 'evaluation' / 'truth_masks.npz',
            '--output', result / 'evaluation')


def plot_results(gates, kind):
    for mode in ('static', 'dynamic'):
        evaluations = [RESULTS / mode / f'gates_{n}' / 'evaluation' for n in gates]
        run('plotting/plot.py', *evaluations, '--output', RESULTS / 'figures' / mode,
            '--kind', kind)


def plot_slices(gates):
    plot_results(gates, 'slices')


def plot_metrics(gates):
    plot_results(gates, 'metrics')
