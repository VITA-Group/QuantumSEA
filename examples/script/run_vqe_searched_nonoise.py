import subprocess
from torchpack.utils.logging import logger
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--name', type=str)
    parser.add_argument('--space', type=str)
    args = parser.parse_args()

    pres = ['python',
            'examples/eval.py',
            f'examples/configs/'
            f'{args.dataset}/{args.name}/eval/x2/real/opt2/all.yml',
            '--jobs=5',
            '--run-dir']

    with open(f'logs/nonoise/{args.dataset}.{args.name}.nonoise.txt', 'a') as \
            wfid:
        for space in [f'u3cu3_s0',
                      f'seth_s0',
                      f'barren_s0',
                      f'farhi_s0',
                      f'maxwell_s0']:

            if 'maxwell' in space:
                n_blk = 4
            else:
                n_blk = 8

            exp = f'runs/{args.dataset}.{args.name}.train.searched.scratch' \
                  f'.nonoise.setting0.{space}.plain.blk{n_blk}s1.1.1'

            logger.info(f"running command {pres + [exp]}")
            subprocess.call(pres + [exp], stderr=wfid)
