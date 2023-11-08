import subprocess
from torchpack.utils.logging import logger
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--name', type=str)

    parser.add_argument('--device', type=str)
    parser.add_argument('--noise', type=str)
    parser.add_argument('--hub', type=str, default=None)
    args = parser.parse_args()

    pres = ['python',
            'examples/eval.py',
            f'examples/configs/'
            f'{args.dataset}/{args.name}/eval/'
            f'{args.device}/real/opt2/noancilla/300_loadop_s18400.yml',
            '--jobs=5',
            '--verbose',
            f'--hub={args.hub}',
            '--run-dir']

    with open(f'logs/{args.device}/{args.dataset}.'
              f'{args.name}.addactreadnoise.u3cu3_0.n2b2'
              f'.txt',
              'a') as \
            wfid:
        for noise in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
            exp = f'runs/{args.dataset}.{args.name}.train.addnoise.' \
                  f'lnorm.actreadnoi.{args.device}.' \
                  f'u3cu3_0.n2b2.noi{noise}.noquant.default'

            logger.info(f"running command {pres + [exp]}")
            subprocess.call(pres + [exp], stderr=wfid)
