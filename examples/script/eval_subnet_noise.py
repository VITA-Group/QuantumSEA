import subprocess
from torchpack.utils.logging import logger


if __name__ == '__main__':
    pres = ['python',
            'examples/eval.py',
            'examples/configs/mnist/four0123/eval/x2/noise/opt2/all.yml',
            '--run-dir']
    with open('eval_subnet_noise2.txt', 'w') as wfid:
        for blk in range(1, 9):
            for ratio in ['0', '0.25', '0.5', '0.75', '1']:
                exp = f'runs/mnist.four0123.train.baseline' \
                      f'.super4digit_arbitrary_fc1.blk8s1.blk{blk}_ratio' \
                      f'{ratio}/'
                logger.info(f"running command {pres + [exp]}")

                subprocess.call(pres + [exp], stderr=wfid)
