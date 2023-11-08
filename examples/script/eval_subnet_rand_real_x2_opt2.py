import subprocess
from torchpack.utils.logging import logger


if __name__ == '__main__':
    pres = ['python',
            'examples/eval.py',
            'examples/configs/mnist/four0123/eval/x2/real/opt2/300.yml',
            '--jobs=4',
            '--run-dir']
    with open('logs/sfsuper/eval_subnet_rand_x2_real_300_opt2_2.txt',
              'w') as \
            wfid:
        for blk in range(7, 9):
            for rand in range(4):
                exp = f'runs/mnist.four0123.train.baseline' \
                      f'.u3cu3_s0.subnet.blk{blk}_rand' \
                      f'{rand}/'
                logger.info(f"running command {pres + [exp]}")

                subprocess.call(pres + [exp], stderr=wfid)
