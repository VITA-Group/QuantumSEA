import subprocess
from torchpack.utils.logging import logger
import sys

if __name__ == '__main__':
    pres = ['python',
            'examples/eval.py',
            'examples/configs/mnist/four0123/eval/tq/all.yml',
            '--run-dir=runs/mnist.four0123.train.super'
            '.u3cu3_s0.plain.blk8s1_ws1_os1',
            '--ckpt.name',
            'checkpoints/step-18400.pt',
            '--dataset.split=valid']
    with open('logs/sfsuper/eval_subnet_tq_insuper_u3cu3_s0_plain_blk8s1_ws1_os1.txt',
              'w') as wfid:
        for blk in range(1, 9):
            for rand in range(4):
                exp = f"--model.arch.sample_arch=sharefront0_bl" \
                      f"k{blk}_rand{rand}"
                logger.info(f"running command {pres + [exp]}")

                subprocess.run(pres + [exp], stderr=wfid)
