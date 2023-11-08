import subprocess
from torchpack.utils.logging import logger
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--supernet', type=str)
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()

    pres = ['python',
            'examples/eval.py',
            'examples/configs/mnist/four0123/eval/x2/noise/opt2/valid_500.yml',
            '--run-dir=runs/mnist.four0123.train.super'
            f'.{args.supernet}',
            '--ckpt.name',
            'checkpoints/step-18400.pt',
            f'--gpu={args.gpu}',
            '--dataset.split=valid']
    with open(f'logs/sfsuper/eval_subnet_noise_x2_opt2_ratio_500'
              f'insuper_{args.supernet}.txt',
              'w') as \
            wfid:
        for blk in range(1, 9):
            # for ratio in ['0', '0.25', '0.5', '0.75', '1']:
            for ratio in ['0', '0.3', '0.6', '1']:
                exp = f"--model.arch.sample_arch=blk{blk}_ratio{ratio}"
                logger.info(f"running command {pres + [exp]}")

                subprocess.run(pres + [exp], stderr=wfid)

        # for blk in range(1, 9):
        #     for rand in range(4):
        #         exp = f"--model.arch.sample_arch=sharefront0_blk{blk}_ran" \
        #               f"d{rand}"
        #         logger.info(f"running command {pres + [exp]}")
        #
        #         subprocess.run(pres + [exp], stderr=wfid)
