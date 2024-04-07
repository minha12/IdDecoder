"""
This file runs the main training/val loop
"""
import os
import json
import sys
import pprint
os.environ["CUDA_VISIBLE_DEVICES"]="0"
sys.path.append(".")
sys.path.append("..")
sys.path.append("...")
from mapper.options.train_options import TrainOptions
from mapper.training.coach_arc_v2 import Coach


def main(opts):
# 	if os.path.exists(opts.exp_dir):
# 		raise Exception('Oops... {} already exists'.format(opts.exp_dir))
	os.makedirs(opts.exp_dir, exist_ok=True)

	opts_dict = vars(opts)
	pprint.pprint(opts_dict)
	with open(os.path.join(opts.exp_dir, 'opt.json'), 'w') as f:
		json.dump(opts_dict, f, indent=4, sort_keys=True)

	coach = Coach(opts)
	coach.train()


if __name__ == '__main__':
	opts = TrainOptions().parse()
	main(opts)
