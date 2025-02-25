import argparse
import json

import popt

parser = argparse.ArgumentParser(description="run popt")

parser.add_argument("--config_file")
parser.add_argument("--seed", type=int)
parser.add_argument("--dev_file")
parser.add_argument("--init_prompt_file")
parser.add_argument("--prompt_template_file")
parser.add_argument("--output_folder")
parser.add_argument("--cache_folder")

args = parser.parse_args()

with open(args.config_file, "r") as f:
    config = json.load(f)

# optimize prompt
final_instr, final_template = popt.run(
    seed=args.seed,
    dev_file=args.dev_file,
    prompt_user_file=args.init_prompt_file,
    prompt_template_file=args.prompt_template_file,
    output_folder=args.output_folder,
    cache_path=args.cache_folder,
    **config,
)

print("optimization end")
print(final_instr)
