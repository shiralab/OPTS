import argparse
import json
from pathlib import Path

import popt

parser = argparse.ArgumentParser(description="run test")

parser.add_argument("--config_file")
parser.add_argument("--seed", type=int)
parser.add_argument("--test_file")
parser.add_argument("--final_prompt_file")
parser.add_argument("--prompt_template_file")
parser.add_argument("--output_folder")

args = parser.parse_args()

with open(args.config_file, "r") as f:
    config = json.load(f)

with open(args.final_prompt_file, "r") as f:
    final_instr = f.read()
final_instr = final_instr.strip()

with open(args.prompt_template_file, "r") as f:
    final_template = f.read()
final_template = final_template.strip()

output_folder_path = Path(args.output_folder)
if not output_folder_path.exists():
    output_folder_path.mkdir(parents=True)

# test
popt.test(
    model_task=config["model_task"],
    model_task_options=config["model_task_options"],
    instruction=final_instr,
    prompt_template=final_template,
    evaluator=config["evaluator"],
    test_path=args.test_file,
    batch_size=config["batch_size"],
    output_folder=args.output_folder,
    seed=args.seed,
)
