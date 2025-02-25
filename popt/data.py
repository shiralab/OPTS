import json
import random
from pathlib import Path

import numpy as np


class Prompt:
    default_template = "Q: <q>\nA: <prompt>\n"
    template = default_template

    def set_template(template_file):
        path = Path(template_file)

        with open(path, "r") as f:
            Prompt.template = f.read()

    def reset_template():
        Prompt.template = Prompt.default_template

    def create_from_userbody(instr_file):
        if isinstance(instr_file, str):
            path = Path(instr_file)
        else:
            path = instr_file

        with open(path, "r") as f:
            lines = f.readlines()

        instrs = [line.rstrip() for line in lines]

        prompts = []
        for instr in instrs:
            prompt = Prompt(instr)
            prompts.append(prompt)

        return prompts

    def write2txt(prompt_file, prompt, template_file=None, template=None):
        if isinstance(prompt_file, str):
            prompt_path = Path(prompt_file)
        else:
            prompt_path = prompt_file
        with open(prompt_path, "a") as f:
            f.write(prompt)
            f.write("\n")

        if template_file is not None and template is not None:
            template_path = Path(template_file)
            with open(template_path, "w") as f:
                f.write(template)

    def __init__(self, instr, template="default"):
        if template == "default":
            self.template = Prompt.template
        else:
            self.template = template

        self.instr = instr
        self.score = np.nan
        self.select_arm = np.nan

    def get_user(self):
        return self.instr

    def join_input(self, text=""):
        prompt = self.template.replace("<prompt>", self.instr).replace("<q>", text)

        return prompt


class Data:
    def load(dataset_path, labels, label_replace=False, delimiter="\t"):
        # Read dataset
        if isinstance(dataset_path, str):
            path = Path(dataset_path)
        else:
            path = dataset_path
        dataset = Data.read_file(path, delimiter)

        if label_replace:
            for i in range(len(dataset)):
                dataset[i][1] = labels[int(dataset[i][1])]

        dataset = Data(dataset)

        return dataset

    def split(dataset_path, dev_num, dev_path, test_path, delimiter="\t"):
        if isinstance(dataset_path, str):
            path = Path(dataset_path)
        else:
            path = dataset_path
        dataset = Data.read_file(path, delimiter)

        dev_set = random.sample(dataset, dev_num)
        test_set = [data for data in dataset if data not in dev_set]

        dev_dataset = Data(dev_set)
        test_dataset = Data(test_set)

        dev_dataset.save_json(dev_path)
        test_dataset.save_json(test_path)

    def read_file(dataset_path: Path, delimiter="\t"):
        if dataset_path.suffix == ".txt":
            with open(dataset_path, "r") as f:
                lines = f.readlines()
            dataset = [line.rstrip() for line in lines]
            dataset = [data.split(delimiter) for data in dataset]

        elif dataset_path.suffix == ".json":
            with open(dataset_path, "r") as f:
                dataset = json.load(f)

            dataset = [[data["input"], data["target"]] for data in dataset["examples"]]
        else:
            dataset = []

        return dataset

    def __init__(self, dataset):
        self.dataset = dataset

    def get_x(self):
        x = [data[0] for data in self.dataset]
        return x

    def get_y(self):
        y = [data[1] for data in self.dataset]
        return y

    def save_json(self, json_path):
        if isinstance(json_path, str):
            path = Path(json_path)
        else:
            path = json_path

        if not path.parent.exists():
            path.parent.mkdir(parents=True)

        write_form = [{"input": data[0], "target": data[1]} for data in self.dataset]
        write_form = {"examples": write_form}

        with open(path, "w") as f:
            json.dump(write_form, f, indent=4)
