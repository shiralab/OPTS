import json
import random
from pathlib import Path

import numpy as np
import torch

from popt.data import Data, Prompt
from popt.evaluator import BBHeval
from popt.llm import setup_llm
from popt.optimizer import APET, EvoPromptDE_OPTS, EvoPromptGA_OPTS
from popt.opts import OPTS_APET, OPTS_TS, OPTS_US
from popt.setting.meta_prompt_template import apet, apet_selection

EVAL_LIST: dict = {"BBHeval": BBHeval}
EVOL_ALGO_LIST = {
    "EvoPromptGA_OPTS": EvoPromptGA_OPTS,
    "EvoPromptDE_OPTS": EvoPromptDE_OPTS,
    "APET": APET,
}
OPTS_TYPE = {
    "OPTS_APET": OPTS_APET,
    "OPTS_TS": OPTS_TS,
    "OPTS_US": OPTS_US,
}
OPTS_INFO = {
    "OPTS_APET": {"default": apet.meta_prompt_info},
    "OPTS_TS": {"default": apet_selection.meta_prompt_info},
    "OPTS_US": {"default": apet_selection.meta_prompt_info},
}


def run(
    # LLM
    model_evol={"type": "openai", "name": "gpt-4o-mini-2024-07-18"},
    model_evol_options=None,
    model_evol_template="default",
    model_opts={"type": "openai", "name": "gpt-4o-mini-2024-07-18"},
    model_opts_options=None,
    model_opts_template="default",
    model_task={"type": "vllm", "name": "meta-llama/Meta-Llama-3-8B-Instruct"},
    model_task_options={"max_tokens": 1024, "temperature": 0.0},
    # Optimizer
    evol_algo="EvoPromptDE_OPTS",
    opts_type="OPTS_TS",
    step_size=50,
    pop_size=10,
    initial_mode="para_topk",
    use_opts=True,
    opts_param=None,
    # seed
    seed=None,
    # Evaluator
    evaluator="BBHeval",
    batch_size=50,
    dev_file=None,
    # Dataset option
    dataset_option={"labels": [], "label_replace": False, "delimiter": "\t"},
    # Initial prompt
    prompt_template_file=None,
    prompt_user_file=None,
    # Output folder
    output_folder=None,
    cache_path=None,
):
    # Create output folder
    output_folder_path = Path(output_folder)
    if not output_folder_path.exists():
        output_folder_path.mkdir(parents=True)
    ## Create evaluation log folder in output folder
    eval_log_file = output_folder + "/eval_log"
    eval_log_path = Path(output_folder + "/eval_log")
    if not eval_log_path.exists():
        eval_log_path.mkdir()
    ## Create log optimizatioin log file in output folder
    opt_log_file = output_folder + "/opt.log"
    output_file = output_folder + "/outputs.xlsx"
    result_prompt_file = output_folder + "/result_prompt.txt"
    result_template_file = output_folder + "/result_template.txt"
    args_file = output_folder + "/args.json"
    if opts_type != "" or opts_type != "OPTS_APET":
        bandit_log_folder = output_folder + "/badit_log"
        bandit_log_folder_path = Path(output_folder + "/badit_log")
        if not bandit_log_folder_path.exists():
            bandit_log_folder_path.mkdir()
        bandit_data_file = output_folder + "/bandit_data.xlsx"
    else:
        bandit_log_folder = ""
        bandit_data_file = ""

    write_dict = {
        "model_evol": model_evol,
        "model_evol_options": model_evol_options,
        "model_evol_template": model_evol_template,
        "model_opts": model_opts,
        "model_opts_options": model_opts_options,
        "model_opts_template": model_opts_template,
        "model_task": model_task,
        "model_task_options": model_task_options,
        "evol_algo": evol_algo,
        "opts_type": opts_type,
        "step_size": step_size,
        "pop_size": pop_size,
        "initial_mode": initial_mode,
        "use_opts": use_opts,
        "opts_param": opts_param,
        "seed": seed,
        "evaluator": evaluator,
        "batch_size": batch_size,
        "dev_file": dev_file,
        "dataset_option": dataset_option,
        "prompt_template_file": prompt_template_file,
        "prompt_user_file": prompt_user_file,
        "output_folder": output_folder,
        "cache_path": cache_path,
    }
    with open(args_file, "w") as f:
        json.dump(write_dict, f, indent=4)

    optimizer_class = EVOL_ALGO_LIST[evol_algo]
    evaluator_class = EVAL_LIST[evaluator]
    opts_class = OPTS_TYPE[opts_type]
    opts_info = OPTS_INFO.get(opts_type, {}).get(model_opts_template)

    ## Set seed
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    ## LLM
    llm_evol = setup_llm(**model_evol, seed=seed)
    if model_evol == model_opts:
        llm_pts = llm_evol
    else:
        llm_pts = setup_llm(**model_opts, seed=seed)
    if model_evol == model_task:
        llm_task = llm_evol
    elif model_opts == model_task:
        llm_task = llm_pts
    else:
        llm_task = setup_llm(**model_task, seed=seed)

    ## OPTS mechanism
    opts_operator = opts_class(
        opts_model=llm_pts,
        model_param=model_opts_options,
        opts_info=opts_info,
        opts_param=opts_param,
        log_file=opt_log_file,
        bandit_log_folder=bandit_log_folder,
        bandit_data_file=bandit_data_file,
    )
    ## Evaluator
    devset = Data.load(dev_file, **dataset_option)
    evaluator = evaluator_class(
        task_model=llm_task,
        model_param=model_task_options,
        dev_x=devset.get_x(),
        dev_y=devset.get_y(),
        batch_size=batch_size,
        log_folder=eval_log_file,
    )
    ## Optimizer
    optimizer = optimizer_class(
        prompt_designing_llm=llm_evol,
        meta_prompt_template=model_evol_template,
        model_param=model_evol_options,
        evaluator=evaluator,
        step_size=step_size,
        pop_size=pop_size,
        initial_mode=initial_mode,
        use_opts=use_opts,
        opts_type=opts_operator,
        output_file=output_file,
        log_file=opt_log_file,
        cache_path=cache_path,
    )
    ## Initial prompts
    Prompt.set_template(prompt_template_file)
    init_prompts = Prompt.create_from_userbody(prompt_user_file)

    # Optimize
    final_prompt = optimizer.optimize(prompt=init_prompts)
    if final_prompt is not None:
        Prompt.write2txt(
            prompt_file=result_prompt_file,
            prompt=final_prompt.get_user(),
            template_file=result_template_file,
            template=final_prompt.template,
        )
        return final_prompt.instr, final_prompt.template
    else:
        return None


def test(
    model_task={"type": "vllm", "name": "meta-llama/Meta-Llama-3-8B-Instruct"},
    model_task_options={"max_new_tokens": 512},
    instruction="",
    prompt_template="default",
    evaluator="BBHeval",
    test_path="",
    dataset_option={"labels": [], "label_replace": False, "delimiter": "\t"},
    batch_size=50,
    output_folder="",
    seed=5,
):
    # Create file
    test_log_folder = output_folder + "/test_log"
    test_log_folder_path = Path(test_log_folder)
    if not test_log_folder_path.exists():
        test_log_folder_path.mkdir()

    # Setup task-solving model
    llm_task = setup_llm(**model_task, seed=seed)

    # Load test set
    testset = Data.load(test_path, **dataset_option)
    # Evaluator
    test_evaluator = EVAL_LIST[evaluator](
        task_model=llm_task,
        model_param=model_task_options,
        dev_x=testset.get_x(),
        dev_y=testset.get_y(),
        batch_size=batch_size,
        log_folder=test_log_folder,
    )

    # Tested prompt
    prompt = Prompt(instr=instruction, template=prompt_template)
    print(prompt.get_user())

    # Test prompt
    prompt_evaluated = test_evaluator.evaluate([prompt], step="test")[0]
    print(prompt_evaluated.score)
    test_result_file = Path(output_folder) / Path("test_result.txt")
    with open(test_result_file, "w") as f:
        f.write(str(prompt_evaluated.score))
