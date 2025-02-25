import random
import re

import numpy as np

from .data import Prompt
from .output import ResultWriter, setup_log


class OPTS:
    def __init__(self, opts_model, model_param, opts_info, opts_param, log_file):
        self.opts_model = opts_model
        self.model_param = model_param
        self.opts_info = opts_info
        self.opts_param = opts_param
        self.opts_logger = setup_log(log_path=log_file, log_name="opt_log")

    def mutation(
        self, prompts, step, start_tag="<prompt>", end_tag="</prompt>", max_try=3
    ):
        pass

    def create_meta_prompt(self, prompt, tag="<input>"):
        meta_prompt = self.template_opts.replace(tag, prompt)
        return meta_prompt

    def prompt_generate(
        self,
        meta_prompt,
        sys_prompt="",
        start_tag="<prompt>",
        end_tag="</prompt>",
        max_try=3,
    ):
        if sys_prompt == "":
            self.opts_logger.info("Meta-Prompt: " + meta_prompt)
        else:
            self.opts_logger.info("System Prompt: " + sys_prompt)
            self.opts_logger.info("Meta-Prompt: " + meta_prompt)

        count = max_try
        ptn = f"{start_tag}.*?{end_tag}"
        while True:
            try:
                if sys_prompt == "":
                    raw_child = self.opts_model.query(
                        requests=meta_prompt, **self.model_param
                    )[0]
                else:
                    raw_child = self.opts_model.query(
                        requests=meta_prompt, sys_request=sys_prompt, **self.model_param
                    )[0]

                self.opts_logger.info("Response: " + raw_child)

                if start_tag != "" and end_tag != "":
                    child = re.findall(ptn, raw_child, flags=re.DOTALL)[-1]
                    child = child.removeprefix(start_tag).removesuffix(end_tag)
                else:
                    child = raw_child
                child = child.strip()
                break
            except Exception:
                count -= 1
                if count < 0:
                    raise

        return child


class OPTS_TS(OPTS):
    def __init__(
        self,
        opts_model,
        model_param,
        opts_info,
        opts_param,
        log_file,
        bandit_log_folder,
        bandit_data_file,
    ):
        super().__init__(opts_model, model_param, opts_info, opts_param, log_file)
        self.opts_name = "OPTS_TS"
        self.bandit_log_folder = bandit_log_folder
        self.t = 0
        self.bandit_data_witer = ResultWriter(
            path=bandit_data_file, sheet=f"bandit{self.t + 1}"
        )

        self.strategies = self.opts_info["strategy"]
        self.sys_prompt = self.opts_info["meta_prompt_sys"]
        self.template_opts = self.opts_info["meta_prompt_template"]
        self.arms_num = len(self.strategies) + 1

        self.strategies_param = []
        for i in range(self.arms_num):
            strategy_param = {"a": 1, "b": 1}
            self.strategies_param.append(strategy_param)

    def get_bandit_logger(self, step):
        log_name = f"bandit_t{step}"
        log_file = self.bandit_log_folder + f"/badit_step{step}.log"
        bandit_logger = setup_log(log_name=log_name, log_path=log_file)
        return bandit_logger

    def cal_param(self, prompt):
        p_list = []
        for i in range(self.arms_num):
            p = np.random.beta(
                self.strategies_param[i]["a"], self.strategies_param[i]["b"]
            )
            p_list.append({"arm": i, "p": p})
            self.bandit_data_witer.write2excel(
                sheet=f"bandit{self.t}",
                header=["arm", "prompt", "a", "b", "bandit value"],
                contents=[
                    [
                        i,
                        prompt.get_user(),
                        self.strategies_param[i]["a"],
                        self.strategies_param[i]["b"],
                        p,
                    ]
                ],
            )

        return p_list

    def select_arm(self, p_list):
        sorted_p_list = sorted(p_list, reverse=True, key=lambda x: x["p"])
        select_p_list = [sorted_p_list[0]]
        for p in sorted_p_list[1:]:
            if p["p"] == select_p_list[0]["p"]:
                select_p_list.append(p)
            else:
                break

        if len(select_p_list) == 1:
            return select_p_list[0]
        else:
            select_p = random.sample(select_p_list, 1)[0]
            return select_p

    def mutation(
        self, prompts, step, start_tag="<prompt>", end_tag="</prompt>", max_try=3
    ):
        self.t += 1
        self.bandit_logger = self.get_bandit_logger(step)
        self.bandit_logger.info(
            f"-------------------------Bandit t{self.t} Start-------------------------\n"
        )
        modified_prompts = []
        for prompt in prompts:
            try:
                p_list = self.cal_param(prompt=prompt)
                self.opts_logger.info(f"Bandit values: {p_list}")
                p_max = self.select_arm(p_list=p_list)

                prompt.select_arm = p_max["arm"]
                self.bandit_logger.info(f"Select Arm: {prompt.select_arm}\n")

                if p_max["arm"] == 0:
                    self.opts_logger.info("Prompt design strategy: NOT SELECT")
                    modified_prompts.append(prompt)
                else:
                    arm_id = p_max["arm"]
                    self.opts_logger.info(f"Prompt design strategy: {arm_id}")
                    self.opts_logger.info("Prompt: " + prompt.get_user())
                    strategy = self.strategies[p_max["arm"] - 1]
                    meta_prompt = self.template_opts.replace(
                        "<strategy>", strategy
                    ).replace("<input>", prompt.get_user())
                    modified_prompt = self.prompt_generate(
                        meta_prompt=meta_prompt,
                        sys_prompt=self.sys_prompt,
                        start_tag="",
                        end_tag="",
                    )
                    modified_prompt = Prompt(modified_prompt)
                    modified_prompt.select_arm = prompt.select_arm
                    modified_prompts.append(modified_prompt)
            except Exception:
                self.opts_logger.exception("opts Error")
                modified_prompts.append(prompt)

        return modified_prompts

    def update_param(self, prompt, parents_max_score, offspring_score):
        if offspring_score > parents_max_score:
            r = 1
        else:
            r = 0

        arm = prompt.select_arm
        self.strategies_param[arm]["a"] += r
        self.strategies_param[arm]["b"] += 1 - r

        updated_a = self.strategies_param[arm]["a"]
        updated_b = self.strategies_param[arm]["b"]

        self.bandit_logger.info(f"Arm{arm} Reward: {r}")
        self.bandit_logger.info(f"Arm{arm} Updated a: {updated_a}")
        self.bandit_logger.info(f"Arm{arm} Updated b: {updated_b}")
        self.bandit_logger.info(
            f"-------------------------Bandit t{self.t} End-------------------------\n"
        )

        self.bandit_data_witer.write2excel(
            sheet=f"bandit{self.t}",
            header=["selected_arm", "reward"],
            contents=[[arm, r]],
        )


class OPTS_APET(OPTS):
    def __init__(
        self,
        opts_model,
        model_param,
        opts_info,
        opts_param,
        log_file,
        bandit_log_folder,
        bandit_data_file,
    ):
        super().__init__(opts_model, model_param, opts_info, opts_param, log_file)
        self.opts_name = "OPTS_APET"
        self.sys_prompt = self.opts_info["meta_prompt_sys"]
        self.template_opts = self.opts_info["meta_prompt_template"]

    def judge_mutation(self, base_p):
        select_flag = np.random.choice(
            np.array([True, False]),
            size=1,
            replace=False,
            p=np.array([base_p, 1 - base_p]),
        ).tolist()

        return select_flag[0]

    def mutation(self, prompts, step, start_tag="", end_tag="", max_try=3):
        modified_prompts = []
        for prompt in prompts:
            try:
                if self.judge_mutation(self.opts_param):
                    self.opts_logger.info("Prompt design strategy: SELECT")
                    self.opts_logger.info("Prompt: " + prompt.get_user())
                    meta_prompt = self.template_opts.replace(
                        "<input>", prompt.get_user()
                    )
                    modified_prompt = self.prompt_generate(
                        meta_prompt=meta_prompt,
                        sys_prompt=self.sys_prompt,
                        start_tag="",
                        end_tag="",
                    )
                    modified_prompt = Prompt(instr=modified_prompt)
                    modified_prompt.select_arm = 1
                    modified_prompts.append(modified_prompt)
                else:
                    self.opts_logger.info("Prompt design strategy: NOT SELECT")
                    prompt.select_arm = 0
                    modified_prompts.append(prompt)
            except Exception:
                self.opts_logger.exception("opts Error")
                modified_prompts.append(prompt)

        return modified_prompts


class OPTS_US(OPTS):
    def __init__(
        self,
        opts_model,
        model_param,
        opts_info,
        opts_param,
        log_file,
        bandit_log_folder,
        bandit_data_file,
    ):
        super().__init__(opts_model, model_param, opts_info, opts_param, log_file)
        self.opts_name = "OPTS_US"
        self.strategies = self.opts_info["strategy"]
        self.sys_prompt = self.opts_info["meta_prompt_sys"]
        self.template_opts = self.opts_info["meta_prompt_template"]
        self.arms_num = len(self.strategies) + 1

        self.strategies_param = np.ones(self.arms_num)
        params = self.strategies_param / self.strategies_param.sum()
        self.opts_logger.info(f"arm prob: {params}")
        prob_sum = self.strategies_param.sum()
        self.opts_logger.info(f"prob sum: {prob_sum}")

    def arm_select(self):
        select_flag = np.random.choice(
            np.arange(self.arms_num),
            size=1,
            replace=False,
            p=self.strategies_param / self.strategies_param.sum(),
        ).tolist()

        return select_flag[0]

    def mutation(self, prompts, step, start_tag="", end_tag="", max_try=3):
        modified_prompts = []
        for prompt in prompts:
            try:
                selected = self.arm_select()
                prompt.select_arm = selected
                if selected == 0:
                    self.opts_logger.info("Prompt design strategy: NOT SELECT")
                    modified_prompts.append(prompt)
                else:
                    arm_id = selected
                    self.opts_logger.info(f"Prompt design strategy: {arm_id}")
                    self.opts_logger.info("Prompt: " + prompt.get_user())
                    strategy = self.strategies[arm_id - 1]
                    meta_prompt = self.template_opts.replace(
                        "<strategy>", strategy
                    ).replace("<input>", prompt.get_user())
                    modified_prompt = self.prompt_generate(
                        meta_prompt=meta_prompt,
                        sys_prompt=self.sys_prompt,
                        start_tag="",
                        end_tag="",
                    )
                    modified_prompt = Prompt(modified_prompt)
                    modified_prompt.select_arm = prompt.select_arm
                    modified_prompts.append(modified_prompt)
            except Exception:
                self.opts_logger.exception("opts Error")
                modified_prompts.append(prompt)

        return modified_prompts
