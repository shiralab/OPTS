import json
import random
import re
from pathlib import Path

import numpy as np
import torch

from popt.data import Prompt
from popt.output import ResultWriter, setup_log
from popt.setting.meta_prompt_template import apet, de, ga


class Optimizer:
    def __init__(
        self,
        prompt_designing_llm,
        meta_prompt_template,
        model_param,
        evaluator,
        step_size,
        pop_size,
        initial_mode="para_topk",
        use_pts=False,
        pts_operator=None,
        output_file="",
        log_file="",
        cache_path="",
    ):
        self.prompt_designing_llm = prompt_designing_llm
        self.meta_prompt = meta_prompt_template
        self.model_param = model_param
        self.evaluator = evaluator
        self.ea_step_size = step_size
        self.ea_pop_size = pop_size
        self.initial_mode = initial_mode
        self.use_opts = use_pts
        self.result_writer = ResultWriter(output_file, sheet="result")
        self.logger = setup_log(log_name="opt.log", log_path=log_file)
        self.cache_path = cache_path
        self.opts_type = pts_operator

    def setting2log(self):
        self.logger.info("evol algorithm:" + self.evol_name)
        self.logger.info(
            "prompt generation LLM: " + self.prompt_designing_llm.model_name
        )
        self.logger.info(
            "prompt generation LLM option: " + json.dumps(self.model_param)
        )
        self.logger.info("OPTS LLM: " + self.opts_type.opts_model.model_name)
        self.logger.info("OPTS LLM option: " + json.dumps(self.opts_type.model_param))
        self.logger.info("prompt evaluate LLM: " + self.evaluator.task_model.model_name)
        self.logger.info(
            "prompt evaluate LLM option: " + json.dumps(self.evaluator.model_param)
        )
        self.logger.info("population size: %s", self.ea_pop_size)
        self.logger.info("step_size: %s", self.ea_step_size)
        self.logger.info("opts name: " + self.opts_type.opts_name)
        self.logger.info("opts param: %s", self.opts_type.opts_param)
        self.logger.info("opts tmplate:\n" + self.opts_type.template_opts)

    def paraphrase(self, prompt):
        meta_prompt_paraphrase = "Generate a variation of the following instruction while keeping the semantic meaning.\nInput:<input>\nOutput:"
        request = meta_prompt_paraphrase.replace("<input>", prompt.get_user())
        para_instr = self.prompt_designing_llm.query(
            requests=[request], **self.model_param
        )[0]

        return para_instr

    def initialize(self, prompts):
        self.logger.info("------------------------Initialize------------------------")
        evaluated_prompts = self.evaluator.evaluate(offspring_prompts=prompts, step=0)
        sorted_prompts = sorted(evaluated_prompts, reverse=True, key=lambda x: x.score)
        if self.initial_mode == "para_topk":
            self.logger.info("initial mode: para_topk")
            if self.ea_pop_size % 2 != 0:
                init_k_pop = self.ea_pop_size // 2 + 1
            else:
                init_k_pop = self.ea_pop_size // 2
            if self.cache_path is not None:
                cache_path = Path(self.cache_path)
                prompt_file = cache_path / Path("para_prompt.txt")
                top_k_prompts = sorted_prompts[:init_k_pop]
                if not prompt_file.exists():
                    if not prompt_file.parent.exists():
                        prompt_file.parent.mkdir(parents=True)

                    para_prompts = []
                    for top_k_prompt in top_k_prompts:
                        para_instr = self.paraphrase(top_k_prompt)
                        para_prompts.append(Prompt(instr=para_instr))

                    for saved_prompt in para_prompts:
                        Prompt.write2txt(
                            prompt_file=prompt_file, prompt=saved_prompt.get_user()
                        )
                else:
                    self.logger.info("load cache")
                    para_prompts = Prompt.create_from_userbody(prompt_file)
            else:
                top_k_prompts = sorted_prompts[:init_k_pop]
                para_prompts = []
                for top_k_prompt in top_k_prompts:
                    para_instr = self.paraphrase(top_k_prompt)
                    para_prompts.append(Prompt(instr=para_instr))

            init_prompts = top_k_prompts + para_prompts
            init_prompts = init_prompts[: self.ea_pop_size]

            evaluated_prompts = []
            for init_prompt in init_prompts:
                if init_prompt.score is np.nan:
                    init_prompt = self.evaluator.evaluate(
                        offspring_prompts=[init_prompt], step=0
                    )[0]
                evaluated_prompts.append(init_prompt)

            evaluated_prompts = sorted(
                evaluated_prompts, reverse=True, key=lambda x: x.score
            )
            str_prompts = "\n".join(
                [f"Prompt{i}: " + p.get_user() for i, p in enumerate(evaluated_prompts)]
            )
            self.logger.info("Initial prompts\n" + str_prompts)

            return evaluated_prompts

        elif self.initial_mode == "topk":
            self.logger.info("initial mode: topk")
            evaluated_prompts = sorted_prompts[: self.ea_pop_size]

            str_prompts = "\n".join(
                [f"Prompt{i}: " + p.get_user() for i, p in enumerate(evaluated_prompts)]
            )
            self.logger.info("Initial prompts\n" + str_prompts)

            return evaluated_prompts

    def prompt_generate(
        self,
        request,
        sys_request="",
        start_tag="<prompt>",
        end_tag="</prompt>",
        max_try=3,
    ):
        count = max_try
        ptn = f"{start_tag}.*?{end_tag}"
        self.logger.info("Request for generation prompts:\n" + request)
        while True:
            try:
                if sys_request == "":
                    response = self.prompt_designing_llm.query(
                        request, **self.model_param
                    )[0]
                else:
                    response = self.prompt_designing_llm.query(
                        requests=request, sys_request=sys_request, **self.model_param
                    )[0]

                self.logger.info("Response:\n" + response)
                if start_tag != "" and end_tag != "":
                    offspring = re.findall(ptn, response, flags=re.DOTALL)[-1]
                    offspring = offspring.removeprefix(start_tag).removesuffix(end_tag)
                else:
                    offspring = response
                offspring = offspring.strip()
                self.logger.info("Offspring Prompt: " + offspring)
                break
            except Exception:
                count -= 1
                if count < 0:
                    raise
        return offspring


class EvoPromptGA_OPTS(Optimizer):
    def __init__(
        self,
        prompt_designing_llm,
        model_param,
        evaluator,
        step_size,
        pop_size,
        initial_mode="para_topk",
        use_opts=False,
        opts_type=None,
        output_file="",
        log_file="",
        cache_path="",
        meta_prompt_template="default",
    ):
        if meta_prompt_template == "default":
            meta_prompt_template = ga.meta_prompt
        super().__init__(
            prompt_designing_llm,
            meta_prompt_template,
            model_param,
            evaluator,
            step_size,
            pop_size,
            initial_mode,
            use_opts,
            opts_type,
            output_file,
            log_file,
            cache_path,
        )
        self.evol_name = "EvoPromptGA_OPTS"
        self.setting2log()

    def roulette(self, parents_prompts, pop_size):
        fitness = np.array([prompt.score for prompt in parents_prompts])
        if np.all(fitness == 0):
            fitness[:] = 1

        parents_ids = np.random.choice(
            np.arange(pop_size), size=pop_size, replace=True, p=fitness / fitness.sum()
        ).tolist()
        return [parents_prompts[i] for i in parents_ids]

    def optimize(self, prompt):
        # Initialize
        current_prompts = self.initialize(prompt)
        scores = [p.score for p in current_prompts]
        self.logger.info(f"Initial score: {scores}")

        # output file
        self.result_writer.write2excel(
            sheet="Initial",
            title="Initial",
            header=["Prompt", "Score"],
            contents=[[prompt.get_user(), prompt.score] for prompt in current_prompts],
        )
        self.result_writer.write2excel(
            sheet="result",
            title="result",
            header=["Step", "Best score"],
            contents=[["Initial", scores[0]]],
        )

        # Optimization iteration
        for step in range(1, self.ea_step_size + 1):
            try:
                self.logger.info(
                    f"-------------------------Step{step}------------------------\n"
                )
                evaluated_prompts = []
                parents_set = self.roulette(current_prompts, self.ea_pop_size)
                for i in range(self.ea_pop_size):
                    self.logger.info(f"Generate offspring {i + 1}")
                    parents = random.sample(parents_set, 2)
                    self.logger.info("Parent Prompt 1: " + parents[0].get_user())
                    self.logger.info("Parent Prompt 2: " + parents[1].get_user())

                    request = self.meta_prompt.replace(
                        "<prompt1>", parents[0].get_user()
                    ).replace("<prompt2>", parents[1].get_user())
                    offspring = self.prompt_generate(request)
                    offspring = Prompt(instr=offspring)

                    if self.use_opts:
                        self.logger.info("Prompt design strategy selection")
                        opts_offspring = self.opts_type.mutation(
                            prompts=[offspring], step=step
                        )[0]
                    else:
                        opts_offspring = offspring

                    self.logger.info("Evaluation")
                    evaluated_offspring = self.evaluator.evaluate(
                        offspring_prompts=[opts_offspring], step=step
                    )[0]
                    evaluated_prompts.append(evaluated_offspring)

                    if (
                        self.use_opts
                        and self.opts_type.opts_name != "OPTS_APET"
                        and self.opts_type.opts_name != "OPTS_US"
                    ):
                        p_score_list = [p.score for p in parents]
                        p_max_score = max(p_score_list)
                        r = evaluated_offspring.score - p_max_score
                        self.opts_type.update_param(
                            evaluated_offspring,
                            parents_max_score=p_max_score,
                            offspring_score=evaluated_offspring.score,
                        )
                    else:
                        r = np.nan

                    self.result_writer.record2excel(
                        step=step,
                        parents=[p.get_user() for p in parents],
                        offspring=offspring.get_user(),
                        ptr=evaluated_offspring.select_arm,
                        applied_offspring=opts_offspring.get_user(),
                        score=evaluated_offspring.score,
                        increse=r,
                    )
                    self.logger.info(f"Score: {evaluated_offspring.score}")

            except Exception:
                self.logger.exception("Optimization Error")
                return None

            sorted_prompts = sorted(
                list(set(current_prompts + evaluated_prompts)),
                reverse=True,
                key=lambda x: x.score,
            )
            current_prompts = sorted_prompts[: self.ea_pop_size]

            scores = [p.score for p in current_prompts]
            self.logger.info(f"Scores: {scores}")
            self.logger.info("Best prompt: " + current_prompts[0].get_user())
            self.result_writer.write2excel(
                sheet="result", contents=[[f"Step{step}", scores[0]]]
            )
            self.result_writer.write2excel(
                sheet=f"step{step}_result",
                header=["selected_prompt", "score"],
                contents=[[p.get_user(), p.score] for p in current_prompts],
            )

        torch.cuda.empty_cache()
        return current_prompts[0]


class EvoPromptDE_OPTS(Optimizer):
    def __init__(
        self,
        prompt_designing_llm,
        model_param,
        evaluator,
        step_size,
        pop_size,
        initial_mode="para_topk",
        use_opts=False,
        opts_type=None,
        output_file="",
        log_file="",
        cache_path="",
        meta_prompt_template="default",
    ):
        if meta_prompt_template == "default":
            meta_prompt_template = de.meta_prompt
        super().__init__(
            prompt_designing_llm,
            meta_prompt_template,
            model_param,
            evaluator,
            step_size,
            pop_size,
            initial_mode,
            use_opts,
            opts_type,
            output_file,
            log_file,
            cache_path,
        )
        self.evol_name = "EvoPromptDE_OPTS"
        self.setting2log()

    def optimize(self, prompt):
        current_prompts = self.initialize(prompt)
        scores = [p.score for p in current_prompts]
        self.logger.info(f"Initial score: {scores}")

        self.result_writer.write2excel(
            sheet="Initial",
            title="Initial",
            header=["Prompt", "Score"],
            contents=[[prompt.get_user(), prompt.score] for prompt in current_prompts],
        )
        self.result_writer.write2excel(
            sheet="result",
            title="result",
            header=["Step", "Best score"],
            contents=[["Initial", scores[0]]],
        )

        for step in range(1, self.ea_step_size + 1):
            try:
                self.logger.info(
                    f"-------------------------Step{step}------------------------\n"
                )
                new_prompts = []
                for i in range(self.ea_pop_size):
                    self.logger.info(f"Generate child {i + 1}")
                    base = current_prompts[i]
                    candidates = [
                        current_prompts[j] for j in range(self.ea_pop_size) if j != i
                    ]
                    a, b, c = np.random.choice(candidates, 3, replace=False)
                    c = current_prompts[0]

                    self.logger.info("Base Prompt: " + base.get_user())
                    self.logger.info("Prompt a: " + a.get_user())
                    self.logger.info("Prompt b: " + b.get_user())
                    self.logger.info("Prompt c: " + c.get_user())

                    request = self.meta_prompt.replace("<prompt0>", base.get_user())
                    request = request.replace("<prompt1>", a.get_user())
                    request = request.replace("<prompt2>", b.get_user())
                    request = request.replace("<prompt3>", c.get_user())

                    offspring = self.prompt_generate(request)
                    offspring = Prompt(instr=offspring)

                    if self.use_opts:
                        self.logger.info("Prompt design strategy selection")
                        opts_offspring = self.opts_type.mutation(
                            prompts=[offspring], step=step
                        )[0]
                    else:
                        opts_offspring = offspring

                    self.logger.info("Evaluation")
                    evaluated_offspring = self.evaluator.evaluate(
                        offspring_prompts=[opts_offspring], step=step
                    )[0]

                    if (
                        self.use_opts
                        and self.opts_type.opts_name != "OPTS_APET"
                        and self.opts_type.opts_name != "OPTS_US"
                    ):
                        base_prompt_score = base.score
                        r = evaluated_offspring.score - base_prompt_score
                        self.opts_type.update_param(
                            evaluated_offspring,
                            parents_max_score=base_prompt_score,
                            offspring_score=evaluated_offspring.score,
                        )
                    else:
                        r = np.nan

                    self.result_writer.record2excel(
                        step=step,
                        parents=[p.get_user() for p in [base, a, b, c]],
                        offspring=offspring.get_user(),
                        ptr=evaluated_offspring.select_arm,
                        applied_offspring=opts_offspring.get_user(),
                        score=evaluated_offspring.score,
                        increse=r,
                    )

                    if evaluated_offspring.score > base.score:
                        new_prompt = evaluated_offspring
                    else:
                        new_prompt = base
                    new_prompts.append(new_prompt)

                current_prompts = new_prompts
                current_prompts = sorted(
                    current_prompts, reverse=True, key=lambda x: x.score
                )

                scores = [p.score for p in current_prompts]
                self.logger.info(f"Scores: {scores}")
                self.logger.info("Best prompt: " + current_prompts[0].get_user())
                self.result_writer.write2excel(
                    sheet="result", contents=[[f"Step{step}", scores[0]]]
                )
                self.result_writer.write2excel(
                    sheet=f"step{step}_result",
                    header=["selected_prompt", "score"],
                    contents=[[p.get_user(), p.score] for p in current_prompts],
                )

            except Exception:
                self.logger.exception("Optimization Error")
                return None

        torch.cuda.empty_cache()
        return current_prompts[0]


class APET(Optimizer):
    def __init__(
        self,
        prompt_designing_llm,
        model_param,
        evaluator,
        step_size,
        pop_size,
        initial_mode="para_topk",
        use_opts=False,
        opts_type=None,
        output_file="",
        log_file="",
        cache_path="",
        meta_prompt_template="default",
    ):
        if meta_prompt_template == "default":
            meta_prompt_template = apet.meta_prompt_template
            self.meta_prompt_sys = apet.meta_prompt_sys
        super().__init__(
            prompt_designing_llm,
            meta_prompt_template,
            model_param,
            evaluator,
            step_size,
            pop_size,
            initial_mode,
            use_opts,
            opts_type,
            output_file,
            log_file,
            cache_path,
        )
        self.evol_name = "APET"
        self.setting2log()

    def optimize(self, prompt):
        if isinstance(prompt, list):
            prompt = prompt[0]
        else:
            prompt = prompt

        request = self.meta_prompt.replace("<input>", prompt.get_user())
        offspring = self.prompt_generate(
            request=request, sys_request=self.meta_prompt_sys, start_tag="", end_tag=""
        )
        offspring = Prompt(instr=offspring)

        return offspring
