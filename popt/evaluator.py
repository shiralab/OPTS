import re

from tqdm import tqdm

from popt.output import setup_log


class Evaluator:
    def __init__(self, task_model, model_param, dev_x, dev_y, batch_size, log_folder):
        self.task_model = task_model
        self.model_param = model_param
        self.batch_size = batch_size
        self.batched_dev_x = self.batch_split(dev_x)
        self.batched_dev_y = self.batch_split(dev_y)
        self.log_folder = log_folder
        self.eval_logger = None

    def gat_eval_logger(self, step):
        log_name = f"eval_step{step}"
        log_path = self.log_folder + f"/eval_step{step}.log"
        eval_logger = setup_log(log_name=log_name, log_path=log_path, stream=False)

        return eval_logger

    def batch_split(self, data_list):
        batched_dataset = []
        batch = []
        count = 0
        for i, elem in enumerate(data_list):
            batch.append(elem)
            count += 1
            if count >= self.batch_size or i >= len(data_list) - 1:
                batched_dataset.append(batch)
                count = 0
                batch = []

        return batched_dataset

    def task_execute(self, prompt):
        outputs = []
        targets = []
        src_list = []
        first = True
        for src_batch, tgt_batch in tqdm(
            zip(self.batched_dev_x, self.batched_dev_y), total=len(self.batched_dev_x)
        ):
            requests = []
            for src in src_batch:
                requests.append(prompt.join_input(text=src))

            if first:
                self.eval_logger.info("Request:\n" + requests[0] + "\n")
                first = False

            output_batch = self.task_model.query(requests, **self.model_param)
            outputs.extend(output_batch)
            targets.extend(tgt_batch)
            src_list.extend(src_batch)

        return outputs, targets, src_list

    def evaluate(self, offspring_prompts, step):
        pass


class BBHeval(Evaluator):
    def __init__(
        self,
        task_model,
        model_param,
        dev_x,
        dev_y,
        batch_size,
        log_folder,
        extract_ans_type="(?<=the answer is )(.*)(?=.)",
    ):
        super().__init__(task_model, model_param, dev_x, dev_y, batch_size, log_folder)
        self.extract_ans_type = extract_ans_type

    def cal_score(self, outputs, targets):
        count = 0
        for output, target in zip(outputs, targets):
            # print(f"{output}_{target}")
            if output == target:
                count += 1

        score = count / len(targets)
        return score

    # Refer to lm-evaluation-harness
    def extract_output(
        self, outputs, group_select=0, match_str="(?<=the answer is )(.*)(?=.)"
    ):
        formated_output_list = []
        for output in outputs:
            hit = re.findall(match_str, output)
            if hit:
                hit = hit[group_select]
                if isinstance(hit, tuple):
                    hit = [h for h in hit if h][0]
                hit = hit.strip()
            else:
                hit = output
            formated_output_list.append(hit)

        return formated_output_list

    def evaluate(self, offspring_prompts, step):
        # Setup log file
        self.eval_logger = self.gat_eval_logger(step)
        evaluated_prompts = []
        for prompt in offspring_prompts:
            self.eval_logger.info(
                "-------------------------Evaluate Start-------------------------\n"
            )
            # Solve task with the task-solving LLM
            outputs, targets, src_list = self.task_execute(prompt)
            # Extract answer parts
            clean_outputs = self.extract_output(
                outputs, match_str=self.extract_ans_type
            )
            # Calculate accuracy
            score = self.cal_score(clean_outputs, targets)
            # Set score for prompt
            prompt.score = score
            evaluated_prompts.append(prompt)
            # Print LLM response
            for i in range(len(outputs)):
                question_index = i + 1
                self.eval_logger.info(
                    f"--------------------Question {question_index}--------------------\n"
                )
                self.eval_logger.info("Question:\n" + src_list[i] + "\n")
                self.eval_logger.info("Response:\n" + outputs[i] + "\n")
                self.eval_logger.info("Clean response:\n" + clean_outputs[i] + "\n")
                self.eval_logger.info("Target: " + targets[i] + "\n")
            self.eval_logger.info(f"Score: {prompt.score}")
            self.eval_logger.info(
                "-------------------------Evaluate End-------------------------\n"
            )

        return evaluated_prompts
