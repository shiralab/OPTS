meta_prompt = """\
Please follow the instruction step-by-step to generate a better prompt.  
1. Crossover the following prompts to generate a new prompt:  
Prompt 1: Your task is to classify the comment as one of the following categories: terrible, bad, okay, good, great.
Prompt 2: In this task, you are given sentences from movie reviews. The task is to classify a sentence as one of the following categories: terrible, bad, okay, good, great.
2. Mutate the prompt generated in Step 1 and generate a final prompt bracketed with <prompt> and </prompt>.

1. Crossover Prompt: In this task, you are given comments from movie reviews. Your task is to classify each comment as one of the following categories: terrible, bad, okay, good, great.
2. <prompt>Given a sentence from a movie review, classify it into one of the following categories: terrible, bad, okay, good, or great.</prompt>

Please follow the instruction step-by-step to generate a better prompt.
1. Crossover the following prompts and generate a new prompt:
Prompt 1: <prompt1>
Prompt 2: <prompt2>
2. Mutate the prompt generated in Step 1 and generate a final prompt bracketed with <prompt> and </prompt>.

1. """
