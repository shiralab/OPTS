meta_prompt = """\
Please follow the instruction step-by-step to generate a better prompt.
1. Identify the different parts between the Prompt 1 and Prompt 2:
Prompt 1: Rewrite the input text into simpler text.
Prompt 2: Rewrite my complex sentence in simpler terms, but keep the meaning.
2. Randomly mutate the different parts
3. Combine the different parts with Prompt 3, selectively replace it with the different parts in step 2 and generate a new prompt.
Prompt 3: Rewrite the given input text into simpler English sentences while preserving the same meaning, so it can be understood by non-native English speakers.
4. Crossover the prompt in the step3 with the following basic prompt and generate a final prompt bracketed with <prompt> and </prompt>:
Basic Prompt: Make the sentence easier for people who do not speak English fluently to comprehend.

1. Identifying the different parts between Prompt 1 and Prompt 2:
Prompt 1: Rewrite the input text into simpler text.
Prompt 2: Rewrite my complex sentence in simpler terms, but keep the meaning.
Different parts:
"input text" vs "my complex sentence"
"simpler text" vs "simpler terms, but keep the meaning"

2. Randomly mutate the different parts:
"input text" -> "provided text"
"my complex sentence" -> "the difficult sentence"
"simpler text" -> "easier language"
"simpler terms, but keep the meaning" -> "simpler words while maintaining the meaning"

3. Combine the different parts with Prompt 3, selectively replace it with the different parts in step 2 and generate a new prompt:
Prompt 3: Rewrite the given input text into simpler English sentences while preserving the same meaning, so it can be understood by non-native English speakers.
New Prompt: Transform the provided text into easier language while maintaining the meaning, making it accessible for non-native English speakers.

4. Crossover the prompt in step 3 with the following basic prompt and generate a final prompt bracketed with <prompt> and </prompt>:
Basic Prompt: Make the sentence easier for people who do not speak English fluently to comprehend.
Final Prompt: <prompt>Convert the difficult sentence into simpler words while preserving the meaning, so it's easier for non-native English speakers to understand.</prompt>


Please follow the instruction step-by-step to generate a better prompt.
1. Identify the different parts between the Prompt 1 and Prompt 2:
Prompt 1: <prompt1>
Prompt 2: <prompt2>
2. Randomly mutate the different parts
3. Combine the different parts with Prompt 3, selectively replace it with the different parts in step2 and generate a new prompt.
Prompt 3: <prompt3>
4. Crossover the prompt in the step3 with the following basic prompt and generate a final prompt bracketed with <prompt> and </prompt>:
Basic Prompt: <prompt0>

1. """
