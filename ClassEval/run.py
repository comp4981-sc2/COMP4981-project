import subprocess
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def run_command(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(f"Error running command: {command}\n{stderr.decode('utf-8')}")
    else:
        print(f"Command ran successfully: {command}\n{stdout.decode('utf-8')}")

# Define variables
data_path = "./data/ClassEval_data.json"
output_path = None  # Set to None to use custom output path maker
max_length = 2048
sample = 3
temperature = 0.2
cuda = "0 1"
openai_base = os.getenv("OPENAI_BASE")
openai_key = os.getenv("OPENAI_KEY")

# Define the models and generation strategies
models = {
    16: "GPT-4o-mini",
    17: "DeepSeekCoder-V2"
}
generation_strategies = {
    0: "H",
    # 1: "I",
    # 2: "C"
}
greedy_strategies = {
    0: f"pass{sample}",
    1: "g"
}

# Base command
base_command = "python generation/inference.py"

# Add optional arguments if provided
if data_path:
    base_command += f" --data_path {data_path}"
if max_length:
    base_command += f" --max_length {max_length}"
if sample:
    base_command += f" --sample {sample}"
if temperature:
    base_command += f" --temperature {temperature}"
if cuda:
    base_command += f" --cuda {cuda}"
if openai_base:
    base_command += f" --openai_base {openai_base}"
if openai_key:
    base_command += f" --openai_key {openai_key}"

# Function to create custom output path
def create_output_path(model, gen_mode, greedy):
    model_name = models[model]
    gen_mode_name = generation_strategies[gen_mode]
    greedy_name = greedy_strategies[greedy]
    return f"output/{model_name}_{gen_mode_name}_{greedy_name}.json"

# Iterate over models and generation strategies
for model in models:
    for gen_mode in generation_strategies:
        for greedy in greedy_strategies:
            custom_output_path = create_output_path(model, gen_mode, greedy)
            command = f"{base_command} --model {model} --generation_strategy {gen_mode} --greedy {greedy} --output_path {custom_output_path}"
            print(f"Running command: {command}")
            run_command(command)
            
            # Run evaluation
            eval_command = f"python classeval_evaluation/evaluation.py --source_file_name {custom_output_path} --greedy {greedy} --eval_data ClassEval_data"
            print(f"Running evaluation: {eval_command}")
            run_command(eval_command)