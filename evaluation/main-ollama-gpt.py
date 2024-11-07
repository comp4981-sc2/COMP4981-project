from openai import AzureOpenAI
from datasets import load_dataset
import unittest
import sys
import io
from contextlib import redirect_stdout
import ast
import traceback
from termcolor import cprint, colored
from langchain_community.llms import Ollama

# act as .env file
# dont commit expose your key
AZURE_API_KEY = "YOUR_AZURE_API_KEY"

class TestResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = 0
        self.failure_messages = []

def create_prompt(example):
    """Creates code generation prompt from class specification."""
    imports = '\n'.join(example['import_statement']) if isinstance(example['import_statement'], list) else example['import_statement']
    
    methods_details = []
    for method in example['methods_info']:
        methods_details.append(
            f"Method: {method['method_name']}\n"
            f"Description: {method['method_description']}\n"
            f"Parameters: {method.get('parameters', 'None')}\n"
            f"Returns: {method.get('return_type', 'None')}"
        )
    
    prompt = (
        f"Create Python class '{example['class_name']}'\n\n"
        f"Description:\n{example['class_description']}\n\n"
        f"Imports:\n{imports}\n\n"
        f"Constructor:\n{example['class_constructor']}\n\n"
        f"Methods:\n" + "\n\n".join(methods_details) + "\n\n"
        "Provide complete implementation code only, no explanations or markdown."
    )
    
    return prompt

def validate_python_code(code):
    """Validate if the code is syntactically correct Python code."""
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)

def clean_generated_code(code):
    """Clean and format the generated code."""
    code = code.strip()
    
    if code.startswith("```python"):
        code = code[len("```python"):].strip()
    if code.startswith("```"):
        code = code[3:].strip()
    if code.endswith("```"):
        code = code[:-3].strip()
    
    lines = code.split('\n')
    cleaned_lines = []
    for line in lines:
        cleaned_line = line.replace('\t', '    ').rstrip()
        cleaned_lines.append(cleaned_line)
    
    return '\n'.join(cleaned_lines)

def generate_code_gpt4(prompt, client):
    print("Generating code with GPT-4...")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a Python code generation assistant. Generate complete, working Python code based on the given class description and requirements. Include all necessary imports and method implementations. Provide only the code without any markdown formatting or explanations."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=2000
    )
    code = response.choices[0].message.content
    return clean_generated_code(code)

def generate_code_ollama(prompt, client):
    print("Generating code with Ollama...")

    messages=[
        {"role": "system", "content": "You are a Python code generation assistant. Generate complete, working Python code based on the given class description and requirements. Include all necessary imports and method implementations. Provide only the code without any markdown formatting or explanations."},
        {"role": "user", "content": prompt}
    ],

    response = client.invoke(messages)
        
    cprint("Response from Ollama:" + response, 'green')
   
    return clean_generated_code(response)

def run_tests(generated_code, test_code, import_statements):
    test_result = TestResult()
    
    namespace = {}
    
    try:
        # Add unittest import to namespace
        exec("import unittest", namespace)
        exec("import datetime", namespace)  # Add datetime import which is used in tests
        
        # Execute the generated code
        exec(generated_code, namespace)
        
        # Execute the test code directly
        exec(test_code, namespace)
        
        # Create test suite
        suite = unittest.TestSuite()
        
        # Add all test classes to the suite
        for name, obj in namespace.items():
            if isinstance(obj, type) and issubclass(obj, unittest.TestCase) and obj != unittest.TestCase:
                suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(obj))
        
        print(f"\nNumber of tests in suite: {suite.countTestCases()}")
               # Run the tests
        stream = io.StringIO()
        runner = unittest.TextTestRunner(stream=stream, verbosity=2)
        result = runner.run(suite)
        
        # Print the captured output
        print("\nTest Output:")
        print(stream.getvalue())
        
        test_result.passed = result.testsRun - len(result.failures) - len(result.errors)
        test_result.failed = len(result.failures)
        test_result.errors = len(result.errors)
        
        for failure in result.failures:
            test_result.failure_messages.append(failure[1])
        for error in result.errors:
            test_result.failure_messages.append(error[1])
            
    except Exception as e:
        test_result.errors += 1
        test_result.failure_messages.append(f"Execution error: {str(e)}\n{traceback.format_exc()}")
        print(f"\nExecution error occurred:\n{str(e)}")
        print(traceback.format_exc())
        
    return test_result

def print_task_accuracies(results):
    cprint("\n" + "="*50, "cyan")
    cprint("TASK-BY-TASK AND OVERALL ACCURACY", "cyan")
    cprint("="*50, "cyan")
    
    total_passed = 0
    total_tests = 0
    
    for i, result in enumerate(results):
        task_total_tests = (result['test_results']['passed'] + 
                          result['test_results']['failed'] + 
                          result['test_results']['errors'])
        
        task_accuracy = (result['test_results']['passed'] / task_total_tests * 100) if task_total_tests > 0 else 0
        
        cprint(f"Task {i}: {task_accuracy:.2f}%", "yellow")
        cprint(f"- Class: {result['class_name']}", "yellow")
        cprint(f"- Passed: {result['test_results']['passed']}", "green")
        cprint(f"- Failed: {result['test_results']['failed']}", "red")
        cprint(f"- Errors: {result['test_results']['errors']}", "red")
        cprint(f"- Total Tests: {task_total_tests}", "yellow")
        cprint("-" * 30, "cyan")
        
        # Accumulate totals for overall accuracy
        total_passed += result['test_results']['passed']
        total_tests += task_total_tests
    
    overall_accuracy = (total_passed / total_tests * 100) if total_tests > 0 else 0
    cprint("\n" + "="*50, "cyan")
    cprint(f"OVERALL ACCURACY: {overall_accuracy:.2f}%", "cyan")
    cprint(f"Total Passed Tests: {total_passed}", "green")
    cprint(f"Total Tests: {total_tests}", "yellow")
    cprint("="*50, "cyan")
    
def display_tests_result(dataset, client, generate_code_func):
    results = []
    for i, example in enumerate(dataset):
        cprint(f"\n{'='*50}", 'cyan')
        cprint(f"Processing Example {i+1}", 'cyan')
        cprint(f"{'='*50}", 'cyan')
        cprint(f"Class Name: {example['class_name']}", 'yellow')
        cprint(f"Class Description: {example['class_description']}", 'yellow')
        
        prompt = create_prompt(example)
        generated_code = generate_code_func(prompt, client)
        
        cprint("\nGenerated Code:", 'green')
        cprint("-" * 40, 'green')
        cprint(generated_code, 'green')
        cprint("-" * 40, 'green')
        
        with open(f"debug_{example['class_name']}_gpt4.py", "w") as f:
            f.write(generated_code)
        
        cprint("\nRunning tests...", 'blue')
        test_result = run_tests(generated_code, example['test'], example['import_statement'])
        
        # Print test results immediately
        cprint("\nTest Results:", 'magenta')
        cprint(f"Passed: {test_result.passed}", 'magenta')
        cprint(f"Failed: {test_result.failed}", 'magenta')
        cprint(f"Errors: {test_result.errors}", 'magenta')
        
        if test_result.failure_messages:
            cprint("\nFailure Messages:", 'red')
            for msg in test_result.failure_messages:
                cprint(f"- {msg}", 'red')
        
        results.append({
            'class_name': example['class_name'],
            'description': example['class_description'],
            'generated_code': generated_code,
            'test_results': {
                'passed': test_result.passed,
                'failed': test_result.failed,
                'errors': test_result.errors,
                'failure_messages': test_result.failure_messages
            }
        })
    
    # Print summary of all results
    cprint("\n" + "="*50, 'cyan')
    cprint("FINAL SUMMARY", 'cyan')
    cprint("="*50, 'cyan')
    for result in results:
        cprint(f"\nClass: {result['class_name']}", 'yellow')
        cprint(f"Tests Passed: {result['test_results']['passed']}", 'yellow')
        cprint(f"Tests Failed: {result['test_results']['failed']}", 'yellow')
        cprint(f"Test Errors: {result['test_results']['errors']}", 'yellow')
        if result['test_results']['failure_messages']:
            cprint("Failure Messages:", 'red')
            for msg in result['test_results']['failure_messages']:
                cprint(f"- {msg}", 'red')
    
    # Whole Evaluation Summary
    print_task_accuracies(results)

# ===========================================
# ===========================================
# ===========================================

# Define the model name constant
model_list = [
    "deepseek-coder-v2:16b",  # 8.9GB
    "llama3.2:1b",            # 1.3GB
    "qwen2.5-coder:1.5b",     # 986MB
]

def get_model_name(num: int) -> str:
    if 1 <= num <= len(model_list):
        return model_list[num - 1]
    else:
        return "qwen2.5-coder:1.5b"  # Default model if the number is out of range
    
def select_model(use_local: bool):
    if use_local:
        selected_model = get_model_name(1)
        client = Ollama(model=selected_model)
        cprint("Selected model: " + colored(client.model, "green"))
        return client
    else:
        client = AzureOpenAI(
            api_key=AZURE_API_KEY,
            api_version="2024-06-01",
            azure_endpoint="https://hkust.azure-api.net"
        )
        cprint("Using Azure OpenAI client", "yellow")
        return client


def main():
    try:
        # Toggle between local and GPT model
        use_local = True  # local mean using ollama
        
        # Define the client
        client = select_model(use_local)
        if client is None:
            return
        
        # Define the dataset
        dataset = load_dataset("FudanSELab/ClassEval", split="test[:20]")
        
        # Run the evaluation
        if use_local:
            display_tests_result(dataset, client, generate_code_ollama)
        else:
            display_tests_result(dataset, client, generate_code_gpt4)
            
    except Exception as e:
        cprint(f"An error occurred: {str(e)}", 'red')
        cprint(traceback.format_exc(), 'red')
        raise

if __name__ == "__main__":
    main()