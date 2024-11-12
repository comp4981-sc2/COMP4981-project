from openai import AzureOpenAI
from datasets import load_dataset
import unittest
import sys
import io
from contextlib import redirect_stdout
import ast
import traceback

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

def calculate_class_accuracy(test_results):
    total_tests = test_results['passed'] + test_results['failed'] + test_results['errors']
    if total_tests == 0:
        return 0
    return (test_results['passed'] / total_tests) * 100

def print_task_accuracies(results):
    print("\n" + "="*50)
    print("TASK-BY-TASK AND OVERALL ACCURACY")
    print("="*50)
    
    total_passed = 0
    total_tests = 0
    
    for i, result in enumerate(results):
        task_total_tests = (result['test_results']['passed'] + 
                          result['test_results']['failed'] + 
                          result['test_results']['errors'])\\\\\
        
        task_accuracy = (result['test_results']['passed'] / task_total_tests * 100) if task_total_tests > 0 else 0
        
        print(f"Task {i}: {task_accuracy:.2f}%")
        print(f"- Class: {result['class_name']}")
        print(f"- Passed: {result['test_results']['passed']}")
        print(f"- Failed: {result['test_results']['failed']}")
        print(f"- Errors: {result['test_results']['errors']}")
        print(f"- Total Tests: {task_total_tests}")
        print("-" * 30)
        
        # Accumulate totals for overall accuracy
        total_passed += result['test_results']['passed']
        total_tests += task_total_tests
    
    overall_accuracy = (total_passed / total_tests * 100) if total_tests > 0 else 0
    print("\n" + "="*50)
    print(f"OVERALL ACCURACY: {overall_accuracy:.2f}%")
    print(f"Total Passed Tests: {total_passed}")
    print(f"Total Tests: {total_tests}")
    print("="*50)

def main():
    try:
        client = AzureOpenAI(
            api_key="",
            api_version="2024-06-01",
            azure_endpoint="https://hkust.azure-api.net"
        )
        
        dataset = load_dataset("FudanSELab/ClassEval", split="test[:20]")
        results = []
        
        for i, example in enumerate(dataset):
            print(f"\n{'='*50}")
            print(f"Processing Example {i+1}")
            print(f"{'='*50}")
            print(f"Class Name: {example['class_name']}")
            print(f"Class Description: {example['class_description']}")
            
            prompt = create_prompt(example)
            generated_code = generate_code_gpt4(prompt, client)
            
            print("\nGenerated Code:")
            print("-" * 40)
            print(generated_code)
            print("-" * 40)
            
            with open(f"debug_{example['class_name']}_gpt4.py", "w") as f:
                f.write(generated_code)
            
            print("\nRunning tests...")
            test_result = run_tests(generated_code, example['test'], example['import_statement'])
            
            # Print test results immediately
            print("\nTest Results:")
            print(f"Passed: {test_result.passed}")
            print(f"Failed: {test_result.failed}")
            print(f"Errors: {test_result.errors}")
            
            if test_result.failure_messages:
                print("\nFailure Messages:")
                for msg in test_result.failure_messages:
                    print(f"- {msg}")
            
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
        print("\n" + "="*50)
        print("FINAL SUMMARY")
        print("="*50)
        for result in results:
            print(f"\nClass: {result['class_name']}")
            print(f"Tests Passed: {result['test_results']['passed']}")
            print(f"Tests Failed: {result['test_results']['failed']}")
            print(f"Test Errors: {result['test_results']['errors']}")
            if result['test_results']['failure_messages']:
                print("Failure Messages:")
                for msg in result['test_results']['failure_messages']:
                    print(f"- {msg}")
        
        # Print task accuracies and overall accuracy
        print_task_accuracies(results)
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()