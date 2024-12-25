import os
import sqlite3
import re
import ast
import keyword
from datasets import load_dataset, DatasetDict
from wonderwords import RandomWord


# Create the dataset directory if it doesn't exist
os.makedirs('dataset/classeval', exist_ok=True)

# Load and save the dataset locally
dataset = load_dataset("FudanSELab/ClassEval")
dataset.save_to_disk('dataset/classeval')

# Load the dataset from the local files
dataset = DatasetDict.load_from_disk('dataset/classeval')

# Extract the "solution_code" column
solution_codes = dataset['test']['solution_code']

# Define the transformations
def obfuscate_function_names(code):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            node.name = ''.join(chr(ord(c) + 1) if c.isalpha() else c for c in node.name)
    return ast.unparse(tree)

def adversarial_function_names(code):
    r = RandomWord()
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.ClassDef):
            node.name = r.word(include_parts_of_speech=["noun"])
    return ast.unparse(tree)

def remove_code_structure(code):
    keywords = keyword.kwlist
    operators = ['+', '-', '*', '**', '/', '^', '>', '<', '==', '!=', '=', '+=', '-=', '*=', '/=', '%=', '>>', '<<', ':']
    delimiters = [',', '(', ')', '{', '}', '[', ']']

    # Replace Exact keywords and operators
    pattern = r'\b(' + '|'.join(re.escape(kw) for kw in keywords) + r')\b'
    code = re.sub(pattern, ' ', code)

    # # Replace operators, and delimiters with spaces
    # for symbol in operators + delimiters:
    #     code = code.replace(symbol, ' ')

    # Replace operators and delimiters with spaces, but skip string literals
    def replace_operators(match):
        text = match.group(0)
        if text.startswith(('"', "'")):
            return text
        for symbol in operators + delimiters:
            text = text.replace(symbol, ' ')
        return text

    code = re.sub(r'".*?"|\'.*?\'|[^"\']+', replace_operators, code)

    # Remove indentation
    lines = code.split('\n')
    stripped_lines = [line.lstrip() for line in lines]
    code = '\n'.join(stripped_lines)

    # Remove all extra spaces (compactify)
    code = re.sub(r'[ ]+', ' ', code)
    code = re.sub(r'\n{3,}', '\n\n', code)
        
    return code

def remove_function_body(code):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            node.body = []
    return ast.unparse(tree)

# Apply transformations
transformed_codes = {
    'solution_code': solution_codes,
    'obfuscated': [obfuscate_function_names(code) for code in solution_codes],
    'adversarial': [adversarial_function_names(code) for code in solution_codes],
    'no_structure': [remove_code_structure(code) for code in solution_codes],
    'no_body': [remove_function_body(code) for code in solution_codes]
}

# Create a SQLite database and insert the transformed codes
conn = sqlite3.connect('dataset/classeval_transform.sqlite')
c = conn.cursor()

# Create table if it doesn't exist
c.execute('''
CREATE TABLE IF NOT EXISTS transformed_codes (
    id INTEGER PRIMARY KEY,
    solution_code TEXT,
    obfuscated TEXT,
    adversarial TEXT,
    no_structure TEXT,
    no_body TEXT
)
''')

# Delete old entries
c.execute('DELETE FROM transformed_codes')

# Insert data
for i in range(len(solution_codes)):
    c.execute('''
    INSERT INTO transformed_codes (id, solution_code, obfuscated, adversarial, no_structure, no_body) VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        i+1,
        transformed_codes['solution_code'][i],
        transformed_codes['obfuscated'][i],
        transformed_codes['adversarial'][i],
        transformed_codes['no_structure'][i],
        transformed_codes['no_body'][i]
    ))

# Commit and close
conn.commit()
conn.close()

# Print transformed codes for verification
# for key, codes in transformed_codes.items():
#     print(f"=============== Transformation: {key} ===============")
#     for code in codes[:0]:  # Print examples for brevity
#         print(code)
#         print()