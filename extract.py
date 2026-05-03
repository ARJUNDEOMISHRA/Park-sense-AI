import json
import sys

def main():
    notebook_path = sys.argv[1]
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    code = ""
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = "".join(cell.get('source', []))
            code += source + "\n\n"
            
    with open('extracted_code.py', 'w', encoding='utf-8') as f:
        f.write(code)

if __name__ == "__main__":
    main()
