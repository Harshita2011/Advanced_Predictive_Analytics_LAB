"""Convert every Jupyter Notebook (.ipynb) in the LAB_03 directory into a executable Python (.py) script."""
import json
import os


def notebook_to_python(nb_path, py_path):
    with open(nb_path, encoding='utf-8') as f:
        nb = json.load(f)

    lines = []
    lines.append(f'"""Auto-generated Python script from {os.path.basename(nb_path)}."""\n')

    for cell in nb.get('cells', []):
        if cell.get('cell_type') != 'code':
            continue
        source = ''.join(cell.get('source', []))
        if not source.strip():
            continue
        lines.append('\n# %%\n')
        lines.append(source)
        if not source.endswith('\n'):
            lines.append('\n')

    with open(py_path, 'w', encoding='utf-8') as f:
        f.write(''.join(lines))


def main():
    folder = os.path.dirname(os.path.abspath(__file__))
    for name in os.listdir(folder):
        if name.endswith('.ipynb'):
            nb_path = os.path.join(folder, name)
            py_path = os.path.join(folder, name.replace('.ipynb', '.py'))
            print(f'Converting {name} -> {os.path.basename(py_path)}')
            notebook_to_python(nb_path, py_path)


if __name__ == '__main__':
    main()
