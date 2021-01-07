import nbformat as nbf
import os
from nbconvert.preprocessors import ExecutePreprocessor
from shutil import copyfile, rmtree
import glob

def create_notebook(file, filename):
    nb = nbf.v4.new_notebook()

    text = "#LONG SHORT TERM MEMORY"

    code1 = "from app import extract, resample, train, gen\nfile = '" + file + "'\nfilename = '" + filename + "'"
    

    code2 = "extract(file, filename)"

    code3 = "resample(file, filename)"

    code4 = "train(file, filename)"

    code5 = "gen(file, filename)"
    

    nb['cells'] = [nbf.v4.new_markdown_cell(text),
                nbf.v4.new_code_cell(code1),
                nbf.v4.new_code_cell(code2),               
                nbf.v4.new_code_cell(code3),
                nbf.v4.new_code_cell(code4),   
              #  nbf.v4.new_code_cell(code5),
                ]
                
    #nbf.write(nb, '../lstm.ipynb')
    ep = ExecutePreprocessor(timeout=-1, kernel_name='python3')
    ep.preprocess(nb, {'metadata': {'path': ''}})
    with open('lstm.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

    i = 1
    while(i < 100):
        if not glob.glob('uploads/html/' + filename + '_' + str(i) + '.html'):
            break
        i += 1

    os.system('jupyter nbconvert --to html ' + 'lstm.ipynb ' + 'lstm.html')
    copyfile('lstm.html', 'uploads/html/' + filename + '_' + str(i) + '.html')
    rmtree('uploads/' + filename + "/")