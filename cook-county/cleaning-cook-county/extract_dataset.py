import nbformat
import nbconvert.preprocessors
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps_to_run', default='1,2,3,4,5', type=str)
    args = parser.parse_args()

    data_dir = os.environ.get('DATA_DIR', None)
    if data_dir is None or not os.path.exists(data_dir):
        raise ValueError("Please set DATA_DIR")


    nb_files_in_exec_order = [
        'cook-county-intro.ipynb',        # preprocessing
        'cook-county-annual.ipynb',       # export annual data
        'cook-county-semiannual.ipynb',   # export semiannual data
        'cook-county-quarterly.ipynb',    # export quarterly data
        'cook-county-SVI.ipynb',          # adds social vulnerability features
        ]

    args.steps_to_run = [int(nn) for nn in args.steps_to_run.split(',')]
    for nn, nb_file_path in enumerate(nb_files_in_exec_order):
        if (nn+1) not in args.steps_to_run:
            print("Skipping nb per user request: %s" % (nb_file_path))
            continue

        with open(nb_file_path, 'r') as f:
            nb = nbformat.read(f, as_version=4)

        ep = nbconvert.preprocessors.ExecutePreprocessor(
            timeout=300, # timeout is 5 minutes
            kernel_name='python3')

        print('Running nb: ' + nb_file_path)
        ep.preprocess(nb)
        with open(nb_file_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)

