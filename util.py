import os


def setup_dirs(base):
    path_templates = ['./{base}/', './{base}/logs/', './{base}/images/']
    for path_template in path_templates:
        path = path_template.format(base=base)
        if not os.path.exists(path):
            print('Creating directory', path)
            os.makedirs(path)
