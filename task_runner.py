import os
import subprocess
import sys
from tqdm import tqdm
import threading
import argparse
import unicodedata
import re


def parse_args(args):
    parser = argparse.ArgumentParser(description='Simple settings.')
    parser.add_argument('-g', '--gpus', nargs="+", help='E.g. 0 0 1 1', required=True)
    parser.add_argument('-i', '--in-file', required=True)
    parser.add_argument('-o',
                        '--out-file',
                        default=None,
                        help='If None generated automatically',
                        required=False)
    parser.add_argument('-l',
                        '--log-dir',
                        default=None,
                        help='If None no log wil be created',
                        required=False)
    return parser.parse_args(args)


def read_tasks(in_file):
    if not os.path.isfile(in_file):
        raise FileNotFoundError(f"{in_file} not found")

    with open(in_file, 'r') as f:
        tasks = set(f.read().splitlines())

    return set(tasks)


def remove_tasks_done(tasks, out_file):
    if not os.path.isfile(out_file):
        return tasks

    with open(out_file, 'r') as f:
        tasks_done = set(f.read().splitlines())

    return tasks.difference(tasks_done)


def slugify(value, allow_unicode=False):
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')


def process_commands(tasks, gpu, out_file, log_dir, pbar):
    while len(tasks):
        try:
            task = tasks.pop()
            orig_task = task
            if log_dir is not None:
                file_name = slugify(task)
                file_name = os.path.join(log_dir, file_name)
                redirect_str = f"> {file_name}.out 2>&1"
            else:
                redirect_str = "> /dev/null 2>&1"
            task = f"CUDA_VISIBLE_DEVICES={gpu} {task} {redirect_str}"
            process = subprocess.Popen(task, shell=True, stdout=subprocess.DEVNULL)
            process.wait()
            if process.returncode == 0:
                with open(out_file, 'a') as f:
                    f.write(f"{orig_task}\n")
            else:
                print(f"Warning: Task ran into error: {task}")
            pbar.update(1)
        except IndexError:
            print("No task left")


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    conf = parse_args(args)
    tasks = read_tasks(conf.in_file)
    if conf.out_file is not None:
        out_file = conf.out_file
    else:
        name, ext = os.path.splitext(conf.in_file)
        out_file = f"{name}_done{ext}"
    tasks = remove_tasks_done(tasks, out_file)

    if not len(tasks):
        print('Every task is done already.')
        return

    if conf.log_dir is not None:
        os.makedirs(conf.log_dir, exist_ok=True)

    pbar = tqdm(total=len(tasks))
    threads = []
    for gpu_id in conf.gpus:
        t = threading.Thread(target=process_commands,
                             args=(tasks, gpu_id, out_file, conf.log_dir, pbar))
        t.start()
        threads.append(t)


if __name__ == '__main__':
    main()