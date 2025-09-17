import argparse
import os
import numpy as np
from utils.data_utils import check_extension, save_dataset


def generate_tsp_data(dataset_size, tsp_size):
    return np.random.uniform(size=(dataset_size, tsp_size, 2)).tolist()


def generate_mec_data(dataset_size, uav_size, dependency=[]):
   
    #task_data = np.random.uniform(size=(dataset_size, uav_size, 1), low=0, high=1000)
    #UAV_start_pos = np.random.randint(size=(dataset_size, 1, 2), low = 0, high = 500)
    #task_position = np.random.uniform(size=(dataset_size, uav_size, 2), low=0, high=500)
    #CPU_circles = np.random.randint(size=(dataset_size, uav_size, 1), low=0, high=1000)
    #IoT_resource = np.random.randint(size=(dataset_size, uav_size, 1), low=100, high=200)
    #UAV_resource = np.max(CPU_circles, axis=1, keepdims=True) // 4
    
    task_position = np.random.uniform(size=(dataset_size, uav_size, 2))
    UAV_start_pos = np.random.uniform(size=(dataset_size, 1, 2))
    task_data = np.random.uniform(size=(dataset_size, uav_size, 1), low=0, high=1)
    CPU_circles = np.random.uniform(size=(dataset_size, uav_size, 1), low=0, high=1)
    IoT_resource = np.random.uniform(size=(dataset_size, uav_size, 1), low=2, high=5)/10
    UAV_resource = np.max(CPU_circles, axis=1, keepdims=True) / 2

    time_window = np.random.uniform(size=(dataset_size, uav_size, 2), low=0, high=10)
    time_window = np.sort(time_window, axis=2)

    dependency = [ i -1 for  i in dependency]

    dep_window = np.take(time_window, indices=dependency, axis=1)
    dep_window = np.sort(dep_window.reshape(dataset_size, -1), axis=1).reshape(dataset_size, len(dependency), 2)
    np.put_along_axis(time_window, np.array(dependency)[None, :, None], dep_window, axis=1)
    time_window[:,:,1] = 1000

    dependency = [ i + 1 for  i in dependency]

    dependencys = [dependency]*dataset_size

    return list(zip(
        task_data.tolist(),
        UAV_start_pos.tolist(),
        task_position.tolist(),
        CPU_circles.tolist(),
        IoT_resource.tolist(),
        UAV_resource.tolist(),
        time_window.tolist(),
        dependencys
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", help="Filename of the dataset to create (ignores datadir)")
    parser.add_argument("--data_dir", default='data', help="Create datasets in data_dir/problem (default 'data')")
    parser.add_argument("--name", type=str, required=True, help="Name to identify dataset")
    parser.add_argument("--problem", type=str, default='all',
                        help="Problem, 'tsp', 'mec'"
                             " or 'all' to generate all")
    parser.add_argument('--data_distribution', type=str, default='all',
                        help="Distributions to generate for problem, default 'all'.")

    parser.add_argument("--dataset_size", type=int, default=10000, help="Size of the dataset")
    parser.add_argument('--graph_sizes', type=int, nargs='+', default=[20, 50, 100],
                        help="Sizes of problem instances (default 20, 50, 100)")
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument('--seed', type=int, default=1234, help="Random seed")

    opts = parser.parse_args()

    assert opts.filename is None or (len(opts.problems) == 1 and len(opts.graph_sizes) == 1), \
        "Can only specify filename when generating a single dataset"

    distributions_per_problem = {
        'tsp': [None],
        'mec': [None]
    }
    if opts.problem == 'all':
        problems = distributions_per_problem
    else:
        problems = {
            opts.problem:
                distributions_per_problem[opts.problem]
                if opts.data_distribution == 'all'
                else [opts.data_distribution]
        }

    for problem, distributions in problems.items():
        for distribution in distributions or [None]:
            for graph_size in opts.graph_sizes:

                datadir = os.path.join(opts.data_dir, problem)
                os.makedirs(datadir, exist_ok=True)

                if opts.filename is None:
                    filename = os.path.join(datadir, "{}{}{}_{}_seed{}.pkl".format(
                        problem,
                        "_{}".format(distribution) if distribution is not None else "",
                        graph_size, opts.name, opts.seed))
                else:
                    filename = check_extension(opts.filename)

                assert opts.f or not os.path.isfile(check_extension(filename)), \
                    "File already exists! Try running with -f option to overwrite."

                np.random.seed(opts.seed)
                if problem == 'tsp':
                    dataset = generate_tsp_data(opts.dataset_size, graph_size)
                elif problem == 'mec':
                    dataset = generate_mec_data(opts.dataset_size, graph_size, opts.priority)
                
                else:
                    assert False, "Unknown problem: {}".format(problem)

                print(dataset[0])

                save_dataset(dataset, filename)
