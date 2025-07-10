import os

import numpy as np

base_dir = r"C:\Users\liuji\OneDrive - UNSW\Project portfolio\portfolio_optimization-master\Output\Journal\LargeScaleConvergenceTerminate"
experiments = [f.path for f in os.scandir(base_dir) if f.is_dir()]

# find all experiment files
for exp in experiments:
    pop_sizes = [f.path for f in os.scandir(exp) if f.is_dir()]

    lowest_dir = os.path.basename(os.path.normpath(exp))
    print(f'"{lowest_dir}"', end=" & ")

    # for each population size directory, merge the best results
    for pop_size in pop_sizes:

        files = [f.path for f in os.scandir(pop_size) if f.is_file() and f.name.endswith(".iter")]
        first_file = True

        lowest_dir = os.path.basename(os.path.normpath(pop_size))

        values = np.empty(len(files))
        index = 0
        # np.ndarray(())
        for filename in files:
            # TODO: have to set as binary to use seek from end with non-zero value - implications?
            with open(filename, "rb") as f:
                first = f.readline()  # Read the first line.
                f.seek(-2, os.SEEK_END)  # Jump to the second last byte.
                while f.read(1) != b"\n":  # Until EOL is found...
                    f.seek(-2, os.SEEK_CUR)  # ...jump back the read byte plus one more.
                last = f.readline()  # Read last line.
                # print(last)
                best_val = float(last.split(b", ")[1])
            values[index] = float(best_val)
            index += 1

        print(f"{np.mean(values):0.3f} ({np.std(values):0.3f})", end=" & ")
    print()
