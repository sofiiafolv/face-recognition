import subprocess
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
from numpy import arange
import time


def run_python_file(path, k, svd_type):
    try:

        subprocess_result = subprocess.run(
            ["python3", "main.py", str(path), str(k), str(svd_type)],
            capture_output=True,
        )
    except FileNotFoundError:
        raise FileNotFoundError("No such file or directory")

    if subprocess_result.returncode != 0:
        return -1


def build_graphs(svd_qr_times_list, svd_ev_ec_times_list, svd_numpy_times_list):
    barWidth = 0.25
    fig = plt.subplots(figsize=(12, 8))
    br1 = arange(len(svd_qr_times_list))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    plt.title(f"Time comparison", fontdict={"fontsize": 30, "fontweight": 600})
    plt.bar(
        br1,
        svd_qr_times_list,
        color="#D7A9E3",
        width=barWidth,
        label="SVD using power method",
    )
    plt.bar(
        br2,
        svd_ev_ec_times_list,
        color="#8BBEE8",
        width=barWidth,
        label="SVD using library functions for EV's and EVc's",
    )
    plt.bar(
        br3,
        svd_numpy_times_list,
        color="#A8D5BA",
        width=barWidth,
        label="SVD from numpy",
    )
    plt.xlabel("k-rank", fontweight="bold", fontsize=15)
    plt.ylabel("Time, seconds", fontweight="bold", fontsize=15)
    plt.xticks(
        [r + barWidth for r in range(len(svd_qr_times_list))],
        [25, 50],
    )
    plt.legend()
    plt.savefig(f"../graphs/graph.png")
    plt.show()


def main():
    times_list = []
    for i in [0, 1, 2]:
        t = []
        for j in [25, 50]:
            start = time.time()
            run_python_file("../testing_faces/yaleB01/yaleB01_P00A-005E-10.pgm", j, i)
            end = time.time()
            t.append(end - start)
            print(i, j)
        times_list.append(t)
    build_graphs(times_list[0], times_list[1], times_list[2])


main()
