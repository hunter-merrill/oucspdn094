import matplotlib.pyplot as plt
import numpy as np

times = {
    "same": {
        1000000: 0.946758,
        2000000: 1.978884,
        4000000: 4.988760,
        8000000: 12.875608,
    },
    "diff": {
        1000000: 1.720234,
        2000000: 3.415052,
        4000000: 6.817071,
        8000000: 13.606987,
    },
}

for node in times:

    x, y = [], []
    for n in times[node]:
        time = times[node][n]
        x.append(n)
        y.append(
            time / 2000 * 1000
        )  # Convert total time in seconds -> time per one-way send in milliseconds

    x, y = np.array(x), np.array(y)
    m, b = np.polyfit(x, y, 1)

    plt.plot(x, m * x + b, color="red", lw=1, ls="solid")
    plt.scatter(x, y, zorder=2)

    plt.suptitle(node)
    plt.title(
        f"Latency = {b:.4f} ms, Bandwidth = {(4/m/1024/1024 + 0.5):.4f} GB/s"
    )  # Bandwidth in bytes per millisecond, rounded to the nearest whole number
    plt.xlabel("Array size (millions)")
    plt.ylabel("Time per send (ms)")
    plt.xlim(300_000, 8_250_000)
    plt.ylim(0, 8.25)

    plt.show()
