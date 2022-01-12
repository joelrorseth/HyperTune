import numpy as np
import matplotlib.pyplot as plt


def plot_trial_results(all_trial_results, num_trials, metric_key, units=None):
    ind = np.arange(num_trials)
    width = 0.2
    i = 0

    # Plot trial results for each parallelization strategy
    for parallelization_strat, trial_results in all_trial_results.items():
        plt.bar(ind + (i * width), list(res[metric_key] for res in trial_results),
                width, label=parallelization_strat)
        i += 1

    y_label = metric_key if units is None else f"{metric_key} ({units})"
    plt.xlabel("Trial")
    plt.ylabel("Peak Memory (Gigabytes)")
    plt.title(f"Resnet peak memory by trial and parallelization strategy".capitalize())

    plt.xticks(ind + width / 2, (f"T{i}" for i in range(num_trials)))
    plt.legend(loc='upper left')
    plt.show()


def plot_overall_runtime(overall_runtimes, units='s'):
    x = [k for k,v in overall_runtimes.items()]
    x_pos = [i for i,_ in enumerate(x)]
    y = [v for _,v in overall_runtimes.items()]

    # Plot overall runtime for each parallelization strategy
    plt.bar(x_pos, y)

    y_label = "Runtime" if units is None else f"Runtime ({units})"
    plt.ylabel(y_label)
    plt.title("Overall HPO runtime by parallelization strategy")

    plt.xticks(x_pos, x)
    plt.legend(loc='best')
    plt.show()


def main():
    num_trials = 9
    all_trial_results = {
        'MP': [
            {"runtime": 449.8547052301001, "mem_params": 102203040, "mem_bufs": 212904, "mem_peak": 10660402176/1073741824},
            {"runtime": 450.23330519208685, "mem_params": 102203040, "mem_bufs": 212904, "mem_peak": 10660402176/1073741824},
            {"runtime": 449.00318854511715, "mem_params": 102203040, "mem_bufs": 212904, "mem_peak": 10660402176/1073741824},
            {"runtime": 443.56554042501375, "mem_params": 102203040, "mem_bufs": 212904, "mem_peak": 10660402176/1073741824},
            {"runtime": 444.7348376698792, "mem_params": 102203040, "mem_bufs": 212904, "mem_peak": 10660402176/1073741824},
            {"runtime": 447.97508198209107, "mem_params": 102203040, "mem_bufs": 212904, "mem_peak": 10660402176/1073741824},
            {"runtime": 452.7358020115644, "mem_params": 102203040, "mem_bufs": 212904, "mem_peak": 17339318272/1073741824},
            {"runtime": 450.60762669099495, "mem_params": 102203040, "mem_bufs": 212904, "mem_peak": 17339318272/1073741824},
            {"runtime": 450.22575891111046, "mem_params": 102203040, "mem_bufs": 212904, "mem_peak": 17339318272/1073741824},
        ],
        'DP': [
            {"runtime": 450.9668800730724, "mem_params": 102203040, "mem_bufs": 212904, "mem_peak": 11167395328/1073741824},
            {"runtime": 447.92581078223884, "mem_params": 102203040, "mem_bufs": 212904, "mem_peak": 11167395328/1073741824},
            {"runtime": 448.4016733467579, "mem_params": 102203040, "mem_bufs": 212904, "mem_peak": 11167395328/1073741824},
            {"runtime": 433.5726760979742, "mem_params": 102203040, "mem_bufs": 212904, "mem_peak": 11585272832/1073741824},
            {"runtime": 433.81447606533766, "mem_params": 102203040, "mem_bufs": 212904, "mem_peak": 11585272832/1073741824},
            {"runtime": 440.061176746618, "mem_params": 102203040, "mem_bufs": 212904, "mem_peak": 11585272832/1073741824},
            {"runtime": 432.31205558404326, "mem_params": 102203040, "mem_bufs": 212904, "mem_peak": 22617665536/1073741824},
            {"runtime": 432.376342760399, "mem_params": 102203040, "mem_bufs": 212904, "mem_peak": 22617665536/1073741824},
            {"runtime": 432.05684545077384, "mem_params": 102203040, "mem_bufs": 212904, "mem_peak": 22617665536/1073741824},
        ],
        'GPipe': [
            {"runtime": 428.4289999678731, "mem_params": 102203040, "mem_bufs": 212904, "mem_peak": 1553804288/1073741824},
            {"runtime": 445.018094942905, "mem_params": 102203040, "mem_bufs": 212904, "mem_peak": 1571913728/1073741824},
            {"runtime": 449.95832289382815, "mem_params": 102203040, "mem_bufs": 212904, "mem_peak": 1571955200/1073741824},
            {"runtime": 421.56840551272035, "mem_params": 102203040, "mem_bufs": 212904, "mem_peak": 1577012224/1073741824},
            {"runtime": 419.8019801322371, "mem_params": 102203040, "mem_bufs": 212904, "mem_peak": 1577012224/1073741824},
            {"runtime": 423.2624448351562, "mem_params": 102203040, "mem_bufs": 212904, "mem_peak": 1586121728/1073741824},
            {"runtime": 421.29980162577704, "mem_params": 102203040, "mem_bufs": 212904, "mem_peak": 2483406848/1073741824},
            {"runtime": 420.3227094472386, "mem_params": 102203040, "mem_bufs": 212904, "mem_peak": 2531476480/1073741824},
            {"runtime": 427.2676247721538, "mem_params": 102203040, "mem_bufs": 212904, "mem_peak": 2532576768/1073741824},
        ]

    }
    plot_trial_results(all_trial_results, num_trials, 'mem_peak', 's')

    # Not particularly meaningful
    #plot_trial_results(all_trial_results, num_trials, 'accuracy', '%')

    overall_runtimes = {
        'MP': 23827 / 3600,
        'DP': 25318 / 3600,
        'GPipe': 25840 / 3600,
    }
    #plot_overall_runtime(overall_runtimes, units='hours')


if __name__ == "__main__":
    main()


