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
    plt.ylabel("Runtime (s)")
    plt.title(f"MNIST Alexnet runtime by trial and parallelization strategy".capitalize())

    plt.xticks(ind + width / 2, (f"T{i}" for i in range(num_trials)))
    plt.legend(loc='lower right')
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
            {"runtime": 60.009412026032805, "mem_params": 244341408, "mem_bufs": 0, "mem_peak": 2156343296/1073741824},
            {"runtime": 60.26496073510498, "mem_params": 244341408, "mem_bufs": 0, "mem_peak": 2156343296/1073741824},
            {"runtime": 60.48147195391357, "mem_params": 244341408, "mem_bufs": 0, "mem_peak": 2156343296/1073741824},
            {"runtime": 57.63888473622501, "mem_params": 244341408, "mem_bufs": 0, "mem_peak": 2154813440/1073741824},
            {"runtime": 58.477564914152026, "mem_params": 244341408, "mem_bufs": 0, "mem_peak": 2154813440/1073741824},
            {"runtime": 57.36283093178645, "mem_params": 244341408, "mem_bufs": 0, "mem_peak": 2154813440/1073741824},
            {"runtime": 59.94434227794409, "mem_params": 244341408, "mem_bufs": 0, "mem_peak": 2154764288/1073741824},
            {"runtime": 57.36915805004537, "mem_params": 244341408, "mem_bufs": 0, "mem_peak": 2154764288/1073741824},
            {"runtime": 57.448675252497196, "mem_params": 244341408, "mem_bufs": 0, "mem_peak": 2154764288/1073741824}
        ],
        'DP': [
            {"runtime": 98.86405460909009, "mem_params": 244341408, "mem_bufs": 0, "mem_peak": 3344906752/1073741824},
            {"runtime": 98.76292283693328, "mem_params": 244341408, "mem_bufs": 0, "mem_peak": 3344906752/1073741824},
            {"runtime": 99.04266539029777, "mem_params": 244341408, "mem_bufs": 0, "mem_peak": 3344906752/1073741824},
            {"runtime": 67.05662207491696, "mem_params": 244341408, "mem_bufs": 0, "mem_peak": 3343928832/1073741824},
            {"runtime": 66.58223777450621, "mem_params": 244341408, "mem_bufs": 0, "mem_peak": 3343928832/1073741824},
            {"runtime": 67.14807305671275, "mem_params": 244341408, "mem_bufs": 0, "mem_peak": 3343928832/1073741824},
            {"runtime": 62.70460104569793, "mem_params": 244341408, "mem_bufs": 0, "mem_peak": 3343360512/1073741824},
            {"runtime": 61.56561255082488, "mem_params": 244341408, "mem_bufs": 0, "mem_peak": 3343360512/1073741824},
            {"runtime": 60.24541654996574, "mem_params": 244341408, "mem_bufs": 0, "mem_peak": 3343360512/1073741824}
        ],
        'Horovod + Ray Tune': [
            {"runtime":  97.3423 , "mem_peak":  3233902592/1073741824 },
            {"runtime":  97.266  , "mem_peak":  3233902592/1073741824 },
            {"runtime":  97.2807 , "mem_peak":  3233902592/1073741824 },
            {"runtime":  66.3573 , "mem_peak":  3233257472/1073741824 },
            {"runtime":  65.9218 , "mem_peak":  3233257472/1073741824 },
            {"runtime":  66.6642 , "mem_peak":  3233257472/1073741824 },
            {"runtime":  66.0314 , "mem_peak":  3233280000/1073741824 },
            {"runtime":  68.2389 , "mem_peak":  3233257472/1073741824 },
            {"runtime":  65.7218 , "mem_peak":  3232670208/1073741824 },

        ]
        # 'GPipe': [
        #     {},
        #     {},
        #     {},
        #     {},
        #     {},
        #     {},
        #     {},
        #     {},
        #     {}
        #    ]

    }
    plot_trial_results(all_trial_results, num_trials, 'runtime', 's')

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


