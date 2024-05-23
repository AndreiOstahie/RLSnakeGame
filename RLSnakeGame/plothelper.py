import matplotlib.pyplot as plt
from IPython import display

plt.ion()


def plot(scores, avg_scores):
    display.clear_output(wait=True)
    # display.display((plt.gcf()))
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(avg_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.text(len(avg_scores) - 1, avg_scores[-1], str(avg_scores[-1]))

    plt.pause(0.1)
    plt.show(block=False)


def plot_thread(queue):
    plt.ion()
    while True:
        # Check if there are arguments in the queue
        if not queue.empty():
            scores, avg_scores = queue.get()  # Get the latest scores and avg_scores
            display.clear_output(wait=True)
            display.display((plt.gcf()))
            plt.clf()
            plt.title('Training...')
            plt.xlabel('Number of Games')
            plt.ylabel('Score')
            plt.plot(scores)
            plt.plot(avg_scores)
            plt.ylim(ymin=0)
            plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
            plt.text(len(avg_scores) - 1, avg_scores[-1], str(avg_scores[-1]))

        plt.pause(0.1)