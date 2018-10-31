import sys


def print_progress(progress, epoch_num, loss):
    """
    This function draw an active progress bar.
    :param progress: Where we are:
                       type: float
                       value: [0,1]
    :param epoch_num: number of epochs for training
    :param loss: The loss for the specific batch in training phase.

    :return: Progressing bar
    """

    # Define the length of bar
    barLength = 30

    # Ceck the input!
    assert type(progress) is float, "id is not a float: %r" % id
    assert 0 <= progress <= 1, "variable should be between zero and one!"

    # Empty status while processing.
    status = ""

    # This part is to make a new line when the process is finished.
    if progress >= 1:
        progress = 1
        status = "\r\n"

    # Where we are in the progress!
    indicator = int(round(barLength*progress))

    # Print the appropriate progress phase!
    list = [str(epoch_num), "#"*indicator , "-"*(barLength-indicator), progress*100, loss, status]
    text = "\rEpoch {0[0]} {0[1]} {0[2]} %{0[3]:.2f} loss={0[4]:.3f} {0[5]}".format(list)
    sys.stdout.write(text)
    sys.stdout.flush()
