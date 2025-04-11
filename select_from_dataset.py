import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd


# load dataset-cleand.picke
with open('dataset.pickle', 'rb') as f:
    d = pickle.load(f)

import cv2

dec = [
]

# set mode to offline 

def plot_day_as_image(day):
    fig = plt.figure()
    plt.clf()
    plt.plot(day['production_energy'])
    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='') # type: ignore
    plt.close(fig)
    return img.reshape(fig.canvas.get_width_height()[::-1] + (3,))


output_days = []
i = iter(d)

for decision in dec:
    if decision == "pass":
        day = next(i)
        output_days.append(day)
    else:
        day = next(i)


for day in i:
    # get the first day
    # day = d[0]
    # plot the day as image
    img = plot_day_as_image(day)
    # save the image to disk
    cv2.imwrite('day.png', img)
    # show the image
    cv2.imshow('day', img)
    output = cv2.waitKey(0)

    # p, s (pass skip)
    if output == 112:
        print("pass")
        output_days.append(day)
        continue
    elif output == 115:
        print("skip")
        continue

# save as pickle (model-filtred.pickle)
with open('dataset-filtred-pedantic.pickle', 'wb') as f:
    pickle.dump(output_days, f)


