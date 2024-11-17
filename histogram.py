import conv

import functools

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.animation as animation



def create_gif(size: int):

    def as_array(s: list) -> list:
        return [len(x) for x in s]


    ex = [conv.animate_for_exponent(size)]
    first = as_array(next(ex[0]))


    def animate(_, bar_container):
        try:
            data = as_array(next(ex[0]))
        except StopIteration:
            ex[0] = conv.animate_for_exponent(size)
            data = as_array(next(ex[0]))
        # Simulate new data coming in.
        for count, rect in zip(data, bar_container.patches):
            rect.set_height(count)

        return bar_container.patches


    fig, ax = plt.subplots()
    _, _, bar_container = ax.hist(first, lw=1, bins=(1 << (size + 1)),
                                fc="green")
    
    total = 1 << size
    bar_container[(total >> 1) - 1].set_facecolor('red')
    bar_container[(3 * (total >> 1)) - 1].set_facecolor('red')

    # set safe limit to ensure that all data is visible.
    ax.set_ylim(top=1.2*(1 << size))
    ax.set_xlim(0, (1 << size) + 1)
    plt.xlabel('Palindromic Center')
    plt.ylabel('Palindromic Length')
    plt.title('Histogram of palindromes captured by the convolution')


    anim = functools.partial(animate, bar_container=bar_container)
    ani = animation.FuncAnimation(fig, anim, size, repeat=True, blit=True, interval=1000)

    # To save the animation using Pillow as a gif
    writer = animation.PillowWriter(fps=2,
                                    metadata=dict(artist='Me'),
                                    bitrate=1800)
    ani.save(f'scatter_{size}.gif', writer=writer)


# Creates a gif file
for i in range(4, 8):
    create_gif(i)