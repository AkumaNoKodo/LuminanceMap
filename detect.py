import matplotlib.pyplot as plt
import numpy as np

def get_colorchecker_corners(image):
    points = []

    def onclick(event):
        nonlocal points
        ix, iy = event.xdata, event.ydata
        if ix is not None and iy is not None:  # Ověření, že klik byl uvnitř os
            points.append((ix, iy))
            plt.scatter(ix, iy, c='red')
            plt.draw()
            if len(points) == 4:
                plt.close()

    fig, ax = plt.subplots()
    ax.imshow(image)
    fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()

    return np.array(points)

# Testovací kód s náhodným obrázkem
image = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
corners = get_colorchecker_corners(image)
print("Výsledné rohy:", corners)
