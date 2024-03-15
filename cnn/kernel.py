# Wykonuje operację iloczynu skalarnego między kanałami koloru RGB a wagami [0.299, 0.587, 0.114] w celu uzyskania odcieni szarości.
def rgb_to_grey(rgb): return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


input_image = plt.imread('cat.png')
grey_img = rgb_to_grey(input_image)  # Konwertuje obraz wejściowy na odcienie szarości
small_img = grey_img[::2, ::2]  # Tworzy mniejszy obraz poprzez pominięcie co drugiego piksela w pionie i w poziomie.



from scipy.signal import convolve2d


def apply_kernel_to_image(img, kernel, title=''):
    # Stosuje filtr konwolucyjny na obrazie, z zachowaniem wymiarów i symetrycznym obszarem brzegowym.
    feature = convolve2d(img, kernel, boundary='symm', mode='same')

    # plot
    fig = plt.figure(figsize=(20, 10))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(img, 'gray')
    ax1.set_title('Input image', fontsize=15)
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(feature, 'gray')
    ax2.set_title(f'Feature map - {title}', fontsize=15)
    ax2.set_xticks([])
    ax2.set_yticks([])

    plt.show()



kernel = np.array([
    [0.01, 0.01, 0.01],
    [0.01, 0.01, 0.01],
    [0.01, 0.01, 0.01]])

apply_kernel_to_image(small_img, kernel, 'blur')