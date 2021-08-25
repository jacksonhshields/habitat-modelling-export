import numpy as np
import cv2
import itertools


def colour_bathy_patch(patch, colour_map='jet'):
    """
    Adds a colourmap to the bathymetry patch without going through matplotlib

    Args:
        patch: (np.ndarray) The bathymetry patch
        colour_map: (str) The type of colourmap to use. One of {jet,hot,ocean,bone}

    Returns:
        np.ndarray: The 3 channel bathymetry patch.

    """
    patch = patch - patch.min()
    patch = patch / patch.max() * 256
    # patch = (patch+1) * 128
    patch = patch.astype(np.uint8)
    if colour_map == 'jet':
        colmap = cv2.COLORMAP_JET
    elif colour_map == 'hot':
        colmap = cv2.COLORMAP_HOT
    elif colour_map == 'ocean':
        colmap = cv2.COLORMAP_OCEAN
    elif colour_map == 'bone':
        colmap = cv2.COLORMAP_BONE
    else:
        colmap = cv2.COLORMAP_JET
    pc = cv2.applyColorMap(patch, colmap)
    return pc

def plot_confusion_matrix(m, classes, normalized=False, save_path=None):
    import matplotlib.pyplot as plt
    if normalized:
        m = m.astype(np.float32)
        for n in range(m.shape[0]):
            m[n,:] = m[n,:] / np.sum(m[n,:])
    plt.figure()
    plt.imshow(m, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalized else 'd'
    thresh = m.max() / 2.
    for i, j in itertools.product(range(m.shape[0]), range(m.shape[1])):
        plt.text(j, i, format(m[i, j], fmt), horizontalalignment="center",
                 color="white" if m[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()