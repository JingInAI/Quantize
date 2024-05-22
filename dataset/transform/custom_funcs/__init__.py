from .rotate import *

from utils import Register

CUSTOMFUNCS = Register({
    'rotate_with_labels': rotate_with_labels,
    'random_rotate': random_rotate,
})
