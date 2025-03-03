# import random
import random
import cv2
import numpy as np
from PIL import Image

from torchvision import transforms


def augment_square_bounding_boxes(bboxes, translation_range=0.1, scale_range=0.1):
    """
    Augment square bounding boxes by translating and scaling while ensuring they remain square.
    
    Parameters:
    - bboxes: numpy array of shape (num_views, 4), where each row is (x_min, y_min, x_max, y_max).
    - translation_range: Fraction of the size (width/height) by which to translate the bounding box.
    - scale_range: Fraction of the size (width/height) by which to scale the bounding box.
    
    Returns:
    - Augmented square bounding boxes of the same shape as input.
    """
    # Compute the size of each square bounding box (width == height)
    sizes = bboxes[:, 2] - bboxes[:, 0]  # Since boxes are square, width == height

    # Random translations
    translation = np.random.uniform(-translation_range, translation_range, size=bboxes.shape[0]) * sizes

    # Random scaling
    scale = 1 + np.random.uniform(-scale_range, scale_range, size=bboxes.shape[0])

    # Apply translations
    bboxes_augmented = np.zeros_like(bboxes)
    bboxes_augmented[:, 0] = bboxes[:, 0] + translation
    bboxes_augmented[:, 1] = bboxes[:, 1] + translation
    bboxes_augmented[:, 2] = bboxes[:, 2] + translation
    bboxes_augmented[:, 3] = bboxes[:, 3] + translation

    # Apply scaling while keeping the boxes square
    new_sizes = sizes * scale
    half_new_sizes = new_sizes / 2
    centers_x = (bboxes_augmented[:, 0] + bboxes_augmented[:, 2]) / 2
    centers_y = (bboxes_augmented[:, 1] + bboxes_augmented[:, 3]) / 2

    bboxes_augmented[:, 0] = centers_x - half_new_sizes
    bboxes_augmented[:, 1] = centers_y - half_new_sizes
    bboxes_augmented[:, 2] = centers_x + half_new_sizes
    bboxes_augmented[:, 3] = centers_y + half_new_sizes

    def validate_square_bounding_boxes(bboxes):
        """
        Validate that the bounding boxes are square. Raise an error if any are not square.
        
        Parameters:
        - bboxes: numpy array of shape (num_views, 4), where each row is (x_min, y_min, x_max, y_max).
        
        Raises:
        - ValueError: If any bounding box is not square.
        """
        widths = bboxes[:, 2] - bboxes[:, 0]
        heights = bboxes[:, 3] - bboxes[:, 1]
        
        # Check if width == height for each bounding box
        square_check = np.isclose(widths, heights)
        
        # If any bounding box is not square, raise an error
        if not np.all(square_check):
            raise ValueError("One or more bounding boxes are not square after augmentation: ", widths, heights)
        
        return square_check

    # validate squared bounding boxes
    # validate_square_bounding_boxes(bboxes_augmented)  # some bboxes have 1-2 pixel difference
    return bboxes_augmented


class BlurAugmentation:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        
        # Convert image to a writable numpy array
        img = np.array(img, copy=True)

        type = random.randint(1, 3)
        kernel_size = random.choice([3, 5, 7])

        if type == 1:
            img = cv2.blur(img, (kernel_size, kernel_size))
        elif type == 2:
            img = cv2.medianBlur(img, kernel_size)
        elif type == 3:
            sigma = random.choice([1, 2, 3])
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma)

        # Convert back to PIL Image and return
        return Image.fromarray(img)


class OcclusionAugmentation:
    def __init__(self, patch_size_range, p=0.5):
        self.p = p
        self.min_size = patch_size_range[0]
        self.max_size = patch_size_range[1]

    def __call__(self, img):
        if random.random() > self.p:
            return img
        
        patch_size = random.randint(self.min_size, self.max_size)

        # Convert image to a writable numpy array
        img = np.array(img, copy=True)

        height, width, channel = img.shape
        row, column = height // patch_size, width // patch_size

        # Randomly select a patch to occlude
        mask_row = random.randint(0, row - 1)
        mask_column = random.randint(0, column - 1)

        # Apply occlusion
        img[mask_row * patch_size:(mask_row + 1) * patch_size,
            mask_column * patch_size:(mask_column + 1) * patch_size, :] = 0

        # Convert back to PIL Image and return
        return Image.fromarray(img)


class SampleAugmentor:
    def __init__(self):

        transforms_list = [
            transforms.ColorJitter(brightness=.2, saturation=0.1, hue=.005),
            BlurAugmentation(p=0.4),
            # transforms.GaussianBlur(kernel_size=kernel_size, sigma=(0.1, 3)),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
            transforms.RandomPosterize(bits=6, p=0.3),
            # transforms.RandomAutocontrast(p=0.3),
            OcclusionAugmentation(patch_size_range=(8, 64), p=0.3)
            # transforms.RandomEqualize(p=0.5)
        ]

        # transforms_list = random.sample(transforms_list, 4)
        self.transform = transforms.Compose(transforms_list)

    def __call__(self, rgb):
        rgb = self.transform(Image.fromarray(rgb))
        rgb = np.asarray(rgb)
        return rgb
