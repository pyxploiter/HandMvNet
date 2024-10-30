import torch


def reverse_transform(image, denormalize=True, IMAGENET_TRANSFORM=False):
    """
    Reverse the normalization and ToTensor transformations.
    """
    if denormalize:
        # TODO: bug in following imagenet transform
        if IMAGENET_TRANSFORM:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        else:
            mean = 0.5
            std = 0.5
        image = image * std + mean  # Reverse Normalize

    image = image.permute(1, 2, 0)  # Convert from CxHxW to HxWxC
    image = image.clamp(0, 1) * 255  # Ensure pixel values are within [0, 255]
    return image.cpu().numpy().astype("uint8")


def get_mano_closed_faces(faces):
    """
    The default MANO mesh is "open" at the wrist. By adding additional faces, the hand mesh is closed,
    which looks much better.
    https://github.com/hassony2/handobjectconsist/blob/master/meshreg/models/manoutils.py
    """
    close_faces = torch.Tensor([
        [92, 38, 122],
        [234, 92, 122],
        [239, 234, 122],
        [279, 239, 122],
        [215, 279, 122],
        [215, 122, 118],
        [215, 118, 117],
        [215, 117, 119],
        [215, 119, 120],
        [215, 120, 108],
        [215, 108, 79],
        [215, 79, 78],
        [215, 78, 121],
        [214, 215, 121],
    ]).to(faces.device)
    th_closed_faces = torch.cat([faces, close_faces.long()])
    return th_closed_faces
