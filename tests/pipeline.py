"""TODO: Formalized test for data pipeline.

Can also output image patches for user to understand if its working.
"""

import torchvision.transforms.functional as F
from PIL import Image

from video_cv_project.cfg import RGB


def _test(im_path: str, pipeline):
    # NOTE: `video_cv_project.data.transform.create_pipeline` has flag ``do_rgb_norm```
    # which should be disabled for this.
    im = F.to_tensor(Image.open(im_path).convert("RGB"))
    ims = pipeline([im])
    ims = ims.unflatten(0, (-1, RGB))
    # Might want to use `torchvision.utils.make_grid` here.
    for i, patch in enumerate(ims):
        F.to_pil_image(patch).save(f"{i}.png")
