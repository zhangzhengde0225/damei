import os, sys
from pathlib import Path

pydir = Path(os.path.abspath(__file__)).parent
sys.path.append(str(pydir.parent.parent))
import damei as dm


def augment():
    source_path = f'{pydir.parent}/sources/dm.data/anno_dataset_Labelmefmt'
    target_path = f'{pydir.parent}/sources/dm.data/augmented_5images'
    background_path = f'{pydir.parent}/sources/dm.data/backgrounds'
    dm.data.augment(
        source_path=source_path,
        target_path=None,  # 使用default
        backgrounds_path=background_path,
        anno_fmt='labelme',
        out_fmt='YOLOfmt',
        use_noise_background=False,
        out_size=640,
        num_augment_images=5,
        train_ratio=0.8,
    )


if __name__ == '__main__':
    augment()
