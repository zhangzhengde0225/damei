import os, sys
from pathlib import Path

pydir = Path(os.path.abspath(__file__)).parent
sys.path.append(str(pydir.parent.parent))
import damei as dm


def augment_mss():
    sp_dir = f'{pydir.parent}/sources/dm.data/anno_dataset_Labelmefmt'
    sp = [f'{sp_dir}/DJI_0001.jpg', f'{sp_dir}/DJI_0011.jpg']

    dm.data.augment_mss(
        source_path=sp,
        target_path=None,
        min_wh=1280,  # 默认640
        max_wh=None,
        stride_ratio=0.75,  # 默认0.5
        pss_factor=1.5,  # 默认1.25
        need_annotation=True,  # 默认False
        anno_fmt='json',
    )


if __name__ == '__main__':
    augment_mss()
