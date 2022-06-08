"""
测试argparse
"""
import os
import damei as dm

parser = dm.argparse.ArgumentParser()

parser.add_argument('-v', '--verbose', action='store_true', help='verbose mode')
parser.add_argument('--weights', nargs='+', type=str,
                    default=f'{os.environ["HOME"]}/weights/rsi_insulator'
                            f'/insulator_m_ep99_fogged_best.pt',
                    help='model.pt path(s)')
parser.add_argument('--demo_for_dm.data', type=str, default='sources/sfid.yaml', help='*.demo_for_dm.data path')
parser.add_argument('--source', type=str, default=None, help='test set for evaluation, use val path in .yaml if None')
parser.add_argument('--batch-size', type=int, default=64, help='size of each image batch')
parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
parser.add_argument('--task', default='val', help="'val', 'test', 'study'")
parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
parser.add_argument('--augment', action='store_true', help='augmented inference')
parser.add_argument('--merge', action='store_true', help='use Merge NMS')
parser.add_argument('--verbose', action='store_true', help='report mAP by class')
parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
parser.add_argument(f'--show-each-cls', action='store_false', help='show evaluation of each class')
parser.add_argument(f'--return-fmt', type=str, default='yolov5', help='yolov5 or damei')
# print(parser)

opt = parser.parse_args()

opt.save_json |= opt.data.endswith('coco.yaml')
# opt.demo_for_dm.data = check_file(opt.demo_for_dm.data)  # check file
print('opt', opt)

print(opt.data, type(opt.data))
print(opt.batch_size, type(opt.batch_size))
print(opt.augment, type(opt.augment))
print(opt.conf_thres, type(opt.conf_thres))
print(opt.show_each_cls, type(opt.show_each_cls))
