"""
post process
"""
from pathlib import Path
import struct
from damei.functions import torch_utils


def pt2wts(weights_path, output_dir='.'):
	"""
	convert pytorch trained model weights [.pt] to tensorrt pre-weights [.wts]
	:param weights_path: /path/to/xxx.pt
	:return: None, generate xxx.wts weights in output_dir.
	"""
	import torch

	output_path = f'{output_dir}/{Path(weights_path).stem}.wts'
	# Initialize
	device = torch_utils.select_device('cpu')
	# Load models
	model = torch.load(weights_path, map_location=device)['model'].float()  # load to FP32
	model.to(device).eval()

	f = open(output_path, 'w')
	f.write('{}\n'.format(len(model.state_dict().keys())))
	for k, v in model.state_dict().items():
		vr = v.reshape(-1).cpu().numpy()
		f.write('{} {} '.format(k, len(vr)))
		for vv in vr:
			f.write(' ')
			f.write(struct.pack('>f', float(vv)).hex())
		f.write('\n')
	f.close()
