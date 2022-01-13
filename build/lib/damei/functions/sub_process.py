import subprocess


def popen(code, need_error=False):
	out = subprocess.Popen(code, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
	if need_error:
		output = out.stdout.read().decode('utf-8').replace('\r', '').split('\n')
		error = out.stderr.read().decode('utf-8').replace('\r', '').split('\n')
		output = output[:-1] if output[-1] == '' else output
		error = error[:-1] if error[-1] == '' else error
		return output, error
	else:
		output = out.stdout.read().decode('utf-8').replace('\r', '').split('\n')
		output = output[:-1] if output[-1] == '' else output
		return output
