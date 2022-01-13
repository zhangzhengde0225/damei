import os, sys


def current_system():
	"""
	get current system: linux, macos or windows
	:return:
	"""
	if sys.platform.startswith('linux'):
		return 'linux'
	elif sys.platform.startswith('darwin'):
		return 'macos'
	elif sys.platform.startswith('win32'):
		return 'windows'
	else:
		raise NameError(f'damei: current system is neither linux nor macos nor windows: {sys.platform}')


def system_lib_suffix():
	csystem = current_system()
	if csystem == 'linux':
		return '.so'
	elif csystem == 'macos':
		return '.dylib'
	elif csystem == 'windows':
		return '.dll'
	else:
		return None
