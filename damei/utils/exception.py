from damei.misc.logger import getLogger

logger = getLogger('exception.py')


def exception_handler(error_type=None, error_info=None, **kwargs):
    # error = kwargs.get('error', )
    mute = kwargs.get('mute', False)
    if mute:
        return
    need_traceback = kwargs.get('traceback', False)
    info2 = kwargs.get('info', None)

    info = f'{error_type}' if error_type else ''
    # print(info)
    info += f': {error_info}.' if error_info else info
    # print(info)
    info += f' {info2}' if info2 else info
    # print(info)

    if need_traceback:
        info += f'\n{error_info.__traceback__.tb_frame.f_globals["__file__"]}:{error_info.__traceback__.tb_lineno}'
    logger.info(info)
