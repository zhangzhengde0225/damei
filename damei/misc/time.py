"""
dm关于时间的函数
"""
import datetime


def current_time(format='%Y-%m-%d %H:%M:%S'):
    """
    读取当前时间

    :param format: format
    :return: time, str

    Example:
        >>> ctime = dm.current_time()
    """
    return datetime.datetime.now().strftime(format)


def plus_time(ctime=None, seconds=1, format='%Y-%m-%d %H:%M:%S'):
    """
    添加时间

    :param ctime: current time, str
    :param seconds: seconds, int
    :param format: format, str
    :return: time, str

    Example:
        >>> future_time = dm.plus_time(ctime=ctime, seconds=50*24*60*60)
    """
    ct = ctime if ctime else current_time(format=format)
    ct = datetime.datetime.strptime(ct, format)
    return (ct + datetime.timedelta(seconds=seconds)).strftime(format)


def within_time(expiration_time, ctime=None, format='%Y-%m-%d %H:%M:%S'):
    """
    输入到期时间，判断当前时间是否在到期时效内

    :param expiration_time: 到期时间, str
    :param ctime: 当前时间，str. Default: current_time()
    :param format: 格式，str
    :return: True or False, bool

    Example:
        >>> is_within = dm.within_time(expiration_time=future_time, current_t=None)
    """
    ct = ctime if ctime else current_time(format=format)  # 读取当前时间，str格式
    # ct = plus_time(ct, seconds=50*24*60*60)  # 测试用的，
    ct = datetime.datetime.strptime(ct, format)  # 转为date格式
    et = datetime.datetime.strptime(expiration_time, format)  # 转为date格式
    return ct < et
