import damei as dm

dmt = dm.Tools()

dmt.test()

user, pwd, ip, channel = "admin", "123qweasd", "192.168.1.210", 1
path = f"rtsp://{user}:{pwd}@{ip}//Streaming/Channels/{1}"
print(path)
dmt.cap_video_save(path, rotate=False, save_dir="/Users/tanmenglu/Downloads/dd")
