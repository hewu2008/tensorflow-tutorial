# encoding:utf8

import subprocess


class PingTester(object):
    def ping(self, hostname):
        p = subprocess.Popen('ping ' + hostname, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        ping_status = 'ok'
        for line in p.stdout:
            output = line.rstrip().decode('GBK')
            if output.find('无法访问目标主机') != -1:
                ping_status = 'failed'
                break
        return ping_status


if __name__ == "__main__":
    tester = PingTester()
    for i in range(0, 255):
        address = "169.254.136.%d" % i
        print(address, tester.ping(address))
