import urllib.request   
import io
import os
from os import listdir
from os.path import isfile, join

path = '../dataset/imageNet/'
onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

for file in onlyfiles:
    file = join(path, file)
    ofile = open(file, 'r')
    lines = ofile.readlines()
    os.mkdir(file[:-4])
    for i in range(len(lines)):
        try:
            urllib.request.urlretrieve(lines[i], join(file[:-4], '%04d.jpg' % i))
        except urllib.error.HTTPError as err:
            continue
        except urllib.error.URLError as err:
            continue
        if i % 10 == 0:
            print('%dth in progress...' % i)
    print(file + ' done')