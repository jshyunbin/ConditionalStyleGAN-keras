import urlib.request
import io

file_dir = input('url file name \n>')
file = open('../dataset/imageNet/'+file_dir, 'r')
lines = file.readlines()

for i in range(lines):
    urlib.request.urlretrieve(lines[i], '../dataset/imageNet/%s/%04d.jpg' % (file_dir, i))