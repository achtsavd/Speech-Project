from ctypes import sizeof
import re
from textwrap import wrap
import numpy as np



file = open("text.txt")
text = file.read()
text = text.lower()
lines = wrap(text, width = 70, break_long_words=False)


for i,line in enumerate(lines):
    line = re.sub(r'[^A-Za-z + '  ']', ' ', line)
    line = re.sub(' +',' ',line)
    line = line.replace('',' ').strip()
    line = re.sub('  ',' SPACE',line)
    lines[i] = line
print(len(text))
print(len(lines))
print(len(lines) - len(lines)//10)
training = lines[0 : len(lines) - len(lines)//10]
test = lines [len(lines) - len(lines)//10 : len(lines)]



with open("training.txt", 'w') as f:
    for line in training:
        f.write(line+' \n')
        
with open("test.txt", 'w') as f:
    for line in test:
        f.write(line+'\n')