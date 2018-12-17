import fileinput
import re
import sys

auto = True
val = ""
new = ""
if len(sys.argv) > 1:
    auto = False
    with open("./benderclient/__init__.py") as f:
        content = f.readlines()
        val = content[9].split('"')[1]
        new = raw_input("[Current " + val + "] - Specify a new version : ")


for line in fileinput.input('./benderclient/__init__.py', inplace=True):
    if auto is True:
        line = re.sub(r'= "(.+\..+\..+\.+)(.+)"', lambda x: '= "' + str(x.group(1)) + str(int(x.group(2)) + 1) + '"', line.rstrip())
    else:
        line = re.sub(r'= "(.+\..+\..+\.+)(.+)"', '= "' + new + '"', line.rstrip())
    print(line)