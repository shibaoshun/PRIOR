''' useful tools '''
class Unbuffered(object):
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def writelines(self, datas):
       self.stream.writelines(datas)
       self.stream.flush()
   def __getattr__(self, attr):
       return getattr(self.stream, attr)

import os
import shutil
def copytree(src, dst, symlinks=False, ignore=None):
    if not os.path.exists(dst):
        os.mkdir(dst)
    for item in os.listdir(src):
        if item != '__pycache__':
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, symlinks, ignore)
            else:
                shutil.copy2(s, d)

def load_parallel(state_dict):
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

def load_reset_key(state_dict, str):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    num = len(str)
    for k, v in state_dict.items():
        name = k[num:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


