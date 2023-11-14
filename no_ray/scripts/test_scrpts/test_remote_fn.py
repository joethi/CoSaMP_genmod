import ray
import os
import time 
@ray.remote
def f(a, b, c):
    i=0
    #while i==0:
    #    omg = 0;
    os.system('sleep 10')
    print('a + b + c:',a + b + c)
    return a + b + c
st = time.time()
object_ref = f.remote(1, 2, 3)
print("out of the function")
print("intermediate time",time.time()-st)
#result = ray.get(object_ref)
#print("result",result)
et = time.time()
print('time to run remote function:',et-st)
#assert result == (1 + 2 + 3)
import pdb; pdb.set_trace()
