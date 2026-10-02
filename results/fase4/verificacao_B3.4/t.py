import time,sys
sys.argv=['v','bauru','Q1','10','1','1','0']
src=open('v.py').read().split("out=dict")[0]
exec(src)
t=time.time(); print(ret(1000,1),time.time()-t)
