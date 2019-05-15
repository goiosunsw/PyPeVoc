from pypevoc.ProgressDisplay import Progress, in_ipynb
from time import sleep

if in_ipynb():
    print('In IPYNB')
else:
    print('In console')

n = 1000

pd = Progress(n)

for ii in range(n):
    pd.update(ii)
    sleep(.002)

pd.finish()

