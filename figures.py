import pickle
import pandas as pd
import numpy as np
from plotnine import ggplot, geom_point, aes, stat_smooth, facet_wrap, xlab, ylab, theme

with open('sparsity_result.pkl', 'rb') as f:
    snnL, criL, hbmE = pickle.load(f)

snnL = snnL[1:]
print(snnL)
criL = criL[1:] #seconds
criL = np.array(criL)*(2.22*10**(-9))
print(criL)
hbmE = hbmE[1:] #picojoules
hbmE = np.array(hbmE)*(256*4)
sparsity = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7]

data = {'Sparsity' : sparsity,
	'Energy' : hbmE,
	'Hardware' : criL,
	'Software' : snnL}


df = pd.DataFrame(data)

latencydf = pd.melt(df, id_vars=['Sparsity'], value_vars=['Software', 'Hardware'],

        var_name='Platform', value_name='Latency')


fig = (ggplot(df, aes('Sparsity', 'Energy'))
 + geom_point()+xlab('% Sparsity')+ylab('Latency (S)'))
fig.save(filename = 'eng.png')

fig = (ggplot(latencydf, aes('Sparsity', 'Latency',color='Platform'))
 + geom_point()+xlab('% Sparsity')+ylab('Energy (pJ)')+theme(legend_position = (0.75, 0.75)))
fig.save(filename = 'latency.png')
