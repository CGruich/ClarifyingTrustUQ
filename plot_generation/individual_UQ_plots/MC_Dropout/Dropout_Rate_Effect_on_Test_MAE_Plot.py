import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Sans serif'
plt.rcParams.update({'font.size': 16})

par = ['0%', '5%', '10%', '20%', '30%']
SUs_13 = np.asarray(
    [
        0.650565663721069,
        0.656844056111952,
        0.6692073030924,
        0.742656429820688,
        0.890508384605024,
    ],
    dtype='float32',
)
X = np.asarray([0, 5, 10, 20, 30], dtype='float32')

fig, ax = plt.subplots()

plt.xticks(X, par)

ax.plot(
    X,
    SUs_13,
    color='black',
    marker='o',
    markerfacecolor='#008080',
    markeredgecolor='black',
    markeredgewidth='2',
    linestyle='dashed',
    linewidth=2.5,
    markersize=12,
)
# ax.set_ylim(0, 250)
# ax.yaxis.set_minor_locator(tck.AutoMinorLocator())
plt.xlabel('Dropout Rate\n(Fraction of Nodes Dropped per Layer)')
plt.ylabel('Avg. Ads. Energy MAE (eV)')
ax.tick_params(which='major', length=6, width=4, axis='y', direction='in')
ax.tick_params(which='major', length=6, width=4, axis='x', direction='in')
ax.spines['right'].set_linewidth(1)

plt.title('Effect of Dropout Rate on Test MAE', fontweight='bold')
plt.savefig(
    'Effect_of_Dropout_Rate_on_Test_MAE.svg', format='svg', bbox_inches='tight', dpi=1200
)
plt.savefig(
    'Effect_of_Dropout_Rate_on_Test_MAE.png', format='png', bbox_inches='tight', dpi=1200
)
plt.show()
