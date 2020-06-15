# Gender occ
from matplotlib import pyplot as plt
# a = [-1.8683822386395362e-05, 8.057661843262061e-08, 2.4109535211062016e-07, 4.187659111485435e-06, 8.693319380791365e-07, 4.792908133319132e-06, -2.0969056647196396e-07, 8.172500282396838e-06, 3.44198182798569e-05]
# a = [-2.5353276186328567e-05, -4.202693051336804e-06, 0.0001164199909676829, 2.5224847126986954e-05, 0.0002348076388051447, 2.7567428771718233e-05, -0.0002833249822681047, 0.00039884162961325393, -0.0004058337079184482]
a = [1.0979139850177564, 1.1852122917312848, 1.031771963741342, 1.0942986528042125, 1.1669241114626085, 1.058872571934432, 1.0394379547943424, 1.1646238235580246, 1.0145957394320888]
plt.plot(a)
plt.scatter(list(range(0, len(a))), a)
plt.xlabel('Decade')
plt.ylabel('Bias score (more positive => occup. more associated with women)')
plt.title('Average Gender occupation bias (COHA)')
ax = plt.gca()
plt.xticks(range(0,len(a)))
ax.set_xticklabels([str(1910 + i*10) for i in range(0,len(a))])
plt.show()

# Asian white occ
from matplotlib import pyplot as plt
# a = [5.643071823371144e-06, -1.0479933877425944e-05, 1.6264965473384432e-05, 3.482771413224929e-06, -2.6416840900041845e-07, 1.7892589418668425e-05, 3.5031668323129826e-05, 9.17906187603579e-06, 0.0001981798546256374]
# a = [-7.463160192214922e-05, -7.196800265344514e-05, -0.0002199445832063566, -3.179587818945474e-05, 0.00021322796829449978, 7.029270480828439e-05, 3.545494784506844e-06, 0.00011045248750286889, -0.00048342712711187347]
a = [-10.598412052298327, -11.260825189077401, -11.262960123749226, -10.685867158498056, -11.22796664893875, -10.701752244613797, -12.188418097299593, -10.711391960024413, -10.701232809342532]
plt.plot(a)
plt.scatter(list(range(0, len(a))), a)
plt.xlabel('Decade')
plt.ylabel('Bias score (more positive => occup. more associated with Asians)')
plt.title('Average Asian occupation bias (COHA)')
ax = plt.gca()
plt.xticks(range(0,len(a)))
ax.set_xticklabels([str(1910 + i*10) for i in range(0,len(a))])
plt.show()

# Asian white outsider
from matplotlib import pyplot as plt
# a = [3.7437701559957346e-07, 2.717068942287786e-06, -4.783603612840441e-07, 4.752760322004315e-07, 5.2670846287353234e-08, 3.4575125937760776e-07, 2.396425367810469e-07, -6.495093893965219e-07, -1.1799624065078072e-06]
# a = [-0.00019287278803408996, -0.00021771091495696675, -0.00020199725596701105, -0.00020248951337774898, -0.000251058792490843, -0.00023721948375760236, -0.0003227127892077105, -0.00022736674183163475, -0.00025156699879981016]
a = [0.03161758756593699, 0.030718183481944127, 0.03245871164704533, 0.030977720549517317, 0.029056472788433756, 0.030138731159401765, 0.03504278460216102, 0.033353902238938894, 0.024306016763769457]
plt.plot(a)
plt.scatter(list(range(0, len(a))), a)
plt.xlabel('Decade')
plt.ylabel('Bias score (more positive => outsider status more associated with Asians)')
plt.title('Average Asian outsider bias (COHA)')
ax = plt.gca()
plt.xticks(range(0,len(a)))
ax.set_xticklabels([str(1910 + i*10) for i in range(0,len(a))])
plt.show()

# Muslim bias
from matplotlib import pyplot as plt
a = [1.0682694677885613e-06, -1.1039327361576717e-06, -4.0275936973160615e-07, 3.3853262503555706e-07, 7.435732133982589e-07, -1.4326262302460674e-07, -4.35794027113813e-06, -2.925392269509142e-06, 8.952399837641679e-07, -1.4978087739081318e-07, 1.5539123289881764e-05, 7.270710696936431e-06, 1.1104675383454682e-05, 9.518263598650421e-07, 2.4538719839550653e-06, 9.508744319924791e-06, -2.2661530942959674e-05, -3.6304738776603014e-07]
plt.plot(a)
plt.scatter(list(range(0, len(a))), a)
plt.xlabel('Year')
plt.ylabel('Bias score (more positive => terrorism more associated with Muslims)')
plt.title('Average Islam Bias (NYT corpus)')
ax = plt.gca()
plt.xticks(range(0,len(a)))
ax.set_xticklabels([str(1988 + i) for i in range(0,len(a))])
plt.show()
