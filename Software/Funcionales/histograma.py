import numpy as np
import matplotlib.pyplot as plt

#Estilos disponibles para pyplot:
#['bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot',
# 'grayscale', 'seaborn-bright', 'seaborn-colorblind', 'seaborn-dark-palette',
# 'seaborn-dark', 'seaborn-darkgrid', 'seaborn-deep', 'seaborn-muted',
# 'seaborn-notebook', 'sea5born-paper', 'seaborn-pastel', 'seaborn-poster',
# 'seaborn-talk', 'seaborn-ticks', 'seaborn-white', 'seaborn-whitegrid',
# 'seaborn', 'Solarize_Light2', '_classic_test']

def histograma(valores_a_binear, bins='auto', errbars=True, density=True,
               logbins=False, logx=False, logy=False,
               ax=None,
               titulo=None, xlabel=None, ylabel=True,
               labelsize=18, ticksize=16,
               ecolor=None, anotacion=False):
    """Función maestra (?) para realizar los histogramas más bellos que usted
    haya soñado jamás.

    Inputs
    ------
    valores_a_binear : lista o ndarray
        Los valores que se van a volcar en el histograma
    bins : int, list, string o 3-tuple
        Si es int, es el número de bines. Si es list, debe ser la lista de
        los bordes de los bines. Si es string 'auto', funciona igual
        que pasarle bins='auto' a la función np.histogram. Si es una tupla,
        sus elementos deben ser la pos. del borde izquierdo del 1er bin, la
        pos. del borde derecho del último bin, y el número de bines deseado
        (en ese orden).
    titulo : string
        Título a ponerle al histograma.
    xlabel : string
        Rótulo del eje x del histograma
    density : Bool
        Si True, grafica el histograma normalizado (funciona bien incluso si
        los bines son de diferente tamaño)
    logbins : Bool
        Si True, y bins es una 3-tupla con inicio, final y número de bines,
        entonces el tamaño de los bines entre inicio y final crece
        exponencialmente (ancho constante en escala logarítmica en eje x).
    logx : Bool
        Si True, se grafica el eje x en escala logarítmica.
    logy : Bool
        Si True, se grafica el eje y en escala logarítmica.
    ax : matplotlib.axes.Axes object
        axes sobre el cual graficar el histograma. Si es None, crea una nueva
        figura para graficar.
    ecolor : string
        Color de los bordes de los bines. Si esNone, los bines no tienen bordes
        diferenciados.
    anotacion : Bool
        Si True, agrega un cuadro con el número de eventos y de bines graficados
        Por ahora, el lugar en que el cuadro se coloca es estático y puede
        solaparse con el histograma.

    Notas
    -----
    Si bien no está expuesto de manera explícita, esta función también sirve
    para hacer histogramas de cantidades discretas en escala lineal (en los
    cuales es habitual no agregar eventos correspondientes a números enteros
    diferentes). Para ello, basta pasar como parámetro
        bins=np.arange(x_inicial, x_final+2).
    El +2 dos es por lo siguiente: un +1 es necesario para que np.arange
    incluya el valor x_final en el array resultante, y otro es necesario para
    que además incluya el valor x_final + 1 que corresponde al borde derecho
    del último bin, el cual es necesario pasar a np.histogram.
    """

    if ax is None:
        with plt.style.context(('seaborn')):
            fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')

    if isinstance(bins, tuple):
        if logbins:
            start, stop, nbins = bins
            bins = np.geomspace(start, stop, num=nbins)
#            ax.plot(bins,[0]*len(bins), '+k') # testing
        else:
            start, stop, nbins = bins
            bins = np.linspace(start, stop, num=nbins)

    conteos, bordes_bines = np.histogram(valores_a_binear, bins=bins)
    w = np.diff(bordes_bines) # Anchos de los bines


    # Normalizar, si es necesario, y asignar errores
    if density == True:
        cnorm = conteos / (np.sum(conteos) * w)
        enorm = np.sqrt(conteos) / (np.sum(conteos) * w)
        conteos, errores = cnorm, enorm
    else:
        errores = np.sqrt(conteos)
    # print('La integral del histograma es igual a ', np.sum(w * conteos))

    # Graficar
    if errbars:
        ax.bar(bordes_bines[:-1], conteos, width=w, yerr=errores, align='edge',
               color='dodgerblue', capsize=0, edgecolor=ecolor)
    else:
        ax.bar(bordes_bines[:-1], conteos, width=w, align='edge',
               color='dodgerblue', edgecolor=ecolor)

    ax.tick_params(labelsize=ticksize)
    if titulo != None:
        ax.set_title(titulo, fontsize=labelsize)
    if xlabel != None:
        ax.set_xlabel(xlabel, fontsize=labelsize)
    if ylabel:
        ylabel = '# de eventos' if density==False else '# de eventos normalizado'
        ax.set_ylabel(ylabel, fontsize=labelsize)
    if anotacion:
        num_bines = len(bordes_bines) - 1
        anotacion = ('$N = $' + str(len(valores_a_binear))+ '\n' +
                     r'$N_{bines}$ = ' + str(num_bines))
        ax.annotate(anotacion,
                    (.8, .8), xycoords='axes fraction',
                    backgroundcolor='w', fontsize=14)

    fig.tight_layout()
    plt.show()

    return fig, ax

def binplot(valores, imin=None, imax=None, titulo=None,
            errorbars=True, ax=None):
    """Gráfico de barras. Recibe un iterable con
    los valores a graficar, de forma tal que la altura del bin i es igual a
    valores[i]. imin e imax son los índices entre los cuales graficar. Si no se
    especifican, se grafica la región en la cual las alturas de los bines
    son disintas de cero.
        Útil para realizar histogramas de cantidades discretas.

    PENDIENTES:
        - Agregar opción de normalización
        - Agregar opción de graficar los errores de manera distinta (simétrica)
        en el caso de que log == True
        - Agregar opción `multiplo` que permita que en el eje x aparezcan
        múltiplos de un cierto valor (indicando las unidades en el rótulo)."""
    if imin is None:
        imin = np.where(valores != 0)[0][0]
        # Si puedo, corro imin 1 bin a la izquierda para que quede más lindo:
        if imin != 0:
            imin -= 1
    if imax is None:
        imax = np.where(valores != 0)[0][-1] + 1
        # El +1 es porque valores[imax] no será graficado.
        # Si puedo, corro imax 1 bin a la derecha para que quede más lindo:
        if imax != len(valores):
            imax += 1
    if ax is None:
        with plt.style.context(('seaborn')):
            fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    xs, hs = range(imin, imax), valores[imin:imax]
    errs = np.sqrt(hs) if errorbars == True else None
    ax.bar(xs, hs, width=1, yerr=errs, align='edge',
           color='dodgerblue', capsize=0)
    if log == True:
        ax.set_yscale('log')
    ax.set_xticks(np.arange(imin, imax))
    if titulo is not None:
        ax.set_title(titulo, fontsize=16)
    fig.tight_layout()

def hist_discreto(xs, imin=None, imax=None, titulo=None,
                  log=False, errorbars=True, ax=None):
    """Realiza un histograma de una variable discreta, a partir
    de los samples contenidos en `xs`.

    Es un envoltorio para la función binplot.
    """
    hs = np.bincount(xs)
    binplot(hs, imin=imin, imax=imax, titulo=titulo, log=log,
            errorbars=errorbars, ax=ax)


if __name__ == '__main__':
   from scipy.stats import expon
   xs = expon(scale = (1 / 0.5)).rvs(int(1e5)) # lambda = 0.5
   histograma(xs, density=True, ecolor='k', bins=(0.01,40,100))
   histograma(xs, logbins=True, bins=(0.01,40,100), density=True,
              ecolor='k')
   histograma(xs, logbins=True, bins=(0.01,40,100), density=True,
              logx=True, ecolor='k')
   histograma(xs, logbins=True, bins=(0.01,40,100), density=True,
              logy=True, ecolor='k')
   histograma(xs, logbins=True, bins=(0.01,40,100), density=True,
              logx=True, logy=True, ecolor='k')
