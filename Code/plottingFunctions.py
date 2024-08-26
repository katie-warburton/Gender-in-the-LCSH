import numpy as np
import matplotlib.pyplot as plt

'''
Get the mean proportion of the frequency of n-grams in the years before and after they were added to LCSH.
Proportion of frequency is calculated by dividing the frequency of the n-gram in a given year by the sum of the 
frequencies in the given window. This window is defined by the number of years before and after the n-gram was added
plus 1 (the year it was added).
- freqs: the frequency matrix of n-grams
- years: the years the n-grams were added to LCSH
- yrsBefore: the number of years before the n-gram was added to LCSH that should be included
- yrsAfter: the number of years after the n-gram was added to LCSH that should be included
'''
def getAvgProp(freqs, years, yrsBefore, yrsAfter):
    total = yrsBefore + yrsAfter + 1
    lines = np.zeros((len(years), total))
    for i in range (len(years)):
        added = years[i]-1975
        windowMat = freqs[i,added-(yrsBefore+1):added+yrsAfter]
        if np.sum(windowMat) == 0:
            continue
        lines[i] = windowMat / np.sum(windowMat)
    avgLine = np.mean(lines, axis=0)
    stdLine = np.std(lines, axis=0)
    return avgLine, stdLine

'''
Get the mean proportion of the frequency of n-grams in the years before and after they were added to LCSH assuming 
that the n-gram was added in any of the years within the frequency range. 
'''
def getControl(freqs, yrsBefore, yrsAfter):
    randomLines = [] 
    for i in range(yrsBefore+1, freqs.shape[1]-yrsAfter):
        windowMat = freqs[:,i-(yrsBefore+1):i+yrsAfter]
        # any non-zero lines - maybe should exclude year where it was added
        windowMat = windowMat[np.any(windowMat, axis=1)]
        line = windowMat / np.sum(windowMat, axis=1, keepdims=True)
        randomLines.append(line)
    control = np.mean(np.vstack(randomLines), axis=0)
    return control

'''
Get the mean frequency of n-grams in the years before and after they were added to LCSH.
The window of frequencies is defined by the number of years before and after the n-gram was added to LCSH
plus 1 (the year it was added).
- freqs: the frequency matrix of n-grams
- years: the years the n-grams were added to LCSH
- yrsBefore: the number of years before the n-gram was added to LCSH that should be included
- yrsAfter: the number of years after the n-gram was added to LCSH that should be included
'''
def getRawFreqs(freqs, years, yrsBefore, yrsAfter):
    total = yrsBefore + yrsAfter + 1
    lines = np.zeros((len(years), total))
    for i in range (len(years)):
        added = years[i]-1975
        windowMat = freqs[i,added-(yrsBefore+1):added+yrsAfter]
        lines[i] = windowMat
    return lines


'''
Compare and plot the Library of Congress Category Distribution of terms for women and terms for men
'''
def compareLCC(wCats, mCats, label, raw=False):
    catSpan = sorted(list(set(list(mCats.keys()) + list(wCats.keys()))))
    for cat in catSpan:
        if cat not in wCats.keys():
            wCats[cat] = 0
        if cat not in mCats.keys():
            mCats[cat] = 0
    wCats = {cat: count for cat, count in sorted(wCats.items(), key = lambda x: x[0])}
    mCats = {cat: count for cat, count in sorted(mCats.items(), key = lambda x: x[0])}
    width = 0.4

    _, ax = plt.subplots(figsize = (10, 5))
    ax.spines[['top', 'right']].set_visible(False)
    x = np.array([i for i in range(len(catSpan))])
    plt.bar(x-width, wCats.values(), width, align='edge', color='rebeccapurple', edgecolor='black', label='Women')
    plt.bar(x, mCats.values(), width, align='edge', color='goldenrod', edgecolor='black', label='Men')
    plt.xticks(x, catSpan)
    plt.legend(frameon=False, loc='upper left')
    plt.title(f'LCC Category Distirbution of {label} for Men and Women')
    plt.xlabel('Category')
    if raw:
        plt.ylabel('Number of Terms')
    else:
        plt.ylabel('Proportion of Terms')
    plt.show()

'''
Create a stacked bar plot for word frequencies over time
'''    
def plotStacked(ax, x, yDict, colours, xLabel=None, yLabel=None, title=None, barLabels=None, legend=True):
    width = 0.8
    bottom = np.zeros(len(x))
    ax.margins(0.02)
    i = 0
    for lab, counts in yDict.items():
        p = ax.bar(x, counts, width, label=lab, bottom=bottom, color=colours[i])
        bottom += counts
        i += 1
    if title is not None:
        ax.set_title(title)
    if legend:
        ax.legend(handleheight=0.5, handlelength=0.5, fontsize=14,
                  loc="lower center", bbox_to_anchor=(0.02, -0.4), ncol=2, frameon=False)
    ax.spines[['right', 'top']].set_visible(False)
    if xLabel is not None:
        ax.set_xlabel(xLabel, fontsize=14)
    if yLabel is not None:
        ax.set_ylabel(yLabel, fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(x, rotation=90)
    
    if barLabels is not None:
        for i in range(len(x)):
            ax.text(x[i], bottom[i], barLabels[i], ha='center', va='bottom', fontsize=10)
    plt.subplots_adjust(wspace=0.5, 
                        hspace=0.35)
    return ax

'''
Create a box plot for the frequency of n-grams over time
'''
def freqBoxPlot(freq, label, colour='green'):
    boxprops=dict(facecolor=colour, color='black')
    _, ax = plt.subplots(figsize = (10, 5))
    ax.spines[['top', 'right']].set_visible(False)
    ax.boxplot(freq, showfliers=False, patch_artist=True, boxprops=boxprops, medianprops=dict(color='black'))
    plt.xticks([i for i in range(1, freq.shape[1]+1)], [-i for i in range(-freq.shape[1]+1, 1)])
    plt.ylabel('Median Frequency')
    plt.xlabel('Years Before')
    plt.title(f'Median Yearly Frequency of N-grams for {label} Before Addition to LCSH')
    plt.show() 

'''
Create a bar plot for the frequency of n-grams over time
'''
def barPlots(ax, culmFreq, before, after, colour='green', ylabel=False):
    x = [i for i in range(after, -(before+1), -1)]
    x.reverse()
    ax.spines[['top', 'right']].set_visible(False)
    ax.bar(x,culmFreq, color=colour, edgecolor='black')
    #ax.plot(x, culmFreq, color='black')
    ax.set_xticks([x[i] for i in range(0, len(x))], [-(x[i]) for i in range(0, len(x))])
    ax.set_xlim(-before-0.5)
    # plt.title(f'Median Yearly Freqeuncy of N-grams for {label} before addition to LCSH')
    ax.set_xlabel('Years Before Addition')
    if ylabel:
        ax.set_ylabel('Median Frequency')
    
'''
Create a plot for the mean proportion of the frequency of n-grams over time
'''
def plotAvgFreq(avgLine, stdLine, label, n, plotType='Distribution', color='green', yrsBefore=10, yrsAfter=2, control=None, title=False):
    x = [i for i in range(yrsAfter, -(yrsBefore+1), -1)]
    x.reverse()
    _, ax = plt.subplots(figsize = (10, 5))
    ax.spines[['top', 'right']].set_visible(False)
    plt.plot(x, avgLine, label=label, color=color)
    plt.fill_between(x, avgLine-stdLine, avgLine+stdLine, alpha=0.5, facecolor=color, edgecolor=None)
    plt.axvline(x = 0, color = 'black', linewidth=0.75)

    if control is not None:
        plt.plot(x, control, label='Control', color='black', linewidth=0.95)
        plt.legend(loc='upper left', frameon=False)
    
    plt.xticks([x[i] for i in range(0, len(x))], [-(x[i]) for i in range(0, len(x))])
    plt.xlim(-yrsBefore-0.5, yrsAfter+0.5)
    plt.margins(0, 0.1)
    if title:
        plt.title(f'Mean {plotType} of {label} N-grams (n={n}) Before Addition to the LCSH')
    plt.xlabel('Years Before Addition')
    plt.ylabel(f'Proportion of Frequency')
    plt.show()


def plotMultipleFreqs(avgLines, labels, counts, colours, yrsBefore=10, yrsAfter=2, title=False):
    x = [i for i in range(yrsAfter, -(yrsBefore+1), -1)]
    x.reverse()
    _, ax = plt.subplots(figsize = (12, 5))
    ax.spines[['top', 'right']].set_visible(False)
    for i in range(len(avgLines)):
        plt.plot(x, avgLines[i], label=f'{labels[i]} (n={counts[i]})', color=colours[i])
    plt.axvline(x = 0, color = 'black', linewidth=0.75)
    plt.xticks([x[i] for i in range(0, len(x))], [-(x[i]) for i in range(0, len(x))])
    plt.xlim(-yrsBefore-0.5, yrsAfter+0.5)
    plt.margins(0, 0.1)
    if title:
        plt.title(f'Mean Probability Distribution of N-grams Before Addition to the LCSH')
    plt.xlabel('Years Before Addition')
    plt.ylabel('Proportion of Frequency')
    plt.legend(frameon=False)

def plotSideBySide(wFreqAvg, mFreqAvg, yrsBefore=10, yrsAfter=2, log=False):
    width = 0.4
    x = [i for i in range(yrsAfter, -(yrsBefore+1), -1)]
    x.reverse()
    x = np.array(x)
    _, ax = plt.subplots(figsize = (10, 5))
    ax.spines[['top', 'right']].set_visible(False)
    ax.bar(x-width,wFreqAvg, width, color='rebeccapurple',label='Women',align='edge', edgecolor='black')
    ax.bar(x,mFreqAvg, width, color='goldenrod',label='Men',align='edge', edgecolor='black')

    plt.xticks([x[i] for i in range(0, len(x))], [-(x[i]) for i in range(0, len(x))])
    plt.xlim(-yrsBefore-0.5, yrsAfter+0.5)
    plt.margins(0, 0.1)
    #plt.title(f'Median Yearly Freqeuncy of N-grams before addition to the LCSH')
    plt.xlabel('Years Before Addition')
    plt.legend(loc='upper left', frameon=False)
    if log:
        plt.yscale('log')
        plt.ylabel('Median Frequency (log)')

    else:
        plt.ylabel('Median Frequency')


def whoCameFirst(x, y):
    _, ax = plt.subplots(figsize = (15, 5))
    ax.spines[['bottom', 'top', 'right']].set_visible(False)
    base = [0 for _ in range(len(y))]
    firstF, firstM = 0, 0
    for i in range(len(y)):
        if y[i] < 0:
            plt.plot([x[i], x[i]], [0, y[i]], color="goldenrod", linewidth=1, zorder=-1)
            firstM += 1
        if y[i] > 0:
            plt.plot([x[i], x[i]], [0, y[i]], color="rebeccapurple", linewidth=1, zorder=-1)
            firstF += 1
    plt.text(0, 39, f'Term for women came first ({(firstF/(firstF + firstM))*100:.1f}%)', verticalalignment='center')
    plt.text(0, -39, f'Term for men came first ({firstM/(firstF + firstM)*100:.1f}%)', verticalalignment='center')

    plt.plot([-2] + x + [len(x)+2], [0] + base + [0], color='black', linewidth=1)
    plt.scatter(x, y, c='black', marker='o', s=6, zorder=1)
    plt.margins(0)
    plt.axis([-2, len(x)+2, -40, 40])
    plt.yticks([i for i in range(-40, 41, 10)], [abs(i) for i in range(-40, 41, 10)])
    plt.ylabel('Years Between')
    plt.xticks([])
    plt.text(-8, 50, f'Total Paired Terms: {len(y)}. Term for [Men/Women] Came First: {firstF+firstM}.', fontsize=11)
    plt.show()