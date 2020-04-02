# creating visualizations to demonstrate other concepts in lessons
# when the code distracts from the lesson, then the code can be added here.

import matplotlib.pyplot as plt
plt.rc("axes.spines", top=False, right=False)
import seaborn as sns
from sklearn.linear_model import LinearRegression

def evaluation_example1(df, x, y):
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, color='dimgray')
    
    # add title
    title_string = 'Where is the line of best fit?'
    plt.title(title_string, fontsize=12, color='black')
    
    # add axes labels
    plt.ylabel('final grade')
    plt.xlabel('exam 1')
    
    # add baseline
    plt.annotate('', xy=(70, y.mean()), xytext=(100, y.mean()), xycoords='data', textcoords='data', arrowprops={'arrowstyle': '-', 'color': 'darkseagreen'})
    plt.text(100, 83,  'This line?', {'color': 'black', 'fontsize': 11, 'ha': 'left', 'va': 'center'})

    # add line connecting min and max of y
    plt.annotate('', xy=(70, y.min()), xytext=(100, y.max()), xycoords='data', textcoords='data', arrowprops={'arrowstyle': '-', 'color': 'darkseagreen'})
    plt.text(100.5, 96,  'This line?', {'color': 'black', 'fontsize': 11, 'ha': 'left', 'va': 'center'})
    
    # add line that is translated up the y-axis a few points from the min/max line
    plt.annotate('', xy=(70, y.min()+3), xytext=(100, y.max()+3), xycoords='data', textcoords='data', arrowprops={'arrowstyle': '-', 'color': 'darkseagreen'})
    plt.text(100, 99,  'This line?', {'color': 'black', 'fontsize': 11, 'ha': 'left', 'va': 'center'})

    return plt.show()


def evaluation_example2(df, x, y):
    plt.figure(figsize=(8, 5))

    # Plot regression line
    plt.plot(x, df.yhat, color='darkseagreen',linewidth=3)
    # add label to the regression line
    plt.annotate('', xy=(87,93), xytext=(90,89), xycoords='data', textcoords='data', arrowprops={'arrowstyle': '<-', 'color': 'black'})
    plt.text(87,93,  r'$\hat{y}=12.5 + .85x$', {'color': 'black', 'fontsize': 11, 'ha': 'right', 'va': 'center'})
    plt.text(80.5,95,  'This line!', {'color': 'black', 'fontsize': 11, 'ha': 'left', 'va': 'center'})

    # Plot the data points
    plt.scatter(x, y, color='dimgray')

    # add title
    title_string = 'Where is the line of best fit?'
    plt.title(title_string, fontsize=12, color='black')
    
    # add axes labels
    plt.ylabel('final grade')
    plt.xlabel('exam 1')

    # add baseline
    plt.annotate('', xy=(70, y.mean()), xytext=(100, y.mean()), xycoords='data', textcoords='data', arrowprops={'arrowstyle': '-', 'color': 'darkseagreen'})
    plt.text(100, 83,  'or this line.', {'color': 'black', 'fontsize': 11, 'ha': 'left', 'va': 'center'})

    # add line connecting min and max of y
    plt.annotate('', xy=(70, y.min()), xytext=(100, y.max()), xycoords='data', textcoords='data', arrowprops={'arrowstyle': '-', 'color': 'darkseagreen'})
    plt.text(100.5, 96,  'or this line...', {'color': 'black', 'fontsize': 11, 'ha': 'left', 'va': 'center'})
    
    # add line that is translated up the y-axis a few points from the min/max line
    plt.annotate('', xy=(70, y.min()+3), xytext=(100, y.max()+3), xycoords='data', textcoords='data', arrowprops={'arrowstyle': '-', 'color': 'darkseagreen'})
    plt.text(100, 99,  'Not this line...', {'color': 'black', 'fontsize': 11, 'ha': 'left', 'va': 'center'})

    return plt.show()

def evaluation_example3(df, x, y, yhat): 
    plt.figure(figsize=(8, 5))
    
    ## plot data points, regression line and baseline
    # plot the data points 
    plt.scatter(x, y, color='dimgray', s=40)

    # plot the regression line
    plt.plot(x, yhat, color='darkseagreen', linewidth=3)

    # add baseline through annotation
    plt.annotate('', xy=(70, y.mean()), xytext=(102, df['y'].mean()), xycoords='data', textcoords='data', arrowprops={'arrowstyle': '-', 'color': 'goldenrod', 'linewidth': 2, 'alpha': .75})

    ## ---------------------------------------------
    ## add line labels
    # the regression line equation
    plt.text(89.5, 90.5, r'$\hat{y}=12.5 + .85x$',
         {'color': 'black', 'fontsize': 11, 'ha': 'center', 'va': 'center', 'rotation': 27})

    # the baseline equation
    plt.text(88, 82, r'$\hat{y}=83$',
         {'color': 'black', 'fontsize': 11, 'ha': 'center', 'va': 'center'})

    ## ---------------------------------------------
    # set and plot title, subtitle, and axis labels
    # set titles
    title_string = r'Difference in Error'
    subtitle_string = "Baseline vs. Regression Line"

    # add titles
    plt.title(subtitle_string, fontsize=12, color='black')
    plt.suptitle(title_string, y=1, fontsize=14, color='black')
    
    # add axes labels
    plt.ylabel('final grade')
    plt.xlabel('exam 1')
    
    ## ----------------------------------------
    # annotate each data point with an error line to the baseline and the error value 
    for i in range(len(df)):
        
        # add error lines from baseline to data points
        plt.annotate('', xy=(x[i]+.1, y[i]), xytext=(x[i]+.1, y.mean()), xycoords='data', textcoords='data', 
                 arrowprops={'arrowstyle': '-', 'color':'goldenrod', 'linestyle': '--', 'linewidth': 2, 'alpha': .5})
        # add error lines from regression line to data points      
        plt.annotate('', xy=(x[i], y[i]), xytext=(x[i], yhat[i]), xycoords='data', textcoords='data', 
                 arrowprops={'arrowstyle': '-', 'color':'darkseagreen', 'linestyle': '--', 'linewidth': 2, 'alpha': .75})

    ## ----------------------------------------
    # annotate some of the error lines with pointers
    # add pointer: the first data point to the regression line
    plt.annotate('', xy=(70.25, 70), xytext=(73, 70), xycoords='data', textcoords='data', arrowprops={'arrowstyle': 'fancy', 'color':'darkseagreen', 'linewidth': 1})

    # add pointer: the last data point to the regression line
    plt.annotate('', xy=(100.25, 97), xytext=(103, 97), xycoords='data', textcoords='data', arrowprops={'arrowstyle': 'fancy', 'color':'darkseagreen', 'linewidth': 1})

    # add pointer: the last data point to the baseline 
    plt.annotate('', xy=(100.25, 90), xytext=(103, 90), xycoords='data', textcoords='data', arrowprops={'arrowstyle': 'fancy', 'color':'goldenrod', 'linewidth': 1})

    ## ----------------------------------------
    ## add text to the annotatations
    # the error of the first data point to the regression line
    plt.text(73, 70, 4.1, ha='left', va='center', color='black')

    # the error of the last data point to the regression line
    plt.text(103, 96, 1.6, ha='left', va='center', color='black')

    # the error of the last data point to the baseline
    plt.text(103, 90, -12.7, ha='left', va='center', color='black')

    ## ----------------------------------------
    
    return plt.show()

def evaluation_example4(df, x, y, r2):
    plt.figure(figsize=(8, 5))

    # Plot regression line
    plt.plot(x, df.yhat, color='darkseagreen',linewidth=3)

    # the regression line equation
    plt.text(89.5, 90.5, r'$r^2 = $'+str(r2),
         {'color': 'black', 'fontsize': 11, 'ha': 'center', 'va': 'center', 'rotation': 27})

    # Plot the data points
    plt.scatter(x, y, color='dimgray')

    # add title
    title_string = 'Strength of the correlation of exam 1 and final grade'
    plt.title(title_string, fontsize=12, color='black')
    
    # add axes labels
    plt.ylabel('final grade')
    plt.xlabel('exam 1')

    return plt.show()


def evaluation_example5(df, x, y):
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, color='dimgray')

    # add the residual line at y=0
    plt.annotate('', xy=(70, 0), xytext=(100, 0), xycoords='data', textcoords='data', arrowprops={'arrowstyle': '-', 'color': 'darkseagreen'})

    # set titles
    plt.title(r'Baseline Residuals', fontsize=12, color='black')
    # add axes labels
    plt.ylabel(r'$\hat{y}-y$')
    plt.xlabel('exam 1')

    # add text
    plt.text(85, 15, r'', ha='left', va='center', color='black')

    return plt.show()