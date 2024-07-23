import sys
import os
import math
import time
from datetime import datetime

import numpy as np
import pandas as pd
# import sympy as sp
# from sympy import pi
# from sympy.printing import latex
from statistics import mode
from scipy.stats import norm
from scipy.spatial import distance
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures



def isFileExist(fileFullPath):
    import os
    if fileFullPath is not None:
        if os.path.isfile(fileFullPath):
            if os.path.exists(fileFullPath):
                return True
            else:
                return False
        else:
            return False
    else:
        return False
    
    
def plot_all_populations_samples(x_data=list(), y_data=list(), graphs_folder_name='./graphs/', polynomial_degree=2, show_graphs=True, plt=None):
    # import matplotlib.pyplot as plt
    if plt is None:
        return
    
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False,
                           subplot_kw={'facecolor': 'white'}, gridspec_kw={})
    ax.grid(True, which='both', axis='both', lw=1, ls='--', c='.2')
    ax.set_xlabel('samples sizes')
    ax.set_ylabel('means values')
    plt.scatter(x=x_data, y=y_data, edgecolors='none', linewidths=0, c='red', marker='o', alpha=0.2, label='All populations/samples')
    ax.legend()
    if show_graphs: 
        fig.show()
    fig.savefig(os.path.dirname(graphs_folder_name) + '/' + 'all_populations_samples.png', transparent=True)
    # Plot linear regressions
    x_max = max(x_data)
    sample_segment = min( 5, int(x_max/min(x_max, 1000 )))
    
    linear_model, polynomial_model = MyLinearRegression(X=np.array(x_data).reshape(len(x_data), 1), 
                        y=np.array(y_data).reshape(len(y_data), 1), 
                        sample_size=len(x_data), sample_increment=sample_segment, reg_type=['linear', 'polynomial'], 
                        degree=polynomial_degree, ax=ax)
    # test returned model for some candidates
    print('\n-----------------------------------\n')
    print('\nTEST OF LINEAR REGRESSION\n')
    print('\n-----------------------------------\n')
    if linear_model is not None:
        slope = linear_model.coef_
        intercept = linear_model.intercept_
        # y_vals = model.predict(X_vals)
        pop_size = 1000
        s_size = 500
        sample = np.random.choice(a=pop_size, size=s_size, replace=False)
        predicted_mean_value = linear_model.predict(sample.reshape(-1, 1 ))
        print('Test Sample vector: ' + str(sample) + 'nPredicted returned mean vector: ' + str(predicted_mean_value))
        predicted_one_value = slope*sample[0] + intercept
        print('\nTest Sample value: ' + str(sample[0]) + '\nPredicted returned mean value: ' + str(predicted_one_value[0][0]))
    
    # print('\n-----------------------------------\n')
    # print('\nTEST OF POLYNOMIAL REGRESSION\n')
    # print('\n-----------------------------------\n')
    # if polynomial_model is not None:
    #     y_vals = model.predict(X_vals)
    #     pop_size = 1000
    #     s_size = 500
    #     sample_value = np.random.random_integers( low=0,high=pop_size, size=1)
    #     predicted_mean_value = polynomial_model.predict(np.array([sample_value]).reshape(-1, 1), np.array([0.5]))
    #     print('Test Sample value: ' + str(sample) + '\nPredicted returned mean value: ' + str(predicted_mean_value))
    
    # plt.close()
    

def normal_function(mu, sigma, x):
    return ( 1/(sigma*math.sqrt(2*math.pi)) )*math.exp(( -pow(x-mu, 2)/(2*sigma*sigma)) )
    

def plot_2d_graph(x_data=list(), y_data=list(), plot_params = None, ax=None):
    # import matplotlib.pyplot as plt
    # set default used plot parameters if passed in plot_params
    # print('x_data : ' + str(x_data)) 
    # print('y_data : ' + str(y_data))
    default_plot_params = {'title': '', 'fontsize': '10', 'fontname': 'arial', 'color': '#000000', 'x_label': 'variable',
                         'y_label': 'Value', 'style': '+-b', 'x_step': (max(x_data)-min(x_data))/10}
    for key in default_plot_params.keys():
        if key in plot_params.keys():
            if not(plot_params[key] is None):
                default_plot_params[key] = plot_params[key]

    # define plot figure instance and axes instances (1x1)
    # fig, ax = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False,
    #                      subplot_kw={'facecolor': 'white'},
    #                       gridspec_kw={})
    # plt.grid(True, which='major', axis='both', lw=1, ls='--', c='.75')
    if not(ax is None):
        if np.shape(ax)==(2,2):
            ax[0, 0].plot(x_data, y_data, default_plot_params['style'], linewidth=0, label='sample points')
            # set ticks list as 10 major ticks by default
            x_ticks = np.arange(x_data[0], x_data[len(x_data)-1] + default_plot_params['x_step'], default_plot_params['x_step'])
            ax[0, 0].set_xticks(x_ticks)
            # set labels
            ax[0, 0].set_xlabel(default_plot_params['x_label'], labelpad=5, fontsize=10, fontname='serif', color=default_plot_params['color'])
            ax[0, 0].set_ylabel(default_plot_params['y_label'], labelpad=5, fontsize=10, fontname='serif', color=default_plot_params['color'])
            # set graph title
            ax[0, 0].set_title(default_plot_params['title'], fontsize=default_plot_params['fontsize'],
                        fontname=default_plot_params['fontname'], color=default_plot_params['color'])
            ax[0,0].legend(loc="upper right")
            
            
            # Plot corresponding normal distribution
            # Fit a normal distribution
            mu, sigma = norm.fit(y_data)

            # Plotting the histogram and fitted normal distribution
            ax[0, 1].hist(y_data, density=False, histtype='stepfilled', label='Samples counts histogram', alpha=0.7)
            ax[0, 1].set_title('Samples data distribution histogram', fontsize=10, color=default_plot_params['color'])
            ax[0, 1].legend()
            
            # box plot to show more precise parameters
            ax[1, 0].boxplot(y_data)
            ax[1, 0].set_title('Sample data box plot', fontsize=10, color=default_plot_params['color'])
            ax[1, 0].legend()
            
            # fit y_data to to a normal distribution
            # print('Max y_data: ' + str(max(y_data)))
            # print('Min y_data: ' + str(min(y_data)))
            x = np.linspace(min(y_data), max(y_data), np.size(y_data, axis=0))
            ax[1, 1].set_xlabel('mean value', color=default_plot_params['color'])
            ax[1, 1].set_ylabel('samples size', color=default_plot_params['color'])
            
    
            ax[1, 1].hist(y_data, density=True, histtype='step', label='Samples counts histogram', alpha=1)
            ax[1, 1].set_title('Fitted normal distribution\n(' + str('\u03C3') + "=" + str(round(sigma, 5)) + '  ' + str(r'$\mu=$') + str(round(mu, 5)), 
                               fontsize=10, color=default_plot_params['color'])
            ax[1, 1].plot(x, norm.pdf(x, mu, sigma),'b-', linewidth=2, label='Fitted Normal distribution')
            
            # plot other metrics (mean, median, mode, sigma) as vertical lines
            ax[1, 1].axvline(np.mean(y_data), ymin=0, ymax=np.max(x_data), linewidth=2, color='r', linestyle='dashed', label=str('$\mu$'), )
            ax[1, 1].axvline(np.median(y_data),  ymin=0, ymax=np.max(x_data), linewidth=2, color='g', linestyle='dotted', label='median')
            # ax[1, 1].axvline(mode(y_data),  ymin=0, ymax=np.max(x_data), linewidth=2, color='orange', linestyle='dashdot', label='mode') # the mode has no meanning in this case
            # ax[1, 1].axvline(sigma,  ymin=0, ymax=np.max(x_data), linewidth=2, color='lime', linestyle='--', label=str('\u03C3'))
            
            
            # add lines corresponding to (1sigma, 2sigma, 3sigma, -1sigma, -2sigma, -3sigma)
            my_points = [[str('\u03C3'), sigma], [str('2\u03C3'), 2*sigma],  [str('3\u03C3'), 3*sigma],
                         [str('-\u03C3'), -sigma], [str('-2\u03C3'), -2*sigma],  [str('-3\u03C3'), -3*sigma]]
            for x in my_points:
                y = normal_function(mu=mu, sigma=sigma, x=x[1] + mu)    # make it centered at 0
                ax[1, 1].axvline(x[1] + mu, ymin=0, ymax=y, linewidth=1, color='lime', linestyle='--', label=str(x[0]))
                # ax[1, 1].set_xticks(ticks=x[1], labels=x[0])
            



def PolynomialRegessionOfMeanSamples(x, y, mode='linear', degree = 2, p_max_size = 1000):
    # see : https://www.askpython.com/python/examples/polynomial-regression-in-python#google_vignette
    # fitting the linear regression model
    import matplotlib.pyplot as plt

    if mode=='linear':
        lin_reg = LinearRegression()
        # print('x=' + str(x))
        # print('y=' + str(y))
        lin_reg.fit(x, y)
        
        # visualising the linear regression model
        plt.scatter(x, y, color='red')  # real samples generated values
        x_values = np.linspace(min(p_max_size, 100 ), min(p_max_size, 500 ),  p_max_size ).reshape(-1, 1)
        # print('x_values=' + str(x_values))
        plt.plot(x_values, lin_reg.predict(x_values), color='blue') # this is the linear regression curve
        plt.title("Sample génerated points")
        plt.xlabel('random generated values')
        plt.ylabel('mean of samples')
        plt.show()
    elif mode=='polynomial':
        # polynomial regression model
        poly_reg = PolynomialFeatures(degree=degree)
        x_poly = poly_reg.fit_transform(x)
        lin_reg = LinearRegression()
        lin_reg.fit(x_poly,y)
        
        # visualising polynomial regression
        X_grid = np.arange(min(x),max(x),0.01)
        X_grid = X_grid.reshape(len(X_grid),1) # vertical grid
        plt.scatter(x, y, color='red') 
        plt.plot(X_grid, lin_reg.predict(poly_reg.fit_transform(X_grid)), color='blue') 
        plt.title("Polynomial predited curve")
        plt.xlabel('Position level')
        plt.ylabel('mean of samples')
        plt.show()
    
    plt.close()
    
    return 0


def GenerateSamples(p_max_size=1000, out_csv_file='output.csv', generate_graphs = True, polynomial_degree = 2, graphs_folder_name='./graphs/', show_graphs=True):
    ret = 0
    try:
        # Variablize the sampling size (25, 50, 100, 150...) and plot corresponding means
        # Generate a random population
        p_max_size = p_max_size # population size
        population = np.random.randn(p_max_size)
        # print ('MAX population size: ' + str(p_max_size))
        import matplotlib.pyplot as plt
        
        out_df = pd.DataFrame(columns=['rank', 'population_name', 'population_size', 'sample_name', 'sample_size', 'sample_mean'])
        idx = 0
        i = 0
        j = 0
        all_samples = []
        all_means = []
        
        for p_size in range(min(p_max_size, 100 ), p_max_size, min(p_max_size, 200 ) ):
            # print ('\tPopulation size: ' + str(p_size))
            sample_segment = min( 5, int(p_max_size/min(p_max_size, 1000 ))) # int(p_max_size/10)
            i = i + 1
            means = []
            sample_sizes = []
            
            # print ('sample_segment: ' + str(sample_segment))
            for s_size in range(sample_segment, p_size, sample_segment):
                # print ('\t\tSample size: ' + str(s_size))
                sample = np.random.choice(a=population, size=s_size, replace=False)
                sample_sizes.append( s_size )
                mean = np.mean(a=sample, axis=0)
                means.append(mean) 
                j = j + 1
                
                out_df.loc[idx] = [idx, 'population_' + str(i), p_size, 'sample_' + str(j), s_size, mean]
                idx = idx + 1
                
            # print('sample_sizes = ' + str(sample_sizes))
            # print('means = ' + str(means))
            
            if generate_graphs==True:
                # generate plot figure instance and axes instances (here 2x2)
                fig, ax = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False,
                                        subplot_kw={'facecolor': 'white'}, gridspec_kw={})
                plt.grid(True, which='major', axis='both', lw=1, ls='--', c='.75')
                
                plot_2d_graph(sample_sizes, means,
                                        {'title': 'Sample size means\n(' + 'Population size: ' + str(p_size) + ' - MAX samples size: ' + 
                                        str(s_size) + ' - Sample increment: ' + str(sample_segment) + ')\n& Linear regression (line/polynomial)', 
                                        'fontsize': '10', 'fontname': 'arial', 'color': '#000000', 
                                        'x_label': 'sample size', 'y_label': 'sample mean value', 'style': '+-r', 'x_step': None}, 
                                        ax=ax)
                
                MyLinearRegression(X=np.array(sample_sizes).reshape(len(sample_sizes), 1), 
                                y=np.array(means).reshape(len(means), 1), 
                                sample_size=s_size, sample_increment=sample_segment, reg_type=['linear', 'polynomial'], 
                                degree=polynomial_degree, ax=ax)
                
                plt.legend(loc="upper left")
                fig.set_size_inches(12, 12)
                fig.tight_layout() 
                if show_graphs: 
                    fig.show()
                fig.savefig(os.path.dirname(graphs_folder_name) + '/' + str(s_size) + '.png', transparent=True)
            
            # all_samples.extend(sample_sizes)
            # all_means.extend(means)
            all_samples = all_samples + sample_sizes
            all_means = all_means + means
            del means
            del sample_sizes
        
        # plot all populations/samples in one graph and plot linear regseions to see progress
        plot_all_populations_samples(all_samples, all_means, graphs_folder_name, polynomial_degree=polynomial_degree, show_graphs=show_graphs, plt=plt)
        
        # set out_df index column
        out_df.set_index('rank')    
        
        # if isFileExist(out_csv_file):
            # write output to csv file for populations and samples
        out_df.to_csv(out_csv_file, sep=';', encoding='utf-8')
    except:
        print('Exception occured in function "GenerateSamples". Please check passed parameters.')
        ret = 1
    finally:
        plt.close()
        return ret
    


def MyLinearRegression(X, y, sample_size=100, sample_increment=1, reg_type=['linear'], degree=1, ax=None):
    ret_model = None
    if sample_increment<=0 | sample_size<=0:
        ret_model = None
        return ret_model
      
    try:
        linear_model = LinearRegression()
        polynomial_model = LinearRegression()
        
        if 'polynomial' in reg_type:
            poly_features = PolynomialFeatures(degree=degree, include_bias=False)
            X_poly = poly_features.fit_transform(X=X)
            polynomial_model.fit( X=X_poly, y=y )
            X_vals = np.linspace(0, sample_size, int(round(sample_size/sample_increment, 0))).reshape(-1, 1)
            X_vals_poly = poly_features.transform(X_vals)
            y_vals = polynomial_model.predict(X_vals_poly)    
            if not(ax is None):
                # ax[0, 0].scatter(X, y, c='r')
                if isinstance(ax, np.ndarray):
                    ax[0, 0].plot(X_vals, y_vals, color='lime', linewidth=2, label='Polynomial regression curve')
                    ax[0, 0].legend(loc="upper right")
                else:
                    ax.plot(X_vals, y_vals, color='lime', linewidth=2, label='Polynomial regression curve')
                    ax.legend(loc="upper right")
        
        if 'linear' in reg_type:
            linear_model.fit( X=X, y=y )
            X_vals = np.linspace(0, sample_size, int(round(sample_size/sample_increment, 0))).reshape(-1, 1)
            y_vals = linear_model.predict(X_vals)
            print( 'Linear regression:\n\tLine slope= {0}\nLine intersept point= {1}'.format(linear_model.coef_[0][0], linear_model.intercept_[0]) )
            
            if not(ax is None):
                # ax[0, 0].scatter(X, y, c='r')
                if isinstance(ax, np.ndarray):
                    ax[0, 0].plot(X_vals, y_vals, color='b', linewidth=1, label='Linear regression curve')
                    ax[0, 0].legend(loc="upper right")
                else:
                    ax.plot(X_vals, y_vals, color='b', linewidth=1, label='Linear regression curve')
                    ax.legend(loc="upper right")
        
        ret_model = linear_model, polynomial_model
    except:
        print('Exception occured in funtion MyLinearRegression. Please check passed parameters. No model has been returned')
        ret_model = None, None
    finally:
        return ret_model
   

def testLinearRegression(f, X):
    import matplotlib.pyplot as plt1
    y = f(X) # 4 + 2*X + 5*X**2 + np.random.rand(100, 1)
    
    reg = LinearRegression()
    reg.fit( X=X, y=y )
    
    X_vals = np.linspace(0, 1, 100).reshape(-1, 1)
    y_vals = reg.predict(X_vals)
    
    plt1.scatter(X, y, c='r')
    plt1.plot(X_vals, y_vals, color='b')
    plt1.show()
    

def testPlynomialRegression(f, X, degree = 2):
    import matplotlib.pyplot as plt2
    y = f(X)
    
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly_features.fit_transform(X=X)
    
    reg = LinearRegression()
    reg.fit( X=X_poly, y=y )
    
    X_vals = np.linspace(0, 1, 100).reshape(-1, 1)
    X_vals_poly = poly_features.transform(X_vals)
    y_vals = reg.predict(X_vals_poly)
    
    plt2.scatter(X, y, c='r')
    plt2.plot(X_vals, y_vals, color='b')
    plt2.show()

# main function
def my_main(population_size=1000, csv_output_file_path='./output.csv', generate_graphs=True, polynomial_degree=2, graphs_folder_name='./graphs/', show_graphs=True):
    # Start measuring time
    start_time = time.monotonic()
    ret = GenerateSamples(p_max_size = population_size, out_csv_file = csv_output_file_path, 
                          generate_graphs = generate_graphs, polynomial_degree=polynomial_degree, graphs_folder_name=graphs_folder_name, show_graphs=show_graphs)
    # Calculate the duration in seconds
    duration = time.monotonic() - start_time
    print(f"Elapsed time: {duration:.2f} seconds")
    return ret

    X = np.random.rand(100, 1)
    def f1(x):
        return 4 + 2*X + 5*X**2 + np.random.rand(100, 1)
    ret = testLinearRegression(f1, X)
    
    X = np.random.rand(100, 1)
    def f2(x):
        return 4 + 2*X + 5*X**2 + np.random.rand(100, 1)
    ret = testPlynomialRegression(f2, X, degree=20)
    
    
    # ret = testPlynomialRegression(degree=4)
    return ret
    
# main usage
def Usage():
    example ="EXAMPLE :\n" + sys.argv[0] + '1000  ./data/output_1000.csv 1 2'
    print("USAGE:")
    print(str(sys.argv[0]) + " /? [population_size] [csv_output_file_path] [generate_graphs] [polynomial_degree] [show_graphs] [graphs_folder_name]")
    print("\tPARAMETERS:\n\t\t/?: type this help\n\t\tpopulation_size: Size of population to be genrated (ex. 1000, 10000...). Default is 1000  \n \
            \tcsv_output_file_path: full path to all generated data in CSV indicated file. Default is output.csv \n \
            \tgenerate_graphs: 1 (generate all graphs) or 0 otherwise. Default is 1 \n \
            \tpolynomial_degree: 2 (Linear polynomial regession degree, starting from 2). Default set to 2 \n \
            \tshow_graphs: 1 to show graphs, anything else if not. Default set to 1 \n \
            \tgraphs_folder_name: folder path name where generated graphics will be saved \n\n" + example)
    
    
# program entry point
if __name__ == '__main__':
    if len(sys.argv)==2:
        Usage()
        sys.exit()
    else:
        # 1st parameter
        population_size = 1000
        if len(sys.argv)>=2:
            if (sys.argv[1] is not None): population_size = int(sys.argv[1])
        
        csv_output_file_path ='./data/output.csv'
        if len(sys.argv)>=3: 
            if (sys.argv[2] is not None): 
                if isFileExist(sys.argv[2]): 
                    t =  datetime.fromtimestamp(time.time())
                    csv_output_file_path = os.path.splitext(os.path.basename(sys.argv[2]))[0] + '-' + str(format(t, '%Y-%m-%d-%I-%M%S%p')) + os.path.splitext(os.path.basename(sys.argv[2]))[1]
        # 2nd parameter
        generate_graphs = True
        if len(sys.argv)>=4: 
            if (sys.argv[3] is not None): 
                generate_graphs = int(sys.argv[3])==1
                    
        # 3thd parameter
        polynomial_degree = 2
        if len(sys.argv)>=5: 
                if (sys.argv[4] is not None): 
                    polynomial_degree = int(sys.argv[4]) if int(sys.argv[4])>=2 else 2   
        
        graphs_folder_name = './graphs/'
        if len(sys.argv)>=6: 
                if (sys.argv[5] is not None): 
                    if isFileExist(sys.argv[5]):
                        l = len(sys.argv[5])
                        graphs_folder_name = sys.argv[5] if sys.argv[5][:l-1]=='/' else sys.argv[5]+'/'
       
        show_graphs = 1
        if len(sys.argv)>=7: 
                if (sys.argv[6] is not None): 
                    show_graphs =int(sys.argv[6])==1
        
        print(f"Passed PARAMETERS:\n \
            population_size: {population_size}  \n \
            csv_output_file_path: {csv_output_file_path} \n \
            generate_graphs: {generate_graphs} \n \
            polynomial_degree: {polynomial_degree} \n \
            show_graphs: {show_graphs} \n \
            graphs_folder_name: {graphs_folder_name} \n\n".format('{0:%d}{1:%s}{2:%d}{3:%d}{4:%s}{5:%s}'))
            
        sys.exit(my_main(population_size, csv_output_file_path, generate_graphs, polynomial_degree, graphs_folder_name, show_graphs))