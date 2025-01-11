import sys
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    """Calcule la valeur de la fonction de densité normale"""
    return (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-(x - mu)**2 / (2 * sigma**2))
    

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
            mean_y = norm.pdf(np.mean(y_data), mu, sigma)  # Hauteur de la courbe à la moyenne
            median_y = norm.pdf(np.median(y_data), mu, sigma)  # Hauteur de la courbe à la médiane

            ax[1, 1].axvline(np.mean(y_data), ymin=0, ymax=mean_y, 
                             linewidth=2, color='r', linestyle='dashed', label=str('$\mu$'))
            ax[1, 1].axvline(np.median(y_data), ymin=0, ymax=median_y, 
                             linewidth=2, color='g', linestyle='dotted', label='median')

            # add lines corresponding to (1sigma, 2sigma, 3sigma, -1sigma, -2sigma, -3sigma)
            my_points = [[str('\u03C3'), sigma], [str('2\u03C3'), 2*sigma], [str('3\u03C3'), 3*sigma],
                         [str('-\u03C3'), -sigma], [str('-2\u03C3'), -2*sigma], [str('-3\u03C3'), -3*sigma]]
            for point in my_points:
                point_x = point[1] + mu  # Point sur l'axe x
                point_y = norm.pdf(point_x, mu, sigma)  # Hauteur de la courbe à ce point
                # Convertir en coordonnées relatives pour ymax (entre 0 et 1)
                ymax = point_y / ax[1, 1].get_ylim()[1]
                ax[1, 1].axvline(point_x, ymin=0, ymax=ymax, 
                                 linewidth=1, color='lime', linestyle='--', label=str(point[0]))
            



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


class Population:
    def __init__(self, size=1000):
        self.size = size
        self.data = np.random.randn(size)

    def generate_sample(self, sample_size, replace=False):
        return np.random.choice(a=self.data, size=sample_size, replace=replace)

class SampleAnalyzer:
    def __init__(self, population, sample_segment=5):
        self.population = population
        self.sample_segment = sample_segment
        self.samples = []
        self.means = []
        
    def analyze_samples(self, max_sample_size):
        sample_sizes = []
        means = []
        
        for s_size in range(self.sample_segment, max_sample_size, self.sample_segment):
            sample = self.population.generate_sample(s_size)
            sample_sizes.append(s_size)
            means.append(np.mean(a=sample, axis=0))
            
        return sample_sizes, means

class DataVisualizer:
    def __init__(self, plt=None):
        self.plt = plt if plt else __import__('matplotlib.pyplot').pyplot
        self.show_graphs = True
        self.graphs_folder_name = './graphs/'

    def plot_2d_graph(self, x_data, y_data, plot_params, ax=None):
        default_plot_params = {
            'title': '', 'fontsize': '10', 'fontname': 'arial', 
            'color': '#000000', 'x_label': 'variable',
            'y_label': 'Value', 'style': '+-b', 
            'x_step': (max(x_data)-min(x_data))/10
        }
        
        for key in default_plot_params:
            if key in plot_params and plot_params[key] is not None:
                default_plot_params[key] = plot_params[key]

        if ax is not None and isinstance(ax, np.ndarray):
            ax[0, 0].plot(x_data, y_data, default_plot_params['style'], 
                         linewidth=0, label='sample points')
            ax[0, 0].set_xlabel(default_plot_params['x_label'])
            ax[0, 0].set_ylabel(default_plot_params['y_label'])
            ax[0, 0].set_title(default_plot_params['title'])
            ax[0, 0].legend()
            
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
            mean_y = norm.pdf(np.mean(y_data), mu, sigma)  # Hauteur de la courbe à la moyenne
            median_y = norm.pdf(np.median(y_data), mu, sigma)  # Hauteur de la courbe à la médiane

            ax[1, 1].axvline(np.mean(y_data), ymin=0, ymax=mean_y, 
                             linewidth=2, color='r', linestyle='dashed', label=str('$\mu$'))
            ax[1, 1].axvline(np.median(y_data), ymin=0, ymax=median_y, 
                             linewidth=2, color='g', linestyle='dotted', label='median')

            # add lines corresponding to (1sigma, 2sigma, 3sigma, -1sigma, -2sigma, -3sigma)
            my_points = [[str('\u03C3'), sigma], [str('2\u03C3'), 2*sigma], [str('3\u03C3'), 3*sigma],
                         [str('-\u03C3'), -sigma], [str('-2\u03C3'), -2*sigma], [str('-3\u03C3'), -3*sigma]]
            for point in my_points:
                point_x = point[1] + mu  # Point sur l'axe x
                point_y = norm.pdf(point_x, mu, sigma)  # Hauteur de la courbe à ce point
                # Convertir en coordonnées relatives pour ymax (entre 0 et 1)
                ymax = point_y / ax[1, 1].get_ylim()[1]
                ax[1, 1].axvline(point_x, ymin=0, ymax=ymax, 
                                 linewidth=1, color='lime', linestyle='--', label=str(point[0]))
            

    def plot_all_populations_samples(self, x_data, y_data, graphs_folder_name, polynomial_degree=2, show_graphs=True):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.grid(True)
        ax.set_xlabel('samples sizes')
        ax.set_ylabel('means values')
        ax.scatter(x_data, y_data, c='red', alpha=0.2, label='All populations/samples')
        ax.legend()
        
        plt.tight_layout()
        
        # Sauvegarder avant d'afficher
        plt.savefig(os.path.join(graphs_folder_name, 'all_populations_samples.png'), 
                    bbox_inches='tight', dpi=300)
        
        if show_graphs:
            plt.show()
        
        plt.close(fig)

    def test_linear_regression(self, f, X):
        y = f(X)
        reg = LinearRegression()
        reg.fit(X=X, y=y)
        
        X_vals = np.linspace(0, 1, 100).reshape(-1, 1)
        y_vals = reg.predict(X_vals)
        
        plt.scatter(X, y, c='r')
        plt.plot(X_vals, y_vals, color='b')
        if self.show_graphs:
            plt.show()
        plt.savefig(os.path.join(self.graphs_folder_name, 'linear_regression_test.png'))
        plt.close()

    def test_polynomial_regression(self, f, X, degree=2):
        y = f(X)
        poly_features = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly_features.fit_transform(X=X)
        
        reg = LinearRegression()
        reg.fit(X=X_poly, y=y)
        
        X_vals = np.linspace(0, 1, 100).reshape(-1, 1)
        X_vals_poly = poly_features.transform(X_vals)
        y_vals = reg.predict(X_vals_poly)
        
        plt.scatter(X, y, c='r')
        plt.plot(X_vals, y_vals, color='b')
        if self.show_graphs:
            plt.show()
        plt.savefig(os.path.join(self.graphs_folder_name, 'polynomial_regression_test.png'))
        plt.close()

class SampleGenerator:
    def __init__(self, population_size=1000, output_file='output.csv', 
                 generate_graphs=True, polynomial_degree=2, 
                 graphs_folder_name='./graphs/', show_graphs=True):
        self.population_size = population_size
        self.output_file = output_file
        self.generate_graphs = generate_graphs
        self.polynomial_degree = polynomial_degree
        self.graphs_folder_name = graphs_folder_name
        self.show_graphs = show_graphs
        
        self.population = Population(population_size)
        self.visualizer = DataVisualizer()
        self.out_df = pd.DataFrame(columns=['rank', 'population_name', 'population_size', 
                                          'sample_name', 'sample_size', 'sample_mean'])

    def generate_samples(self):
        try:
            idx = 0
            i = 0
            j = 0
            all_samples = []
            all_means = []

            for p_size in range(min(self.population_size, 100), 
                              self.population_size, 
                              min(self.population_size, 500)):
                
                analyzer = SampleAnalyzer(self.population)
                sample_sizes, means = analyzer.analyze_samples(p_size)
                
                # Mise à jour du DataFrame
                for s_size, mean in zip(sample_sizes, means):
                    j += 1
                    self.out_df.loc[idx] = [idx, f'population_{i}', p_size, 
                                          f'sample_{j}', s_size, mean]
                    idx += 1

                if self.generate_graphs:
                    self._generate_graphs(sample_sizes, means, p_size)

                all_samples.extend(sample_sizes)
                all_means.extend(means)

            # Génération du graphique final
            if self.generate_graphs:
                self.visualizer.plot_all_populations_samples(
                    all_samples, all_means, self.graphs_folder_name,
                    self.polynomial_degree, self.show_graphs
                )

            # Sauvegarde des données
            self.out_df.set_index('rank')
            self.out_df.to_csv(self.output_file, sep=';', encoding='utf-8')
            return 0

        except Exception as e:
            print(f'Exception dans SampleGenerator.generate_samples: {str(e)}')
            return 1

    def _generate_graphs(self, sample_sizes, means, p_size):
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
        plot_params = {
            'title': f'Sample size means\nPopulation size: {p_size}',
            'x_label': 'sample size',
            'y_label': 'sample mean value'
        }
        self.visualizer.plot_2d_graph(sample_sizes, means, plot_params, ax=ax)
        
        plt.tight_layout()  # Ajout de tight_layout pour bien organiser les sous-graphiques
        
        # Sauvegarder avant d'afficher
        plt.savefig(os.path.join(self.graphs_folder_name, f'{p_size}.png'), 
                    bbox_inches='tight', dpi=300)
        
        if self.show_graphs:
            plt.show()
        
        plt.close(fig)  # Fermer spécifiquement cette figure

def my_main(population_size=1000, csv_output_file_path='./output.csv', 
            generate_graphs=True, polynomial_degree=2, 
            graphs_folder_name='./graphs/', show_graphs=True):
    """
    Fonction principale gérant la génération et l'analyse des échantillons
    """
    try:
        # Start measuring time
        start_time = time.monotonic()
        
        # Création et exécution du générateur d'échantillons
        generator = SampleGenerator(
            population_size=population_size,
            output_file=csv_output_file_path,
            generate_graphs=generate_graphs,
            polynomial_degree=polynomial_degree,
            graphs_folder_name=graphs_folder_name,
            show_graphs=show_graphs
        )
        
        ret = generator.generate_samples()
        
        # Tests additionnels de régression
        if generate_graphs:
            # Création d'un échantillon de test
            X = np.random.rand(100, 1)
            
            # Test de régression linéaire
            def f1(x): return 4 + 2*X + 5*X**2 + np.random.rand(100, 1)
            generator.visualizer.test_linear_regression(f1, X)
            
            # Test de régression polynomiale
            def f2(x): return 4 + 2*X + 5*X**2 + np.random.rand(100, 1)
            generator.visualizer.test_polynomial_regression(f2, X, degree=polynomial_degree)
        
        # Calculate the duration in seconds
        duration = time.monotonic() - start_time
        print(f"Temps écoulé: {duration:.2f} secondes")
        
        return ret
        
    except Exception as e:
        print(f"Erreur dans my_main: {str(e)}")
        return 1


   
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
    

def test_main():
    """
    Fonction de test pour valider différents scénarios d'utilisation de my_main
    """
    print("\n=== DÉBUT DES TESTS DE LA FONCTION MAIN ===\n")
    
    tests = [
        {
            "name": "Test 1: Paramètres par défaut",
            "params": {},
            "expected": 0
        },
        {
            "name": "Test 2: Grande population",
            "params": {
                "population_size": 10000,
                "polynomial_degree": 3
            },
            "expected": 0
        },
        {
            "name": "Test 3: Sans génération de graphiques",
            "params": {
                "generate_graphs": False,
                "show_graphs": False
            },
            "expected": 0
        },
        {
            "name": "Test 4: Dossier de sortie personnalisé",
            "params": {
                "csv_output_file_path": "./test_output/test.csv",
                "graphs_folder_name": "./test_graphs/"
            },
            "expected": 0
        }
    ]
    
    for test in tests:
        print(f"\n--- {test['name']} ---")
        try:
            # Créer les dossiers nécessaires si spécifiés
            if 'csv_output_file_path' in test['params']:
                os.makedirs(os.path.dirname(test['params']['csv_output_file_path']), exist_ok=True)
            if 'graphs_folder_name' in test['params']:
                os.makedirs(test['params']['graphs_folder_name'], exist_ok=True)
                
            # Exécuter le test
            result = my_main(**test['params'])
            
            # Vérifier le résultat
            if result == test['expected']:
                print(f"✅ Test réussi (retour: {result})")
                
                # Vérifications supplémentaires
                if 'csv_output_file_path' in test['params']:
                    if os.path.exists(test['params']['csv_output_file_path']):
                        print("✅ Fichier CSV généré avec succès")
                    else:
                        print("❌ Fichier CSV non généré")
                        
                if test['params'].get('generate_graphs', True):
                    graphs_folder = test['params'].get('graphs_folder_name', './graphs/')
                    if os.path.exists(graphs_folder) and len(os.listdir(graphs_folder)) > 0:
                        print("✅ Graphiques générés avec succès")
                    else:
                        print("❌ Graphiques non générés")
            else:
                print(f"❌ Test échoué (retour: {result}, attendu: {test['expected']})")
                
        except Exception as e:
            print(f"❌ Erreur lors du test: {str(e)}")
            
    print("\n=== FIN DES TESTS ===\n")

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_main()
    else:
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