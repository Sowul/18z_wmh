#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

from collections import namedtuple
from copy import deepcopy
import multiprocessing
from operator import attrgetter
from timeit import default_timer as timer
import sys

import matplotlib.pyplot as plt
import numpy as np
from pathos.multiprocessing import ProcessPool
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from tqdm import trange, tqdm

class EvoAlgo:
    """Algorithm used for fidning SVM params."""

    def __init__(self, clf, cv=2, pop_members=30, max_iter=100):
        """Init method.

        Args:
            clf : classifier object implementing 'fit'
                Classfier used for scoring new features, here used for calculating _base_score.

            cv : int, cross-validation generator or an iterable
                Determines the cross-validation splitting strategy,
                see also http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html.

            pop_members : int
                Determines the number of individuals in a population.

            max_iter : int
                Determines how many iterations a evolutionary algorithm runs.

        """
        self.clf = clf
        self.cv = cv
        self.metric = 'neg_log_loss'
        self.pop_members = pop_members
        self.max_iter = max_iter
        self.elite = 2
        self.n_params = 2
        self._individuals = []
        self._Individual = namedtuple('Individual',
                            ['params', 'score'])
        self._best_individuals = []
        self._BestIndividual = namedtuple('BestIndividual',
                               ['gen_num', 'params', 'score'])
        self._gen_score = []
        self._Generations = []
        self._Generation = namedtuple('Generation',
                            ['gen_num', 'mean_score', 'best_ind'])
        self.generation_plot = []

    def __repr__(self):
        return '{}(clf={}, cv={},\n\t\tmetric={}, \n\t\t{}={}, pop_members={})'.format(
            self.__class__.__name__, self.clf.__class__.__name__, self.cv, self.metric, 
            'max_iter', self.max_iter, self.pop_members)

    def _create_individual(self):
        """Return new individual.

        Returns:
            Array of floats.

        """
        return np.reshape(100*np.random.sample((self.n_params, )), (-1, self.n_params))

    def _create_population(self):
        """Return new population.

        Returns:
            Array of individuals (floats).

        """
        population = np.empty((0, self.n_params), dtype=np.int8)
        for i, member in enumerate(range(self.pop_members)):
            population = np.append(population, self._create_individual(),
                                   axis=0)
        return population

    def _get_fitness(self, clf, X, y):
        """Compute the scores based on the testing set for each iteration of cross-validation.

        Args:
            clf : classifier object implementing 'fit'
                Classfier used for scoring new features.

            X : array-like
                The data to fit.

            y : array-like
                The target variable.

        Returns:
            Cross-validation score for a given dataset.

        """
        return cross_val_score(clf, X, y,
                            scoring=self.metric, cv=self.cv, n_jobs=1).mean()

    def _select_parents(self, q=4):
        """Select parents from a population of individuals using tournament selection.

        Args:
            q : int
                Tournament size.

        Returns:
            Array of integers.

        """
        parents = np.empty((0, q), dtype=np.int8)
        for i in range(self.pop_members-self.elite):
            parent = np.random.choice(
                                self.pop_members, size=(1, q), replace=False)
            parents = np.append(parents, parent, axis=0)
        parents = np.amin(parents, axis=1)
        np.random.shuffle(parents)
        return np.reshape(parents, (len(parents)//2, 2))

    def _crossover(self, parents, population):
        """Produce a child solution from two parents using mean crossover scheme.

        Args:
            parents : array of integers
                Indices of parents from a given population.

            population : list of tuples
                List of individuals and their fitness.

        Returns:
            Array of floats.

        """
        new_population = np.empty((0, self.n_params), dtype=np.int8)
        children = np.empty((0, self.n_params), dtype=np.int8)
        population = deepcopy(population)
        for couple in parents:
            first_parent = np.reshape(population[couple[0]].params,
                                                        (-1, self.n_params))
            second_parent = np.reshape(population[couple[1]].params,
                                                        (-1, self.n_params))
            first_child = first_parent + np.multiply(np.random.uniform(size=first_parent.shape), second_parent-first_parent)
            second_child = second_parent + first_parent - first_child
            children = np.append(children, first_child, axis=0)
            children = np.append(children, second_child, axis=0)
        new_population = np.append(new_population, children, axis=0)
        return new_population

    def _mutate(self, new_population):
        """Alter gene values in an individual using the Cauchy distribution.

        Args:
            new_population : array of integers
                Array of individuals.

        Returns:
            Array of floats.

        """
        mutation = np.random.standard_cauchy(size=new_population.shape)
        new_population = new_population + mutation
        return new_population

    def _create_next_generation(self, population):
        """Create next generation.

        Args:
            population : list of tuples
                Previous generation.

        Returns:
            New population, array of floats.

        """
        population = sorted(population, key=attrgetter('score'), reverse=True)
        parents = self._select_parents()
        new_population = self._crossover(parents, population)
        new_popualtion = self._mutate(new_population)
        for i in range(self.elite):
            elitist = np.reshape(population[i].params,
                                (-1, self.n_params))
            new_population = np.append(new_population, elitist, axis=0)
        return new_population

    def _score_ind(self, ind):
        score = self._get_fitness(SVC(kernel='rbf', probability=True, C=ind[0], gamma=ind[1]), self.X, self.y)
        return self._Individual(ind, score)

    def fit(self, X, y):
        """Fit estimator.

        Args:
            X : array-like
                The data to fit.

            y : array-like
                The target variable.

        """
        self.X = np.asarray(X)
        self.y = np.asarray(y).reshape(y.shape[0], )
        self._base_score = cross_val_score(self.clf, self.X, y=self.y,
                                           scoring=self.metric, cv=self.cv).mean()
        print("Base score: {}\n".format(self._base_score))
        self._best_score = self._base_score

        population = self._create_population()
        gen = 0
        total_time = 0

        for i in trange(self.max_iter, desc='Generation', leave=False):
            self.generation_plot.append(population.tolist())
            p = ProcessPool(nodes=multiprocessing.cpu_count())

            start = timer()
            self._individuals = p.map(self._score_ind, population)
            total_time = total_time + timer() - start

            self._Generations.append(self._individuals)

            best = sorted(self._individuals, key=lambda tup: tup.score,
                            reverse=True)[0]
            
            self._best_individuals.append(
                self._BestIndividual(gen, best.params, best.score))
            if (gen == 0):
                self._best_score = self._best_individuals[gen]
            if (best.score > self._best_score.score):
                self._best_score = self._best_individuals[gen]
            else:
                pass

            self._gen_score.append(self._Generation(gen,
            sum([tup[1] for tup in self._individuals])/len(self._individuals),
            self._best_individuals[gen]))
            
            population = self._create_next_generation(self._individuals)
            self._individuals = []
            gen += 1
        else:
            print('gen: {}'.format(gen))
            print('avg time per gen: {0:0.1f}'.format(total_time/gen))

    def get_params(self, ind='best'):
        """Print best set of params.

        Args:
            ind : string, 'best'
                Determines which set of params print.

        Returns:
            C : float
                Penalty parameter C of the error term.
            
            gamma: float
                Kernel coefficient for ‘rbf’.
        """
        if ind == 'best':
            print('Best params:')
            print('C: {}'.format(self._best_score.params[0]))
            print('Gamma: {}'.format(self._best_score.params[1]))
            print('neg_log_loss: {}\n'.format(self._best_score.score))
            return self._best_score.params[0], self._best_score.params[1]
        else:
            print("Proper call should be: .get_params() or .get_params(ind='best')")

    def plot(self):
        """Plot data from the evolutionary algorithm."""
        mean_score = [x[1] for x in self._gen_score]
        best_score = [x[2][2] for x in self._gen_score]

        plt.plot(self._best_score.gen_num, self._best_score.score, 'ro')
        plt.plot([0, len(mean_score)], [self._base_score, self._base_score], 'b--', lw=2, label='wynik bazowy')
        plt.plot(range(len(mean_score)), mean_score, 'k', label='wynik sredni generacji')
        plt.plot(range(len(best_score)), best_score, 'g', label='wynik najlepszego osobnika')
        plt.xlabel('przewidywane prawdopodobienstwo')
        plt.ylabel('- log loss')
        plt.legend(loc=0)
        plt.show()

    def plot_gen(self, gif=False):
        global gen
        gen = 0

        def on_keyboard(event):
            global gen
            if event.key == 'right':
                gen += 1
            elif event.key == 'left':
                gen -= 1

            if gen <= self.max_iter:
                plt.clf()
                plt.scatter([row[0] for row in self.generation_plot[gen]], [row[1] for row in self.generation_plot[gen]])
                plt.draw()
            else:
                sys.exit()

        plt.scatter([row[0] for row in self.generation_plot[gen]], [row[1] for row in self.generation_plot[gen]])
        plt.gcf().canvas.mpl_connect('key_press_event', on_keyboard)
        plt.show()

        if gif:
            for gen in range(self.max_iter):
                plt.clf()
                plt.title('Generation {}'.format(str(gen).zfill(2)))
                plt.xlabel('C')
                plt.ylabel('Gamma')
                plt.scatter([row[0] for row in self.generation_plot[gen]], [row[1] for row in self.generation_plot[gen]])
                plt.savefig('./figs/'+str(gen).zfill(2)+'.png')

if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from sklearn.svm import SVC

    clf = SVC(kernel='rbf', probability=True, C=1.0, gamma='auto')
    ea = EvoAlgo(clf, pop_members=60, max_iter=3)
    iris = load_iris()
    ea.fit(iris.data, iris.target)

    ea.get_params()
    ea.plot()
    ea.plot_gen()