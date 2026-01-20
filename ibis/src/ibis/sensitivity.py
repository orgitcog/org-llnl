from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.pyplot import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.feature_selection import mutual_info_regression, f_regression
from sklearn.linear_model import lars_path
from sklearn.preprocessing import PolynomialFeatures

from trata.sampler import LatinHyperCubeSampler
from ibis.pce_model import PolynomialChaosExpansionModel


def _variance_network_plot(ax,
                           feature_data,
                           response_data,
                           feature_names,
                           response_names,
                           score_function,
                           method_label,
                           degree=2,
                           max_size=20.0,
                           alpha=.5,
                           label_size=12,
                           **kwargs):
    """
    Create a set of network plots based on a given set of data and score function
    for each combination of output and degree.

    Network plots compare scores between different degree of interaction.
    In all plots, each parameter by itself is represented as a node in a graph.
    For plots of degree 2, interactions between 2 parameters are represented
    as an edge between the respective nodes.
    For plots of degree 3 or higher, interactions between the respective parameters
    are represented as a hyper edge i.e. an edge between 3 or more nodes.
    The sizes and thicknesses of the nodes and edges correspond to the scores from the
    given score function.

    Args:
        - ax ([[matplotlib.Axes]]): Array-like of Axes to plot to. Dimension is number of
          outputs by (degree-1)
        - feature_data ([[float]]): Array-like of feature data. Each column is a feature;
          each row is an observation
        - response_data ([[float]]): Array-like of response data. Rows correspond to rows
          in feature data.
        - feature_names ([string]): Array-like of feature names
        - response_names ([string]): Array-like of response names
        - score_function (function): Function which scores features based on importance to
          responses.
        - method_label (string): Label of method used
        - degree: maximum degree of interactions to plot
        - max_size (float): Maximum size of elements in plot. Measured in points
        - alpha (float): Opacity of elements in plot.
        - label_size (int): Font size of labels. Measured in points
        - kwargs: Keyword arguments to be passed to score_function

    Raises:
        -
    """
    assert degree >= 2

    feature_interaction_data, feature_interaction_names, powers = _make_interactions(feature_data,
                                                                                     feature_names,
                                                                                     degree=degree,
                                                                                     interaction_only=True)
    feature_interaction_names = np.array(feature_interaction_names)
    powers_degree = powers.sum(axis=1)

    # Reshape axis for case with one output gives a 1D array; force to 2D
    if len(ax.shape) == 1:
        ax = ax.reshape(-1, 1)

    for response_column_data, response_column_name, plot_column in zip(response_data.T,
                                                                       response_names, ax.T):

        scores = score_function(feature_interaction_data, response_column_data, powers=powers, **kwargs)

        node_weights = scores[powers_degree == 1]

        node_names = np.array(feature_names)
        node_labels = {name: "{}\n{:.2e}".format(name,
                                                 weight) for name, weight in zip(node_names,
                                                                                 node_weights)}

        for degree_to_plot, plot_row in zip(range(2, degree + 1), plot_column):
            edge_weights = scores[powers_degree == degree_to_plot]

            size_factor = max_size / max(node_weights.max(), edge_weights.max())

            node_sizes = size_factor * node_weights
            edge_sizes = size_factor * edge_weights

            edge_list = [tuple(node_names[p])
                         for p in powers[powers_degree == degree_to_plot].astype('bool')]
            edge_names = feature_interaction_names[powers_degree == degree_to_plot]
            edge_labels = {names: "{}\n{:.2e}".format(int_name,
                                                      weight) for names, int_name, weight
                           in zip(edge_list, edge_names, edge_weights)}

            if degree_to_plot == 2:
                # normal graph
                graph = nx.complete_graph(node_names)

                pos = nx.circular_layout(graph, scale=1.0)

                nx.draw_networkx_nodes(graph, pos,
                                       node_size=node_sizes ** 2,
                                       node_color='xkcd:red',
                                       linewidths=1.0,
                                       alpha=alpha,
                                       ax=plot_row)

                nx.draw_networkx_labels(graph, pos,
                                        labels=node_labels,
                                        font_size=label_size,
                                        ax=plot_row)

                nx.draw_networkx_edges(graph, pos,
                                       width=edge_sizes,
                                       edge_color='xkcd:green',
                                       alpha=alpha,
                                       ax=plot_row)

                nx.draw_networkx_edge_labels(graph, pos,
                                             edge_labels=edge_labels,
                                             font_size=label_size,
                                             label_pos=.6,
                                             ax=plot_row)

            else:
                # hyper graph
                graph = nx.Graph()
                graph.add_nodes_from(node_names)
                graph.add_nodes_from(edge_list)
                for edge, weight in zip(edge_list, edge_weights):
                    for node in edge:
                        graph.add_edge(node, edge, weight=weight)

                hyper_edge_weights = [size_factor * el['weight'] for el in graph.edges.values()]

                pos = nx.bipartite_layout(graph, edge_list, align='horizontal')

                nx.draw_networkx_nodes(graph, pos,
                                       nodelist=node_names,
                                       node_size=node_sizes ** 2,
                                       node_color='xkcd:red',
                                       linewidths=1.0,
                                       alpha=alpha,
                                       ax=plot_row)

                nx.draw_networkx_nodes(graph, pos,
                                       nodelist=edge_list,
                                       node_size=edge_sizes ** 2,
                                       node_color='xkcd:green',
                                       linewidths=1.0,
                                       alpha=alpha,
                                       ax=plot_row)

                nx.draw_networkx_labels(graph, pos,
                                        labels=node_labels,
                                        font_size=label_size,
                                        ax=plot_row)

                nx.draw_networkx_labels(graph, pos,
                                        labels=edge_labels,
                                        font_size=label_size,
                                        ax=plot_row)

                nx.draw_networkx_edges(graph, pos,
                                       edge_color='xkcd:green',
                                       width=hyper_edge_weights,
                                       alpha=alpha,
                                       ax=plot_row)

            plot_row.set_title("'{}' degree {} ({})".format(response_column_name,
                                                            degree_to_plot,
                                                            method_label))


def _rank_plot(ax, feature_data, response_data, feature_names, response_names, score_function,
               degree=1, interaction_only=True, **kwargs):
    """
    Create a rank plot based on a given set of data and score function.

    Ranks each feature from 1 to N based on the result of score_function.
    A grid of boxes are plotted with response names on the x-axis
    and feature names (and possibly interactions) on the y-axis.
    Each box is colored and labeled according to its calculated rank.
    Ranks are only calculated within responses, not across responses.

    Args:
        - ax (matplotlib.Axes): Axes to plot to
        - feature_data ([[float]]): Array-like of feature data. Each column
          is a feature; each row is an observation.
        - response_data ([[float]]): Array-like of response data. Rows correspond
          to rows in feature data.
        - feature_names ([string]): Array-like of feature names
        - response_names ([string]): Array-like of response names
        - score_function (function): Function which scores features based on
          importance to responses.
        - degree (int): Maximum degree of interaction
        - interaction_only (bool): Whether to only include lowest powers of
          interaction or include higher powers.
        - kwargs: Keyword arguments passed to score_function

    Raises:
        -
    """
    feature_interaction_data, feature_interaction_names, powers = \
        _make_interactions(feature_data, feature_names, degree=degree, interaction_only=interaction_only)

    num_features = feature_interaction_data.shape[1]
    num_responses = response_data.shape[1]

    if "pce_score" in str(score_function):
        scores = np.vstack([score_function(feature_interaction_data,
                                              response_column, powers=powers,
                                              **kwargs) for response_column in response_data.T])

    else:
        scores = np.vstack([score_function(feature_interaction_data,
                                              response_column,
                                              **kwargs) for response_column in response_data.T])

    order = np.argsort(-scores)
    rank_table = np.argsort(order)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    color_mesh = ax.pcolormesh(num_features - rank_table.T, edgecolors='None', lw=0.0, cmap=cm.plasma)

    ax.hlines(np.arange(1, num_features + 1), 0, num_responses, color='k')
    ax.vlines(np.arange(1, num_responses + 1), 0, num_features, color='k')

    cbar = plt.colorbar(color_mesh, cax=cax, orientation='vertical')

    for i in range(num_responses * num_features):
        idx = np.unravel_index(i, (num_responses, num_features))
        ax.text(idx[0] + .5, idx[1] + .5, rank_table[idx] + 1, ha='center', va='center')

    ax.set_xticks(np.arange(num_responses) + 0.5)
    ax.set_xticklabels(response_names, rotation=35, ha='right')
    ax.set_xlabel('Outputs', ha='center')

    ax.set_yticks(np.arange(num_features) + 0.5)
    ax.set_yticklabels(feature_interaction_names)
    ax.set_ylabel('Parameters')

    cbar_labels = np.arange(1, num_features + 1).astype('str')
    cbar_labels[-1] += ' (Least Sensitive)'
    cbar_labels[0] += ' (Most Sensitive)'

    cbar.set_ticks(num_features - np.arange(num_features))
    cbar.set_ticklabels(cbar_labels)

    ax.set_xlim([0, num_responses])
    ax.set_ylim([0, num_features])
    ax.invert_yaxis()


def _score_plot(ax, feature_data, response_data, feature_names, response_names, score_function, title,
                degree=1, interaction_only=True, y_axis_label='Score', **kwargs):
    """
    Create a set of bar plots based on a given set of data and score function for each output.

    Uses score_function to assign a score to each feature or interaction of features.
    These scores are then plotted as a bar plot, with the height of each bar corresponding to
    the calculated score.

    Args:
        - ax ([matplotlib.Axes]): Array-like of Axes to plot to
        - feature_data ([[float]]): Array-like of feature data. Each column is a feature;
          each row is an observation.
        - response_data ([[float]]): Array-like of response data. Rows correspond to rows
          in feature data.
        - feature_names ([string]): Array-like of feature names
        - response_names ([string]): Array-like of response names
        - score_function (function): Function which scores features based on importance to
          responses.
        - title (string): Title of plot
        - degree (int): Maximum degree of interaction
        - interaction_only (bool): Whether to only include lowest powers of interaction or
          include higher powers.
        - y_axis_label (string): Y axis label
        - kwargs: Keyword arguments passed to score_function

    Raises:
        -
    """
    feature_interaction_data, feature_interaction_names, powers = \
        _make_interactions(feature_data, feature_names, degree=degree, interaction_only=interaction_only)

    for y, y_name, axes in zip(response_data.T, response_names, ax):
        scores = score_function(feature_interaction_data, y, powers=powers, **kwargs)

        axes.bar(feature_interaction_names, scores)
        axes.tick_params(axis='x', labelrotation=70)
        axes.set_xlabel('Parameter Interaction')
        axes.set_ylabel(y_axis_label)
        axes.set_title('{} ({})'.format(title, y_name))


def _make_interactions(feature_data, feature_names, degree=2, interaction_only=False):
    """
    Create interactions between features

    Creates the polynomial interactions by creating all possible combinations between
    each individual feature.
    An interaction data column is the product of 2 or more features.
    An interaction name is a concatenation of the names of the columns used to create
    the interaction.
    If interaction_only is False, then feature columns are allowed to interact with
    themselves.
    In other word, if interaction_only is False,
    then feature columns may be raised to a power in their interactions.

    Args:
        - feature_data ([[float]]): Array-like of feature data
        - feature_names ([string]): Array-like of feature names
        - degree (int): Maximum degree of interaction
        - interaction_only (bool): Whether to only include lowest powers of interaction
          or include higher powers.

    Returns:
         - Feature interaction data ([[float]]): The resulting feature data from the interactions
         - Feature interaction names ([string]): The resulting feature names from the interactions
         - Feature interaction powers ([[int]]): An array containing the powers to which each feature
                                                 was raised.
                                                 to create the interactions.

    Raises:
        -
    """
    feature_interaction_names = []

    if interaction_only:
        transformer = PolynomialFeatures(degree=degree,
                                         interaction_only=True,
                                         include_bias=False)

        for deg in range(1, degree + 1):
            feature_interaction_names.extend([':'.join(ls) for ls in itertools.combinations(feature_names,
                                                                                            r=deg)])
    else:
        transformer = PolynomialFeatures(degree=degree,
                                         interaction_only=False,
                                         include_bias=False)
        for deg in range(1, degree + 1):
            feature_interaction_names.extend(
                [':'.join(ls) for ls in itertools.combinations_with_replacement(feature_names, r=deg)])

    transformer.fit(feature_data)
    return transformer.transform(feature_data), feature_interaction_names, transformer.powers_


def _f_score(feature_data, response_data, center=True, **kwargs):
    """
    Scores features based on an F-test of linear regression coefficients

    Each features is scored using an F-test of linear regression coefficients.
    The F statistic is calculated by looking at the difference between a linear model with
    and without the feature included.
    See sklearn.feature_selction.f_regression for more information.

    Args:
        - feature_data ([[float]]): Array-like of feature data. Each column is a feature;
          each row is an observation.
        - response_data ([[float]]): Array-like of response data. Rows correspond to rows
          in feature data.
        - center (bool): Whether to center data feature_data before calculating scores
        - kwargs: Captures superfluous keyword arguments

    Returns:
        - Scores ([float]): The scores of each feature

    Raises:
        -
    """
    scores, p_values = f_regression(feature_data, response_data, center=center)
    return scores


def _f_p_score(feature_data, response_data, center=True, **kwargs):
    """
    Give the p-value for an F-test of linear regression coefficients for each feature.

    Each features is scored using an F-test of linear regression coefficients.
    The F statistic is calculated by looking at the difference between a linear model with
    and without the feature included.
    The p-value of this F statistic is returned.
    In general, a p-value < 0.05 is considered significant, meaning that particular feature
    is worth keeping.
    See sklearn.feature_selction.f_regression for more information.

    Args:
        - feature_data ([[float]]): Array-like of feature data. Each column is a feature;
          each row is an observation.
        - response_data ([[float]]): Array-like of response data. Rows correspond to rows
          in feature data.
        - center (bool): Whether to center data feature_data before calculating scores
        - kwargs: Captures superfluous keyword arguments

    Returns:
        - p_values ([float]): The p-values of each feature

    Raises:
        -
    """
    scores, p_values = f_regression(feature_data, response_data, center=center)
    return p_values


def _mutual_info_score(feature_data, response_data, n_neighbors=3, **kwargs):
    """
    Scores each feature based on the amount of mutual information shared with the response.

    Each feature is scored using an estimate of mutual information between it and the response data.
    Mutual information measures the amount of information that can be obtained about the response
    by observing the particular feature.
    See sklearn.eature_selection.mutual_info_regression for more information.

    Args:
        - feature_data ([[float]]): Array-like of feature data. Each column is a feature;
          each row is an observation.
        - response_data ([[float]]): Array-like of response data. Rows correspond to rows
          in feature data.
        - n_neighbors (int): How many neighboring bins to consider when estimating mutual
          information.
        - kwargs: Captures superfluous keyword arguments

    Returns:
        - Scores ([float]): The scores of each feature

    Raises:
        -
    """
    return mutual_info_regression(feature_data, response_data, n_neighbors=n_neighbors)


def _pce_score(feature_data, response_data, ranges, powers, pce_degree=1, model_degrees=1, **kwargs):
    """
    Scores each feature based on the amount of variance it contributes to the response in a PCE model.

    Each feature is scored using variance decomposition with a Polynomial Chaos Expansion (PCE) model.
    The PCE model uses linear regression on an basis of orthogonal polynomials.
    Using the coefficients of this model,
    each feature's contribution the the variance of the response can be assigned.
    The score is the portion of response variance that the feature is estimated to have contributed.

    Args:
        - feature_data ([[float]]): Array-like of feature data. Each column is a feature; each row
          is an observation.
        - response_data ([[float]]): Array-like of response data. Rows correspond to rows in feature
          data.
        - ranges ([[float]]): Array-like of feature ranges.
                              Each row is a length 2 array of the lower and upper bounds.
        - pce_degree (int): Maximum degree of interaction to score
        - model_degrees (int): Maximum degree of interaction for PCE model
        - kwargs: Captures superfluous keyword arguments

    Returns:
        - Scores ([float]): The scores of each feature

    Raises:
        -
    """
    feature_data_to_fit = feature_data[:, powers.sum(axis=-1) == 1]

    model = PolynomialChaosExpansionModel(num_degrees=model_degrees, ranges=ranges)
    model.fit(feature_data_to_fit, response_data[:, np.newaxis])

    return model.make_contributions(pce_degree).reshape(-1)


def one_at_a_time_effects(feature_data, response_data):
    """
    Calculates the elementary effects for a one-at-a-time study.

    The `feature_data` points should have been generated from trata.sampler.OneAtATimeSampler
    with `do_oat=True` or similar.
    Two elementary effects are calculated per feature resulting in 2*p total effects

    Args:
        - feature_data ([[float]]): Array-like of feature data. Each column is a feature; each row
          is an observation.
        - response_data ([[float]]): Array-like of response data. Rows correspond to rows in feature
          data.

    Returns:
        - Elementary Effects ([[float]]): (p by 2) Numpy Array of elementary effects
    """
    m, n = feature_data.shape
    assert m == 2*n+1

    diff_features = feature_data[0] - feature_data
    diff_response = response_data[0] - response_data

    which_var = diff_features != 0

    val = diff_response[1:]/diff_features[which_var]

    return val.reshape(-1, 2).T


def morris_effects(feature_data, response_data, tol=1e-12):
    """
    Calculates the elementary effects and Morris statistics for a Morris one-at-a-time study.

    The `feature_data` points should have been generated from trata.sampler.MorrisOneAtATimeSampler
    or similar.
    The number of elementary effects calculated depends on the number of paths in the study.
    Each path produces one elementary effect per dimension.
    So for r paths, p*r total effects will be calculated.

    Args:
        - feature_data ([[float]]): Array-like of feature data. Each column is a feature; each row
          is an observation.
        - response_data ([[float]]): Array-like of response data. Rows correspond to rows in feature
          data.
        - tol (float): Tolerance for numerical stability in parameter change detection.

    Returns:
        - dict: Contains 'elementary_effects' (r by k array), 'mu', 'mu_star', and 'sigma' arrays
    """
    # Input validation and conversion
    feature_data = np.asarray(feature_data)
    response_data = np.asarray(response_data)
    
    # Handle 1D response data
    if response_data.ndim == 1:
        response_data_1d = response_data
    else:
        response_data_1d = response_data.squeeze()

    if feature_data.shape[0] != len(response_data_1d):
        raise ValueError(f"Feature and response data must have same number of rows. "
                        f"Feature: {feature_data.shape[0]}, Response: {len(response_data_1d)}")

    n, k = feature_data.shape
    if n % (k + 1) != 0:
        raise ValueError(f"Number of samples ({n}) must be divisible by (k+1) = {k+1}")

    r = int(n / (k + 1))

    def make_effect(feature_partition, response_partition):
        diff_response = response_partition[1:] - response_partition[:-1]
        diff_features = feature_partition[1:] - feature_partition[:-1]
        which_var = np.where(np.abs(diff_features) > tol)

        # Check that exactly one parameter changes per step
        changes_per_step = np.sum(np.abs(diff_features) > tol, axis=1)
        if not np.all(changes_per_step == 1):
            raise ValueError("Invalid Morris path: multiple or no parameters changed in a step")

        val = diff_response / diff_features[which_var]
        # Re-order val
        val[which_var[1]] = val[range(k)]
        return val

    elementary_effects = np.array([make_effect(_x, _y) for _x, _y in zip(feature_data.reshape(r, k + 1, k),
                                                                    response_data_1d.reshape(r, k + 1))])

    # Calculate Morris statistics
    mu = np.mean(elementary_effects, axis=0)
    mu_star = np.mean(np.abs(elementary_effects), axis=0)
    sigma = np.std(elementary_effects, axis=0, ddof=1)

    return {
        'mu': mu,
        'mu_star': mu_star,
        'sigma': sigma,
        'elementary_effects': elementary_effects
    }


def sobol_indices(feature_data, response_data, include_second_order=False, **kwargs):
    """
    Calculates Sobol indices

    The 'feature_data' points should have been generated from
    trata.sampler.SobolIndexSampler or similar.

    Args:
        - feature_data ([[float]]): Array-like of feature data
          Each column is a feature; each row is an observation
        - response_data ([[float]]): Array-like of response data
          Rows correspond to rows in feature data. There should
          only be one column of outputs for each row of inputs.

    Returns:
        - First-order indices ([[float]]): (1 by k) Numpy array of first-order indices
        - Total-order indices ([[float]]): (1 by k) Numpy array of total-order indices
        - Second-order indices ([[float]]): (k by k) Numpy array of second-order indices
          (upper triangular)

    Raises:
        - ValueError, TypeError
    """
    r, k = feature_data.shape
    response_row, response_col = response_data.shape

    # Check data for correct shape and type
    if response_row != r:
        msg = "Feature data and response data should have the same number of rows. \n"
        msg += f"Feature data has {r} and response data has {response_row}."
        raise ValueError(msg)
    if response_col != 1:
        msg = f"Response data should have 1 column. Was given {response_col}."
        raise ValueError(msg)
    if not isinstance(feature_data[0, 0], float):
        msg = f"feature_data is {type(feature_data[0, 0])} type. "
        msg += "It should be float type"
        raise TypeError(msg)
    if not isinstance(response_data[0, 0], float):
        msg = f"response_data is {type(response_data[0, 0])} type. "
        msg += "It should be float type"
        raise TypeError(msg)

    if include_second_order and r % (2 * k + 2) == 0:
        n = int(r / (2 * k + 2))
    elif not include_second_order and r % (k + 2) == 0:
        n = int(r / (k + 2))
    else:
        raise ValueError("feature_data is not in the right format")

    y_A = response_data[:n]
    y_B = response_data[n: 2 * n]

    y_AB = np.zeros((n, k))
    S_i = np.zeros(k)
    S_Ti = np.zeros(k)
    for i in range(k):
        y_AB[:, i] = np.squeeze(response_data[(i + 2) * n: (i + 3) * n])
        S_i[i] = np.nanmean(y_B * (y_AB[:, i] - y_A)) / np.nanvar(y_A)
        S_Ti[i] = 0.5 * np.nanmean((y_A - y_AB[:, i]) ** 2) / np.nanvar(y_A)

    if include_second_order:
        y_BA = np.zeros((n, k))
        S_ij = np.zeros((k, k))
        for i in range(k):
            y_BA[:, i] = response_data[(i + k + 2) * n: (i + k + 3) * n]
            for j in range(i + 1, k):
                S_ij[i, j] = np.nanmean(
                    y_BA[:, i] * y_AB[:, j] - y_A * y_B, axis=0
                ) / np.nanvar(y_A)
                S_ij[i, j] = S_ij[i, j] - S_i[i] - S_i[j]
        return S_i, S_Ti, S_ij
    else:
        return S_i, S_Ti


def oat_score_function(feature_data, response_data, method='morris', statistic='mu_star', **kwargs):
    """
    Score function that works with both Morris and One-at-a-Time methods.

    Args:
        feature_data: Array of input samples
        response_data: Array of model outputs
        method: 'morris' or 'oat' (one-at-a-time)
        statistic: For Morris: 'mu_star', 'sigma', 'mu'. For OAT: 'mean', 'std', 'max', 'min'
        **kwargs: Additional arguments passed to the underlying functions

    Returns:
        Array of scores for each parameter
    """
    # Filter out 'powers' argument since Morris/OAT don't use it
    filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'powers'}

    if method.lower() == 'morris':
        results = morris_effects(feature_data, response_data, **filtered_kwargs)
        if statistic == 'mu_star':
            return results['mu_star']
        elif statistic == 'sigma':
            return results['sigma']
        elif statistic == 'mu':
            return results['mu']
        else:
            raise ValueError(f"Unknown Morris statistic: {statistic}")

    elif method.lower() in ['oat', 'one_at_a_time']:
        effects = one_at_a_time_effects(feature_data, response_data)  # Shape: (p, 2)
        if statistic == 'mean':
            return np.mean(effects, axis=1)
        elif statistic == 'std':
            return np.std(effects, axis=1, ddof=1)
        elif statistic == 'max':
            return np.max(np.abs(effects), axis=1)  # Max absolute effect
        elif statistic == 'min':
            return np.min(np.abs(effects), axis=1)  # Min absolute effect
        elif statistic == 'range':
            return np.ptp(effects, axis=1)  # Peak-to-peak (range)
        else:
            raise ValueError(f"Unknown OAT statistic: {statistic}")
    else:
        raise ValueError(f"Unknown method: {method}")


def morris_score_plot(ax, feature_data, response_data, feature_names, response_names, 
                      show_both=True, degree=1, interaction_only=True):
    """
    Plots Morris score plot showing mu_star and optionally sigma

    Args:
        - ax ([matplotlib.Axes]): Array-like of Axes to plot to
        - feature_data ([[float]]): Array-like of feature data
        - response_data ([[float]]): Array-like of response data
        - feature_names ([string]): Array-like of feature names
        - response_names ([string]): Array-like of response names
        - show_both (bool): If True, shows both mu_star and sigma; if False, only mu_star
        - degree (int): Maximum degree of interaction (Morris only works with degree=1)
        - interaction_only (bool): Whether to only include lowest powers of interaction
    """
    # Morris only works with original parameters (degree=1)
    if degree > 1:
        raise ValueError("Morris method only works with degree=1 (no polynomial interactions)")

    # Convert to numpy arrays and handle 1D response data
    feature_data = np.asarray(feature_data)
    response_data = np.asarray(response_data)
    if response_data.ndim == 1:
        response_data = response_data.reshape(-1, 1)

    # Handle single axes case
    if not hasattr(ax, '__iter__'):
        ax = [ax]

    # Calculate Morris statistics
    results = morris_effects(feature_data, response_data.squeeze())
    mu_star = results['mu_star']
    sigma = results['sigma']

    # Create the plot
    for y_col, y_name, axes in zip(response_data.T, response_names, ax):
        if show_both:
            x = np.arange(len(feature_names))
            width = 0.35

            bars1 = axes.bar(x - width/2, mu_star, width, label='μ* (Overall Effect)', alpha=0.8)
            bars2 = axes.bar(x + width/2, sigma, width, label='σ (Interactions)', alpha=0.8)

            axes.set_xlabel('Parameter')
            axes.set_ylabel('Morris Score')
            axes.set_title(f'Morris Sensitivity Analysis ({y_name})')
            axes.set_xticks(x)
            axes.set_xticklabels(feature_names, rotation=70)
            axes.legend()

            # Add value labels on bars
            for bar in bars1:
                height = bar.get_height()
                axes.text(bar.get_x() + bar.get_width()/2., height,
                         f'{height:.3f}', ha='center', va='bottom', fontsize=9)
            for bar in bars2:
                height = bar.get_height()
                axes.text(bar.get_x() + bar.get_width()/2., height,
                         f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        else:
            bars = axes.bar(feature_names, mu_star)
            axes.tick_params(axis='x', labelrotation=70)
            axes.set_xlabel('Parameter')
            axes.set_ylabel('μ* Score')
            axes.set_title(f'Morris μ* (Overall Sensitivity) ({y_name})')

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                axes.text(bar.get_x() + bar.get_width()/2., height,
                         f'{height:.3f}', ha='center', va='bottom', fontsize=9)


def morris_rank_plot(ax, feature_data, response_data, feature_names, response_names,
                     rank_by='mu_star', show_both=False, degree=1, interaction_only=True):
    """
    Plots Morris rank plot

    Args:
        - ax (matplotlib.Axes): Axes to plot to
        - feature_data ([[float]]): Array-like of feature data
        - response_data ([[float]]): Array-like of response data
        - feature_names ([string]): Array-like of feature names
        - response_names ([string]): Array-like of response names
        - rank_by (string): 'mu_star' or 'sigma' - which statistic to use for ranking
        - show_both (bool): If True, creates separate ranking for both mu_star and sigma
        - degree (int): Maximum degree of interaction (Morris only works with degree=1)
        - interaction_only (bool): Whether to only include lowest powers of interaction
    """
    # Morris only works with original parameters (degree=1)
    if degree > 1:
        raise ValueError("Morris method only works with degree=1 (no polynomial interactions)")

    # Convert to numpy arrays and handle 1D response data
    feature_data = np.asarray(feature_data)
    response_data = np.asarray(response_data)
    if response_data.ndim == 1:
        response_data = response_data.reshape(-1, 1)

    # Calculate Morris statistics for all responses
    all_mu_star = []
    all_sigma = []
    for y_col in response_data.T:
        results = morris_effects(feature_data, y_col)
        all_mu_star.append(results['mu_star'])
        all_sigma.append(results['sigma'])

    mu_star_scores = np.array(all_mu_star)
    sigma_scores = np.array(all_sigma)

    if show_both:
        # Show both statistics - extend feature names and scores
        extended_feature_names = [f"{name} (μ*)" for name in feature_names] + [f"{name} (σ)" for name in feature_names]
        scores = np.concatenate([mu_star_scores, sigma_scores], axis=1)
        num_features = len(extended_feature_names)
    else:
        # Show only the requested statistic
        scores = mu_star_scores if rank_by == 'mu_star' else sigma_scores
        extended_feature_names = feature_names
        num_features = len(feature_names)

    num_responses = len(response_names)

    # Create rank plot manually
    order = np.argsort(-scores)
    rank_table = np.argsort(order)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)

    color_mesh = ax.pcolormesh(num_features - rank_table.T, edgecolors='None', lw=0.0, cmap=cm.plasma,
                               vmin=0, vmax=num_features)

    ax.hlines(np.arange(1, num_features + 1), 0, num_responses, color='k')
    ax.vlines(np.arange(1, num_responses + 1), 0, num_features, color='k')

    cbar = plt.colorbar(color_mesh, cax=cax, orientation='vertical')

    for response_idx in range(num_responses):
        for feature_idx in range(num_features):
            rank_value = rank_table[response_idx, feature_idx] + 1
            ax.text(response_idx + 0.5, feature_idx + 0.5, rank_value, 
                   ha='center', va='center')

    ax.set_xticks(np.arange(num_responses) + 0.5)
    ax.set_xticklabels(response_names, rotation=35, ha='right')
    ax.set_xlabel('Outputs', ha='center')

    ax.set_yticks(np.arange(num_features) + 0.5)
    ax.set_yticklabels(extended_feature_names)
    ax.set_ylabel('Parameters')

    cbar_labels = np.arange(1, num_features + 1).astype('str')
    cbar_labels[-1] += ' (Least Sensitive)'
    cbar_labels[0] += ' (Most Sensitive)'

    cbar.set_ticks(num_features - np.arange(num_features))
    cbar.set_ticklabels(cbar_labels)

    ax.set_xlim([0, num_responses])
    ax.set_ylim([0, num_features])
    ax.invert_yaxis()


def oat_score_plot(ax, feature_data, response_data, feature_names, response_names,
                   statistic='max'):
    """
    Plots One-at-a-Time score plot
    
    Args:
        - ax ([matplotlib.Axes]): Array-like of Axes to plot to
        - feature_data ([[float]]): Array-like of feature data
        - response_data ([[float]]): Array-like of response data
        - feature_names ([string]): Array-like of feature names
        - response_names ([string]): Array-like of response names
        - statistic (string): 'mean', 'std', 'max', 'min', or 'range'
    """
    # Convert to numpy arrays and handle 1D response data
    feature_data = np.asarray(feature_data)
    response_data = np.asarray(response_data)
    if response_data.ndim == 1:
        response_data = response_data.reshape(-1, 1)

    # Handle single axes case
    if not hasattr(ax, '__iter__'):
        ax = [ax]

    title = f'OAT {statistic.title()} Effects'
    y_label = f'{statistic} Effect'

    # Create the plot directly
    for y_col, y_name, axes in zip(response_data.T, response_names, ax):
        scores = oat_score_function(feature_data, y_col, method='oat', statistic=statistic)
        
        axes.bar(feature_names, scores)
        axes.tick_params(axis='x', labelrotation=70)
        axes.set_xlabel('Parameter')
        axes.set_ylabel(y_label)
        axes.set_title('{} ({})'.format(title, y_name))


def oat_rank_plot(ax, feature_data, response_data, feature_names, response_names,
                  statistic='max'):
    """
    Plots One-at-a-Time rank plot

    Args:
        - ax (matplotlib.Axes): Axes to plot to
        - feature_data ([[float]]): Array-like of feature data
        - response_data ([[float]]): Array-like of response data
        - feature_names ([string]): Array-like of feature names
        - response_names ([string]): Array-like of response names
        - statistic (string): 'mean', 'std', 'max', 'min', or 'range'
    """
    # Convert to numpy arrays and handle 1D response data
    feature_data = np.asarray(feature_data)
    response_data = np.asarray(response_data)
    if response_data.ndim == 1:
        response_data = response_data.reshape(-1, 1)

    # Calculate OAT statistics for all responses using the existing function
    all_scores = []
    for y_col in response_data.T:
        scores = oat_score_function(feature_data, y_col, method='oat', statistic=statistic)
        all_scores.append(scores)

    scores = np.array(all_scores)
    num_features = len(feature_names)
    num_responses = len(response_names)

    # Create rank plot manually
    order = np.argsort(-scores)
    rank_table = np.argsort(order)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    color_mesh = ax.pcolormesh(num_features - rank_table.T, edgecolors='None', lw=0.0, cmap=cm.plasma)

    ax.hlines(np.arange(1, num_features + 1), 0, num_responses, color='k')
    ax.vlines(np.arange(1, num_responses + 1), 0, num_features, color='k')

    cbar = plt.colorbar(color_mesh, cax=cax, orientation='vertical')

    for i in range(num_responses * num_features):
        idx = np.unravel_index(i, (num_responses, num_features))
        ax.text(idx[0] + .5, idx[1] + .5, rank_table[idx] + 1, ha='center', va='center')

    ax.set_xticks(np.arange(num_responses) + 0.5)
    ax.set_xticklabels(response_names, rotation=35, ha='right')
    ax.set_xlabel('Outputs', ha='center')

    ax.set_yticks(np.arange(num_features) + 0.5)
    ax.set_yticklabels(feature_names)
    ax.set_ylabel('Parameters')

    cbar_labels = np.arange(1, num_features + 1).astype('str')
    cbar_labels[-1] += ' (Least Sensitive)'
    cbar_labels[0] += ' (Most Sensitive)'

    cbar.set_ticks(num_features - np.arange(num_features))
    cbar.set_ticklabels(cbar_labels)

    ax.set_xlim([0, num_responses])
    ax.set_ylim([0, num_features])
    ax.invert_yaxis()


def sobol_score_plot(ax, feature_data, response_data, feature_names, response_names,
                     index_type='first_order', include_second_order=False):
    """
    Plots Sobol indices score plot

    Args:
        - ax ([matplotlib.Axes]): Array-like of Axes to plot to
        - feature_data ([[float]]): Array-like of feature data
        - response_data ([[float]]): Array-like of response data
        - feature_names ([string]): Array-like of feature names
        - response_names ([string]): Array-like of response names
        - index_type (string): 'first_order' or 'total_order'
        - include_second_order (bool): Whether to calculate second-order indices
    """
    # Convert to numpy arrays and handle 1D response data
    feature_data = np.asarray(feature_data)
    response_data = np.asarray(response_data)
    if response_data.ndim == 1:
        response_data = response_data.reshape(-1, 1)

    # Calculate Sobol indices
    if include_second_order:
        S_i, S_Ti, S_ij = sobol_indices(feature_data, response_data, include_second_order=True)
    else:
        S_i, S_Ti = sobol_indices(feature_data, response_data, include_second_order=False)

    if index_type == 'first_order':
        scores = S_i
    elif index_type == 'total_order':
        scores = S_Ti
    else:
        raise ValueError(f"Unknown index_type: {index_type}")

    # Handle single axes case
    if not hasattr(ax, '__iter__'):
        ax = [ax]

    title = f'Sobol {index_type.replace("_", " ").title()} Indices'
    y_label = f'{index_type.replace("_", " ").title()} Index'

    # Create the plot directly
    for y_col, y_name, axes in zip(response_data.T, response_names, ax):
        axes.bar(feature_names, scores)
        axes.tick_params(axis='x', labelrotation=70)
        axes.set_xlabel('Parameter')
        axes.set_ylabel(y_label)
        axes.set_title('{} ({})'.format(title, y_name))


def sobol_rank_plot(ax, feature_data, response_data, feature_names, response_names,
                    index_type='first_order', include_second_order=False):
    """
    Plots Sobol indices rank plot

    Args:
        - ax (matplotlib.Axes): Axes to plot to
        - feature_data ([[float]]): Array-like of feature data
        - response_data ([[float]]): Array-like of response data
        - feature_names ([string]): Array-like of feature names
        - response_names ([string]): Array-like of response names
        - index_type (string): 'first_order' or 'total_order'
        - include_second_order (bool): Whether to calculate second-order indices
    """
    # Convert to numpy arrays and handle 1D response data
    feature_data = np.asarray(feature_data)
    response_data = np.asarray(response_data)
    if response_data.ndim == 1:
        response_data = response_data.reshape(-1, 1)

    # Calculate Sobol indices for all responses
    all_scores = []
    for y_col in response_data.T:
        y_col_2d = y_col.reshape(-1, 1)  # Sobol expects 2D

        if include_second_order:
            S_i, S_Ti, S_ij = sobol_indices(feature_data, y_col_2d, include_second_order=True)
        else:
            S_i, S_Ti = sobol_indices(feature_data, y_col_2d, include_second_order=False)

        if index_type == 'first_order':
            scores = S_i
        elif index_type == 'total_order':
            scores = S_Ti
        else:
            raise ValueError(f"Unknown index_type: {index_type}")

        all_scores.append(scores)

    scores = np.array(all_scores)
    num_features = len(feature_names)
    num_responses = len(response_names)
    
    # Create rank plot manually
    order = np.argsort(-scores)
    rank_table = np.argsort(order)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    color_mesh = ax.pcolormesh(num_features - rank_table.T, edgecolors='None', lw=0.0, cmap=cm.plasma)

    ax.hlines(np.arange(1, num_features + 1), 0, num_responses, color='k')
    ax.vlines(np.arange(1, num_responses + 1), 0, num_features, color='k')

    cbar = plt.colorbar(color_mesh, cax=cax, orientation='vertical')

    for i in range(num_responses * num_features):
        idx = np.unravel_index(i, (num_responses, num_features))
        ax.text(idx[0] + .5, idx[1] + .5, rank_table[idx] + 1, ha='center', va='center')

    ax.set_xticks(np.arange(num_responses) + 0.5)
    ax.set_xticklabels(response_names, rotation=35, ha='right')
    ax.set_xlabel('Outputs', ha='center')

    ax.set_yticks(np.arange(num_features) + 0.5)
    ax.set_yticklabels(feature_names)
    ax.set_ylabel('Parameters')

    cbar_labels = np.arange(1, num_features + 1).astype('str')
    cbar_labels[-1] += ' (Least Sensitive)'
    cbar_labels[0] += ' (Most Sensitive)'

    cbar.set_ticks(num_features - np.arange(num_features))
    cbar.set_ticklabels(cbar_labels)

    ax.set_xlim([0, num_responses])
    ax.set_ylim([0, num_features])
    ax.invert_yaxis()


def lasso_path_plot(ax, feature_data, response_data, feature_names, response_names,
                    degree=1, method='lasso'):
    """
        Plots Lasso paths

        Plots the path of linear regression coefficients for different amounts of l1 regularization.
        The x-axis is the shrinkage ratio which varies between 0 and 1.
        The shrinkage ratio is a ratio between the L1 norm of the regularized coefficients and the
        L1 norm of the un-regularized coefficients.
        Features that go to zero sooner tend to contribute less to the sensitivity of the response.

        Args:
            - ax ([matplotlib.Axes]): Array-like of Axes to plot to
            - feature_data ([[float]]): Array-like of feature data. Each column is a feature; each
              row is an observation.
            - response_data ([[float]]): Array-like of response data. Rows correspond to rows in
              feature data.
            - feature_names ([string]): Array-like of feature names
            - response_names ([string]): Array-like of response names
            - degree (int): Maximum degree of interaction
            - method (string): Which algorithm to use; lasso: Coordinate descent, lars: least angle
              regression.

        Raises:
            -
    """
    feature_interaction_data, feature_interaction_names, _ = \
        _make_interactions(feature_data, feature_names, degree=degree, interaction_only=False)

    for response_column_data, response_column_name, axes in zip(response_data.T, response_names, ax):

        alphas, active, coefficients = lars_path(feature_interaction_data,
                                                 response_column_data,
                                                 method=method)

        shrinkage = np.sum(np.abs(coefficients.T), axis=1)
        shrinkage /= shrinkage[-1]

        lines = [axes.plot(shrinkage, coefficient, color=np.random.rand(3))[0]
                 for coefficient in coefficients]
        axes.legend(lines, feature_interaction_names)

        for s in shrinkage:
            axes.axvline(s, 0, 1, linestyle='dashed', color='grey')
        axes.hlines(0, 0, 1, linestyles='dashed', color='grey')

        axes.set_xlabel('Shrinkage Factor')
        axes.set_ylabel('Coefficient')
        axes.set_title('{} Path'.format(method.upper()))


def sensitivity_plot(ax, surrogate_model, feature_names, response_names, feature_ranges,
                     num_plot_points=100, num_seed_points=5, seed=2018):
    """
    Plots sensitivity plots

    Also called spaghetti plot.
    Plots a set of 1-d slices of the response surface.
    Colors across plots correspond to the same seed point.

    Args:
        - ax ([[matplotlib.Axes]]): Array-like of Axes to plot to
        - surrogate_model (surrogate_model): Surrogate model which has been fit
        - feature_names ([string]): Array-like of feature names
        - response_names ([string]): Array-like of response names
        - feature_ranges ([[float]]): Array-like of feature ranges.
                                      Each row is a length 2 array of the lower and upper bounds.
        - num_plot_points (int): Number of points to plot on each dimension sweep
        - num_seed_points (int): Number of points to use as default points
        - seed (int): RNG seed

    Raises:
        -
    """
    rand_gen = np.random.default_rng(seed)

    np_seed_points = LatinHyperCubeSampler.sample_points(num_points=num_seed_points,
                                                         box=feature_ranges,
                                                         seed=seed)
    colors = rand_gen.random((num_seed_points, 3), dtype='float')

    for color, point in zip(colors, np_seed_points):
        for feature_index, feature_name in enumerate(feature_names):

            dimension_sweep = np.linspace(feature_ranges[feature_index][0],
                                          feature_ranges[feature_index][1],
                                          num_plot_points)
            new_feature_data = np.tile(point, reps=(num_plot_points, 1))
            new_feature_data[:, feature_index] = dimension_sweep
            response_prediction = surrogate_model.predict(new_feature_data)

            for response_index, response_name in enumerate(response_names):
                if len(response_names) == 1:
                    ax[feature_index].set_xlabel(feature_name)
                    ax[feature_index].set_ylabel(response_name)
                    ax[feature_index].plot(dimension_sweep, response_prediction, color=color, alpha=.75)
                else:
                    ax[feature_index, response_index].set_xlabel(feature_name)
                    ax[feature_index, response_index].set_ylabel(response_name)
                    ax[feature_index, response_index].plot(dimension_sweep,
                                                           response_prediction[:, response_index],
                                                           color=color,
                                                           alpha=.75)


def f_score_plot(ax, feature_data, response_data, feature_names, response_names,
                 degree=1, interaction_only=True, use_p_value=False):
    """
    Plots F score plot

    Args:
        - ax ([matplotlib.Axes]): Array-like of Axes to plot to
        - feature_data ([[float]]): Array-like of feature data. Each column is a feature;
          each row is an observation.
        - response_data ([[float]]): Array-like of response data. Rows correspond to rows
          in feature data.
        - feature_names ([string]): Array-like of feature names
        - response_names ([string]): Array-like of response names
        - degree (int): Maximum degree of interaction
        - interaction_only (bool): Whether to only include lowest powers of interaction or
          include higher powers.
        - use_p_value (bool): Whether to use p-values or raw F-score

    Raises:
        -
    """
    _score_plot(ax=ax,
                feature_data=feature_data,
                response_data=response_data,
                feature_names=feature_names,
                response_names=response_names,
                degree=degree,
                interaction_only=interaction_only,
                score_function=_f_p_score if use_p_value else _f_score,
                title='F Score',
                y_axis_label='p-value' if use_p_value else 'Score')

    if use_p_value:
        for axes in ax:
            axes.axhline(.05, 0, 1, linestyle='dashed', color='red', label='alpha=.05')


def mutual_info_score_plot(ax, feature_data, response_data, feature_names, response_names, n_neighbors=3):
    """
    Plots mutual information score plot

    Args:
        - ax ([matplotlib.Axes]): Array-like of Axes to plot to
        - feature_data ([[float]]): Array-like of feature data. Each column is a feature;
          each row is an observation.
        - response_data ([[float]]): Array-like of response data. Rows correspond to rows
          in feature data.
        - feature_names ([string]): Array-like of feature names
        - response_names ([string]): Array-like of response names
        - n_neighbors (int): How many neighboring bins to consider when estimating mutual
          information.

    Raises:
        -
    """
    _score_plot(ax=ax,
                feature_data=feature_data,
                response_data=response_data,
                feature_names=feature_names,
                response_names=response_names,
                degree=1,
                interaction_only=False,
                score_function=_mutual_info_score,
                title='Mutual Information',
                y_axis_label='Shared Information (nats)',
                n_neighbors=n_neighbors)


def pce_score_plot(ax, feature_data, response_data, feature_names, response_names, feature_ranges,
                   degree=1, model_degrees=1):
    """
    Plots PCE score plot

    Args:
        - ax ([matplotlib.Axes]): Array-like of Axes to plot to
        - feature_data ([[float]]): Array-like of feature data. Each column is a feature;
          each row is an observation.
        - response_data ([[float]]): Array-like of response data. Rows correspond to rows in
          feature data.
        - feature_names ([string]): Array-like of feature names
        - response_names ([string]): Array-like of response names
        - feature_ranges ([[float]]): Array-like of feature ranges. Each row is a length 2
          array of the lower and upper bounds.
        - degree (int): Maximum degree of interaction to plot
        - model_degrees (int): Maximum degree of interaction for PCE model

    Raises:
        -
    """
    _score_plot(ax=ax,
                feature_data=feature_data,
                response_data=response_data,
                feature_names=feature_names,
                response_names=response_names,
                degree=degree,
                interaction_only=True,
                score_function=_pce_score,
                title='PCE Variance Decomposition',
                y_axis_label='Variance Contribution',
                ranges=feature_ranges,
                pce_degree=degree,
                model_degrees=model_degrees)


def f_score_rank_plot(ax, feature_data, response_data, feature_names, response_names,
                      degree=1, interaction_only=True, use_p_value=False):
    """
    Plots F score rank plot

    Args:
        - ax ([matplotlib.Axes]): Array-like of Axes to plot to
        - feature_data ([[float]]): Array-like of feature data. Each column is a feature;
          each row is an observation.
        - response_data ([[float]]): Array-like of response data. Rows correspond to rows
          in feature data.
        - feature_names ([string]): Array-like of feature names
        - response_names ([string]): Array-like of response names
        - degree (int): Maximum degree of interaction
        - interaction_only (bool): Whether to only include lowest powers of interaction or
          include higher powers.
        - use_p_value (bool): Whether to use p-values or raw F-score

    Raises:
        -
    """
    _rank_plot(ax=ax,
               feature_data=feature_data,
               response_data=response_data,
               feature_names=feature_names,
               response_names=response_names,
               score_function=(lambda *args: -_f_p_score(*args)) if use_p_value else _f_score,
               degree=degree,
               interaction_only=interaction_only)


def mutual_info_rank_plot(ax, feature_data, response_data, feature_names, response_names, n_neighbors=3):
    """
    Plots mutual information rank plot

    Args:
        - ax (matplotlib.Axes): Array-like of Axes to plot to
        - feature_data ([[float]]): Array-like of feature data. Each column is a feature;
          each row is an observation.
        - response_data ([[float]]): Array-like of response data. Rows correspond to rows
          in feature data.
        - feature_names ([string]): Array-like of feature names
        - response_names ([string]): Array-like of response names
        - n_neighbors (int): How many neighboring bins to consider when estimating mutual
          information.

    Raises:
        -
    """
    _rank_plot(ax=ax,
               feature_data=feature_data,
               response_data=response_data,
               feature_names=feature_names,
               response_names=response_names,
               score_function=_mutual_info_score,
               degree=1,
               interaction_only=False,
               n_neighbors=n_neighbors)


def pce_rank_plot(ax, feature_data, response_data, feature_names, response_names, feature_ranges,
                  degree=1, model_degrees=1):
    """
    Plots PCE rank plot

    Args:
        - ax (matplotlib.Axes): Array-like of Axes to plot to
        - feature_data ([[float]]): Array-like of feature data. Each column is a feature;
          each row is an observation.
        - response_data ([[float]]): Array-like of response data. Rows correspond to rows
          in feature data.
        - feature_names ([string]): Array-like of feature names
        - response_names ([string]): Array-like of response names
        - feature_ranges ([[float]]): Array-like of feature ranges. Each row is a length
          2 array of the lower and upper bounds.
        - degree (int): Maximum degree of interaction to plot
        - model_degrees (int): Maximum degree of interaction for PCE model

    Raises:
        -
    """
    _rank_plot(ax=ax,
               feature_data=feature_data,
               response_data=response_data,
               feature_names=feature_names,
               response_names=response_names,
               score_function=_pce_score,
               degree=degree,
               interaction_only=True,
               ranges=feature_ranges,
               pce_degree=degree,
               model_degrees=model_degrees)


def f_score_network_plot(ax, feature_data, response_data, feature_names, response_names,
                         degree=2, max_size=10.0, label_size=10, alpha=.5):
    """
    Plots F score network plot

    Args:
        - ax ([matplotlib.Axes]): Array-like of Axes to plot to
        - feature_data ([[float]]): Array-like of feature data. Each column is a feature;
          each row is an observation.
        - response_data ([[float]]): Array-like of response data. Rows correspond to rows
          in feature data.
        - feature_names ([string]): Array-like of feature names
        - response_names ([string]): Array-like of response names
        - degree (int): Maximum degree of interaction
        - max_size (float): Maximum size of elements in plot. Measured in points
        - label_size (int): Font size of labels. Measured in points
        - alpha (float): Opacity of elements in plot.

    Raises:
        -
    """
    _variance_network_plot(ax=ax,
                           feature_data=feature_data,
                           response_data=response_data,
                           feature_names=feature_names,
                           response_names=response_names,
                           score_function=_f_score,
                           method_label='F score',
                           degree=degree,
                           max_size=max_size,
                           label_size=label_size,
                           alpha=alpha)


def pce_network_plot(ax, feature_data, response_data, feature_names, response_names, feature_ranges,
                     degree=2, model_degrees=2, max_size=10.0, label_size=10, alpha=.5):
    """
    Plots PCE network plot

    Args:
        - ax (matplotlib.Axes): Array-like of Axes to plot to
        - feature_data ([[float]]): Array-like of feature data. Each column is a feature;
          each row is an observation.
        - response_data ([[float]]): Array-like of response data. Rows correspond to rows in
          feature data.
        - feature_names ([string]): Array-like of feature names
        - response_names ([string]): Array-like of response names
        - feature_ranges ([[float]]): Array-like of feature ranges. Each row is a length 2
          array of the lower and upper bounds.
        - degree (int): Maximum degree of interaction to plot
        - model_degrees (int): Maximum degree of interaction for PCE model
        - max_size (float): Maximum size of elements in plot. Measured in points
        - label_size (int): Font size of labels. Measured in points
        - alpha (float): Opacity of elements in plot.

    Raises:
        -
    """
    _variance_network_plot(ax=ax,
                           feature_data=feature_data,
                           response_data=response_data,
                           feature_names=feature_names,
                           response_names=response_names,
                           score_function=_pce_score,
                           method_label='PCE',
                           degree=degree,
                           max_size=max_size,
                           label_size=label_size,
                           alpha=alpha,
                           ranges=feature_ranges,
                           pce_degree=degree,
                           model_degrees=model_degrees)


if __name__ == "__main__":
    pass
