"""Causal discovery plots"""

# Authors: Annika Högner


import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.models import LinearMediation, Prediction
from tigramite import plotting as tp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.robust_parcorr import RobustParCorr
from tigramite.independence_tests.gpdc import GPDC
from tigramite.independence_tests.cmiknn import CMIknn

# Dictionary to map mask names to abbreviations
mask_abbreviations = {
    'Southern AR dry season': 'SAR_ds',
    'Southern AR dry season (MJJA)': 'SAR_MJJA',
    'Southern AR dry season (JJA)': 'SAR_JJA',
    'Southern AR dry season (JJAS)': 'SAR_JJAS',
    'Southern AR dry season (JASO)': 'SAR_JASO',
    'Southern AR wet season': 'SAR_ws',
    'Northern AR wet season': 'NAR_ws',
    'Northern AR dry season': 'NAR_ds',
    'South eastern AR dry season': 'SEAR_ds',
    'South western AR dry season': 'SWAR_ds',
    'South western AR wet season': 'SWAR_ws',
    'South eastern AR wet season': 'SEAR_ws',
    'South eastern AR dry season (MJJA)': 'SEAR_MJJA',
    'South eastern AR dry season (JJA)': 'SEAR_JJA',
    'South eastern AR dry season (JJAS)': 'SEAR_JJAS',
    'South eastern AR dry season (JASO)': 'SEAR_JASO',
    'South western AR dry season (MJJA)': 'SWAR_MJJA',
    'South western AR dry season (JJA)': 'SWAR_JJA',
    'South western AR dry season (JJAS)': 'SWAR_JJAS',
    'South western AR dry season (JASO)': 'SWAR_JASO'
}

def load_data(file_path, variables):
    df = pd.read_csv(file_path, index_col=[0])
    data = np.array(df[variables])
    return data, df

def apply_mask(df, mask_column, num_vars):
    mask = np.array(df[[mask_column] * num_vars])
    return mask

def create_dataframe(data, mask, names):
    dataframe = pp.DataFrame(data, mask=mask, datatime={0: np.arange(len(data))}, var_names=names)
    return dataframe

def run_pcmci(dataframe, alpha_level, tau_max, CI_test, mask_type):
    pcmci = PCMCI(
        dataframe=dataframe,
        cond_ind_test=CI_test(mask_type=mask_type),
        verbosity=0
    )
    results = pcmci.run_pcmciplus(tau_min=0, tau_max=tau_max, pc_alpha=alpha_level, reset_lagged_links=True)
    q_matrix = pcmci.get_corrected_pvalues(p_matrix=results['p_matrix'], tau_max=tau_max,
                                           fdr_method='fdr_bh', exclude_contemporaneous=False)
    pcmci.print_significant_links(p_matrix=q_matrix, val_matrix=results['val_matrix'], alpha_level=alpha_level)
    graph = pcmci.get_graph_from_pmatrix(p_matrix=q_matrix, alpha_level=alpha_level,
                                         tau_min=0, tau_max=tau_max, link_assumptions=None)    
    
    return pcmci, results, q_matrix, graph

def run_mediation(dataframe, all_parents, tau_max, mask_type):
    med = LinearMediation(dataframe=dataframe, mask_type=mask_type)
    med.fit_model(all_parents=all_parents, tau_max=tau_max)
    Links = med.get_val_matrix()
    return Links, med

def plot_graph(Links, graph, names, alpha_level, tau_max, CI_test, mask_name, med, optional_text='', optional_text_suffix='', show_table=False):
    # Plot the graph
    tp.plot_graph(
        val_matrix=Links,
        graph=graph,
        var_names=names,
        show_colorbar=False,
        show_autodependency_lags=False,
        node_label_size=12,
        link_label_fontsize=15,
    	node_size=0.35,
	    arrow_linewidth=9,
        curved_radius=0.2,
        vmax_nodes=0.9,
        vmin_nodes=-0.9,
        vmax_edges=0.9,
        vmin_edges=-0.9
    )
    
    alpha_label = int(alpha_level * 100)
    plt.suptitle(f'{mask_name}{optional_text}', fontsize=14, weight='bold')
    plt.title(f'ERA5 PCMCI+ {CI_test.__name__}', fontsize=12)
    plt.xlabel(f'$\\tau_{{max}}$={tau_max}, $\\alpha_{{PC}}$={alpha_label}%', fontsize=12)
      #  output_path = "/Users/ernest/Research/thesis/code/figure_scripts/figures/"  # Adjust this path to your desired output directory

    output_path = "/Users/ernest/Research/thesis/code/figure_scripts/figures/paper/"  # Adjust this path to your desired output directory
    output_filename = f"{output_path}{mask_abbreviations.get(mask_name, 'NoMask')}-{'_'.join(names)}-{CI_test.__name__}-{tau_max}-{alpha_level}{optional_text_suffix}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    
    # Plot the table
    if show_table:
        # Create values for table
        CEs = [
            ["{:.2f}".format(np.round(med.get_ce(i=0, tau=-2, j=1), 2))],
            ["{:.2f}".format(np.round(med.get_ce(i=1, tau=-1, j=2), 2))],
            ["{:.2f}".format(np.round(med.get_ce(i=2, tau=-1, j=3), 2))],
            ["{:.2f}".format(np.round(med.get_ce(i=0, tau=-2, j=2), 2))],
            ["{:.2f}".format(np.round(med.get_ce(i=0, tau=-3, j=2), 2))],
            ["{:.2f}".format(np.round(med.get_ce(i=0, tau=-4, j=3), 2))],
            ["{:.2f}".format(np.round(med.get_ce(i=0, tau=-1, j=0), 2))],
            ["{:.2f}".format(np.round(med.get_ce(i=1, tau=-2, j=1), 2))],
            ["{:.2f}".format(np.round(med.get_ce(i=2, tau=-1, j=2), 2))],
            ["{:.2f}".format(np.round(med.get_ce(i=3, tau=-4, j=3), 2))]
        ]
        
        rowLabels = [
        	rf'{ "total" if np.abs(tau) > 2 else "direct" } CE {names[i]}$\rightarrow${names[j]}'
         if i != j else
         rf'auto-CE {names[i]}'	
           # rf'{ "total" if np.abs(tau) > 2 else "direct" } CE {names[i]}$\rightarrow${names[j] if i != j else {names[i]}}:'
        for (i, j), tau in zip([(0, 1), (1, 2), (2, 3), (0, 2), (0, 2), (0, 3), (0, 0), (1, 1), (2, 2), (3, 3)], 
        [-2, -1, -1, -2, -3, -4, -1, -2, -1, -4])
        ]

        # Create table figure
        fig, ax = plt.subplots(figsize=(2, 3.5))

        # Create table
        table = ax.table(
            cellText=CEs,
            rowLabels=rowLabels,
            cellLoc='center',
            loc='center'
        )

        ax.set_title(r'Causal effect ($\sigma$)') 

        table.set_fontsize(12)
        table.scale(1, 1.6)
        ax.axis('off')

        # Save table figure
        table_output_filename = f"{output_path}{mask_abbreviations.get(mask_name, 'NoMask')}-{'_'.join(names)}-{CI_test.__name__}-{tau_max}-{alpha_level}{optional_text_suffix}_table.png"
        plt.savefig(table_output_filename, dpi=300, bbox_inches='tight', transparent=True)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Arguments for causal discovery plots")
    parser.add_argument("--alpha", type=float, default=0.05, help="Value for parameter alpha (default: 0.05)")
    parser.add_argument("--taumax", type=int, default=5, help="Value for parameter tau_max (default: 5)")
    parser.add_argument("--vars", type=str, help="Comma-separated list of variables to select from the dataset")
    parser.add_argument("--varnames", type=str, help="Comma-separated list of variable names to use in the plots")
    parser.add_argument("--CI", type=str, default='ParCorr', help="Name of the conditional independence test function to apply to the data")
    parser.add_argument("--mask", type=str, help="Mask selection column name in the dataset", default=None)
    parser.add_argument("--show-table", action="store_true", help="Whether to include a table in the plot")
    parser.add_argument("--optional-text", type=str, default='', help="Optional text to add to the suptitle")
    args = parser.parse_args()

    # Parse the variables and names
    variables = args.vars.split(',')
    names = args.varnames.split(',')

    # Check if the number of variables and names are equal
    if len(variables) != len(names):
        raise ValueError("Number of variables and variable names must be equal.")

    # If optional_text parameter is non-empty, add it to output filename
    optional_text = args.optional_text.strip()  # Strip leading/trailing whitespaces
    optional_text_suffix = f"_{optional_text.replace(' ', '_')}" if optional_text else ""

    # Map CI test names to actual functions
    ci_tests = {
        'ParCorr': ParCorr,
        'RobustParCorr': RobustParCorr,
        'GPDC': GPDC,
        'CMIknn': CMIknn
    }

    # Validate the CI test function
    if args.CI not in ci_tests:
        raise ValueError(f"CI test {args.CI} is not recognised. Available tests: {list(ci_tests.keys())}")
    CI_test = ci_tests[args.CI]

    # Load data
    data, df = load_data('/Users/ernest/Research/thesis/_paper_/data/ERA5_1982-2022.csv', variables)

    # Apply masking if a mask is provided
    if args.mask:
        data_mask = apply_mask(df, args.mask, len(variables))
        mask_type = 'y'
    else:
        data_mask = None
        mask_type = None

    # Create dataframe
    dataframe = create_dataframe(data, data_mask, names)

    # Run PCMCI
    pcmci, results, q_matrix, graph = run_pcmci(dataframe, args.alpha, args.taumax, CI_test, mask_type)

    # Get all parents from the graph
    all_parents = pcmci.return_parents_dict(graph, val_matrix=results['val_matrix'], include_lagzero_parents=True)
    print(all_parents)

    # Run mediation analysis
    Links, med = run_mediation(dataframe, all_parents, args.taumax, mask_type)

    # Plot the graph
    plot_graph(Links, graph, names, args.alpha, args.taumax, CI_test, args.mask, med, args.optional_text, optional_text_suffix, args.show_table)

if __name__ == "__main__":
    main()