import os
import json
import codecs
import numpy as np
import scienceplots
import matplotlib.pyplot as plt
import seaborn as sns


def save_results(n_train, n_sim, n_pts, n_pts_asymp, alpha, sigma, seed, mse_emp, mse_asy, fix_phi=True, save_path="../result/ridgeless/"):
    """
    Save results to a JSON file.
    
    Parameters:
    - n_train, alpha, sigma, seed: parameters of the simulation
    - phi_range: range of phi values
    - mse_emp: array of mean squared error values from simulation
    - phi: actual phi values
    - mse_asy: array of asymptotic mean squared error values
    - save_path: directory to save the file
    """

    if fix_phi: 
        phi_range = list(np.logspace(-1, 1, n_pts))
        phi = list(np.logspace(-1, 1, n_pts_asymp))
        data = {
            "n_train": n_train,
            "alpha": alpha,
            "sigma": sigma,
            "seed": seed,
            "emp_phi": phi_range,
            "emp_risk": mse_emp.tolist(),
            "phi": phi,
            "risk": mse_asy.tolist()
        }
    else:
        psi_range = list(np.linspace(0.11, 0.49, int(n_pts/2))) + list(np.linspace(0.51, 1, int(n_pts/2)))
        psi = list(np.linspace(0.11, 0.49, int(n_pts_asymp/2))) + list(np.linspace(0.51, 1, int(n_pts_asymp/2)))
        data = {
            "n_train": n_train,
            "alpha": alpha,
            "sigma": sigma,
            "seed": seed,
            "emp_psi": psi_range,
            "emp_risk": mse_emp.tolist(),
            "psi": psi,
            "risk": mse_asy.tolist()
        }
    
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
        
    file_path = save_path + 'SNR' + str(np.round(alpha/sigma, 2)) + '_nsim' + str(n_sim) + '.json'
    with open(file_path, 'w') as file:
        json.dump(data, file,indent=4)


def save_results_cmp(n_train, n_sim, n_pts, n_pts_asymp, alpha, sigma, seed, mse_sketched_emp, mse_sketched_asy, mse_nonsketched_emp, mse_nonsketched_asy, fix_phi=True, save_path="../result/ridgeless/"):
    if fix_phi: 
        phi_range = list(np.logspace(-1, 1, n_pts))
        phi = list(np.logspace(-1, 1, n_pts_asymp))
        data = {
            "n_train": n_train,
            "alpha": alpha,
            "sigma": sigma,
            "seed": seed,
            "emp_phi": phi_range,
            "emp_sketch_risk": mse_sketched_emp.tolist(),
            "emp_ridgeless_risk": mse_nonsketched_emp.tolist(),
            "phi": phi,
            "sketch_risk": mse_sketched_asy.tolist(),
            "ridgeless_risk": mse_nonsketched_asy.tolist()
        }
    else:
        psi_range = list(np.linspace(0.11, 0.49, int(n_pts/2))) + list(np.linspace(0.51, 1, int(n_pts/2)))
        psi = list(np.linspace(0.11, 0.49, int(n_pts_asymp/2))) + list(np.linspace(0.51, 1, int(n_pts_asymp/2)))
        data = {
            "n_train": n_train,
            "alpha": alpha,
            "sigma": sigma,
            "seed": seed,
            "emp_psi": psi_range,
            "emp_risk": mse_sketched_emp.tolist(),
            "psi": psi,
            "risk": mse_sketched_asy.tolist()
        }
    
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
        
    file_path = save_path + 'alpha' + str(alpha) + 'sigma' + str(sigma) + '_nsim' + str(n_sim) + '.json'
    with open(file_path, 'w') as file:
        json.dump(data, file,indent=4)


def load_files(file_loc, fix_psi=True):
    """
    Load results from a JSON file.
    
    Parameters:
    - file_loc: location of the file
    
    Returns:
    - snr, emp_psi, emp_risk, psi, risk
    """
    file = json.loads(codecs.open('result/' + file_loc, 'r').read())
    if fix_psi:
        emp_phi = file["emp_phi"]
        emp_risk = file["emp_risk"]
        phi = file["phi"]
        risk = file["risk"]
        return file["alpha"]/file["sigma"], emp_phi, emp_risk, phi, risk
    else:
        emp_psi = file["emp_psi"]
        emp_risk = file["emp_risk"]
        psi = file["psi"]
        risk = file["risk"]
        return file["alpha"]/file["sigma"], emp_psi, emp_risk, psi, risk


def load_files_cmp(file_loc, correlated=False):
    file = json.loads(codecs.open('result/' + file_loc, 'r').read())
    emp_phi = file["emp_phi"]
    emp_sketch_risk = file["emp_sketch_risk"]

    phi = file["phi"]
    ridgeless_risk = file["ridgeless_risk"]

    if not correlated:
        emp_ridgeless_risk = file["emp_ridgeless_risk"]
        sketch_risk = file["sketch_risk"]
        return file["alpha"], file["sigma"], emp_phi, emp_sketch_risk, emp_ridgeless_risk, phi, sketch_risk, ridgeless_risk
    
    return file["alpha"], file["sigma"], emp_phi, emp_sketch_risk, phi, ridgeless_risk


def plot_results(files, fix_psi=True, save_path=None):
    """
    Plots the results.
    
    Parameters:
    - files: a list of file paths, each file contains data for one line in the plot
    - save_path: the path where to save the plot
    """
    ticks = [0.1, 0.2, 0.5, 1, 2, 5, 10]
    labels = [i for i in ticks]

    with plt.style.context(['science', 'no-latex', 'std-colors']):
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        
        if fix_psi:
            markers = ['o', 'X', 'd']
            for i, file in enumerate(files):
                snr, emp_psi, emp_risk, psi, risk = load_files(file, fix_psi)
                ax = sns.scatterplot(x=emp_psi, y=emp_risk, marker=markers[i%len(markers)], s=60, label = "SNR = " + str(np.round(snr, 2)))
                ax = sns.lineplot(x=psi, y=risk)

            ax.set(xticks = ticks, xticklabels = labels)
            ax.set_xlabel(r"$\phi$", fontsize=18)
            ax.set_ylim(-10, 300)
            ax.set_xlim(0.1, 9.99)
            ax.set_xscale('log')
            ax.axvline(x=1, color="grey", linestyle='dashed', linewidth=1)
            ax.set_ylabel(r"$R_{X}\left(\widehat{\beta} ; \beta\right)$", fontsize=18)
        else:
            for i, file in enumerate(files):
                is_orthogonal = 'orthogonal' in file
                label = 'orthogonal sketching' if is_orthogonal else 'i.i.d. sketching'
                marker = 'o' if is_orthogonal else 'X'

                snr, emp_psi, emp_risk, psi, risk = load_files(file, fix_psi)
                ax = sns.scatterplot(x=emp_psi, y=emp_risk, marker=marker, s=60, label=label)
                ax = sns.lineplot(x=psi, y=risk)

            ax.set_xlabel(r"$\psi$", fontsize=18)
            ax.set_ylim(-50, 300)
            ax.set_xlim(0, 1)
            ax.axvline(x=0.5, color="grey", linestyle='dashed', linewidth=1)
            ax.invert_xaxis()
            ax.set_ylabel(r"$R_{(S, X)}\left(\widehat{\beta}^S ; \beta\right)$", fontsize=18)

        ax.get_legend().remove()
        handles, labels = ax.get_legend_handles_labels()

        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.52, -0.1), ncol = 4, prop={'size':12})
        plt.tight_layout()

        if save_path: 
            plt.savefig(save_path, dpi=400, facecolor='w', edgecolor='w', pad_inches=None)
        plt.show()


def plot_results_cmp(files, snr=None, correlated=False, save_path=None):
    """
    Plots the results.
    
    Parameters:
    - files: a list of file paths, each file contains data for one line in the plot
    - save_path: the path where to save the plot
    """
    ticks = [0.1, 0.2, 0.5, 1, 2, 5, 10]
    labels = [i for i in ticks]

    with plt.style.context(['science', 'no-latex', 'std-colors']):
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))

        for i, file in enumerate(files):

            if 'orthogonal' in file:
                sketching_type = "orthogonal sketching"
            else:
                sketching_type = "i.i.d. sketching"
        
            if correlated:
                alpha, sigma, emp_phi, emp_sketch_risk, phi, ridgeless_risk = load_files_cmp(file, correlated)
                ax = sns.scatterplot(x=emp_phi, y=emp_sketch_risk, color='darkred', s=60, label=sketching_type)
                ax = sns.lineplot(x=phi, y=ridgeless_risk, label="no sketching")
                ax = sns.lineplot(x=emp_phi, y=emp_sketch_risk, color='darkred')
            else:
                alpha, sigma, emp_phi, emp_sketch_risk, emp_ridgeless_risk, phi, sketch_risk, ridgeless_risk = load_files_cmp(file, correlated=correlated)
                ax = sns.scatterplot(x=emp_phi, y=emp_ridgeless_risk, marker='X', s=60, label="no sketching", ax=ax)
                ax = sns.scatterplot(x=emp_phi, y=emp_sketch_risk, color='darkred', s=60, label=sketching_type)
                ax = sns.lineplot(x=phi, y=ridgeless_risk)
                ax = sns.lineplot(x=phi, y=sketch_risk, color='darkred')
            
            if snr is not None:
                if snr > 1: 
                    ax.axvline(x=1, color="grey", linestyle='dashed', linewidth=1)
                    ax.axvspan((1 - sigma/(2*alpha)), alpha/(alpha - sigma), facecolor='pink', alpha=0.3, **dict())
                    ax.axvline(x=(1 - sigma/(2*alpha)), color="grey", linestyle='dashed', linewidth=1)
                    ax.axvline(x=alpha/(alpha - sigma), color="grey", linestyle='dashed', linewidth=1)
                else:
                    ax.axvline(x=1, color="grey", linestyle='dashed', linewidth=1)
                    ax.axvspan((alpha**2 / (alpha**2 + sigma**2)), 10, facecolor='pink', alpha=0.3, **dict())
                    ax.axvline(x=(alpha**2 / (alpha**2 + sigma**2)), color="grey", linestyle='dashed', linewidth=1)

        ax.set(xticks = ticks, xticklabels = labels)
        ax.set_xlabel(r"$\phi$", fontsize=18)
        if correlated:
            ax.set_ylim(-5, 100)
            ax.axvline(x=1, color="grey", linestyle='dashed', linewidth=1)
        else:
            ax.set_ylim(-15, 60)
        ax.set_xlim(0.1, 10)
        ax.set_xscale('log')
        ax.set(xticks = ticks, xticklabels = labels)
        ax.legend(loc='lower right')
        ax.set_ylabel(r"$R_{(S, X)}\left(\widehat{\beta}^S ; \beta\right)$", fontsize=18)

        ax.get_legend().remove()
        handles, labels = ax.get_legend_handles_labels()

        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.52, -0.1), ncol = 4, prop={'size':12})
        plt.tight_layout()

        if save_path: 
            plt.savefig(save_path, dpi=400, facecolor='w', edgecolor='w', pad_inches=None)
        plt.show()


def plot_results_emp_m(files_base, files_emp_m, save_path=None):
    """
    Plots the results.
    
    Parameters:
    - files: a list of file paths, each file contains data for one line in the plot
    - save_path: the path where to save the plot
    """
    ticks = [0.1, 0.2, 0.5, 1, 2, 5, 10]
    labels = [i for i in ticks]

    with plt.style.context(['science', 'no-latex', 'std-colors']):
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        for i, file in enumerate(files_base):
            alpha, sigma, emp_phi, emp_sketch_risk, phi, ridgeless_risk = load_files_cmp(file, correlated=True)
            ax = sns.lineplot(x=phi, y=ridgeless_risk, label="no sketching")
            ax = sns.lineplot(x=emp_phi, y=emp_sketch_risk, color='darkred')
            ax = sns.scatterplot(x=emp_phi, y=emp_sketch_risk, color='darkred', label=r"sketching with ${m^{*}}$")

        markers = ['v', 'P', 's']
        colors = ["plum", "mediumvioletred", "mediumpurple"]
        for i, file in enumerate(files_emp_m):
            n_val = file.split('_')[-1].split('.')[0]
            alpha, sigma, emp_phi, emp_sketch_risk, phi, ridgeless_risk = load_files_cmp(file, correlated=True)
            ax = sns.scatterplot(x=emp_phi, y=emp_sketch_risk, color=colors[i], marker=markers[i], s=60, label=r"sketching with $\widehat{m}, n_{val}=$"+str(n_val))
            ax = sns.lineplot(x=emp_phi, y=emp_sketch_risk, color=colors[i])

        ax.set(xticks = ticks, xticklabels = labels)
        ax.set_xlabel(r"$\phi$", fontsize=18)
        ax.set_ylim(-5, 100)
        ax.axvline(x=1, color="grey", linestyle='dashed', linewidth=1)
        ax.set_xlim(0.1, 10)
        ax.set_xscale('log')
        ax.set(xticks = ticks, xticklabels = labels)
        ax.legend(loc='lower right')
        ax.set_ylabel(r"$R_{(S, X)}\left(\widehat{\beta}^S ; \beta\right)$", fontsize=18)

        ax.get_legend().remove()
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='lower center', bbox_to_anchor =(0.5, -0.4), ncol = 2, prop={'size':10})

        plt.tight_layout()

        if save_path: 
            plt.savefig(save_path, dpi=400, facecolor='w', edgecolor='w', pad_inches=None)
        plt.show()