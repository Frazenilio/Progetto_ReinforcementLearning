"""Utilità condivise per gli studi sugli algoritmi di Reinforcement Learning.

Funzioni per eseguire esperimenti con più seed e generare grafici
di learning curve e analisi di sensibilità agli iperparametri.
"""

import numpy as np
import matplotlib.pyplot as plt
import os


# ---------------------------------------------------------------------------
# Configurazione ambienti
# ---------------------------------------------------------------------------

def make_cartpole_bins(num_positions=24, num_angles=30,
                       num_velocities=24, num_angular_velocities=24):
    """Crea bins per CartPole-v1."""
    positions = np.linspace(-2.4, 2.4, num_positions)
    velocities = np.linspace(-3.5, 3.5, num_velocities)
    angles = np.linspace(-0.21, 0.21, num_angles)
    angular_velocities = np.linspace(-3.5, 3.5, num_angular_velocities)
    return [positions, velocities, angles, angular_velocities]


def make_mountaincar_bins(num_positions=24, num_velocities=24):
    """Crea bins per MountainCar-v0."""
    positions = np.linspace(-1.2, 0.6, num_positions)
    velocities = np.linspace(-0.07, 0.07, num_velocities)
    return [positions, velocities]


# Dizionario per accedere rapidamente a env_name e bins
ENVIRONMENTS = {
    "CartPole": {
        "env_name": "CartPole-v1",
        "bins_list": make_cartpole_bins(),
    },
    "MountainCar": {
        "env_name": "MountainCar-v0",
        "bins_list": make_mountaincar_bins(),
    },
}


# ---------------------------------------------------------------------------
# Esperimenti multi-seed
# ---------------------------------------------------------------------------

def run_experiment(algorithm_fn, env_name, bins_list, num_episodes=2000,
                   seeds=(42, 123, 456), **kwargs):
    """Esegue un algoritmo con più seed e raccoglie le reward history.

    Parametri
    ---------
    algorithm_fn : callable
        Funzione dell'algoritmo (es. q_learning, sarsa, montecarlo_control, ...).
    env_name : str
        Nome dell'ambiente Gymnasium.
    bins_list : list[np.ndarray]
        Bin per la discretizzazione.
    num_episodes : int
        Numero di episodi per run.
    seeds : tuple[int]
        Semi per la riproducibilità.
    **kwargs
        Iperparametri aggiuntivi da passare all'algoritmo.

    Ritorna
    -------
    all_rewards : np.ndarray di shape (n_seeds, num_episodes)
        Reward per episodio per ogni seed.
    """
    all_rewards = []
    for seed in seeds:
        np.random.seed(seed)
        result = algorithm_fn(env_name, bins_list, num_episodes=num_episodes, **kwargs)
        # Gli algoritmi di controllo ritornano (table, reward_history),
        # quelli di prediction (v_table, reward_history)
        if isinstance(result, tuple):
            reward_history = result[-1]  # ultimo elemento è sempre reward_history
        else:
            reward_history = result
        all_rewards.append(reward_history)
    return np.array(all_rewards)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def smooth(data, window=1):
    """Applica una media mobile per rendere le curve più leggibili."""
    if window <= 1 or len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode='valid')


def plot_learning_curve(results_dict, title, filename, window=50):
    """Plotta le learning curve (reward vs episodi) con media ± deviazione standard.

    Parametri
    ---------
    results_dict : dict[str, np.ndarray]
        Dizionario {label: array di shape (n_seeds, n_episodes)}.
    title : str
        Titolo del grafico.
    filename : str
        Percorso del file di output (PNG).
    window : int
        Dimensione della finestra per il smoothing.
    """
    plt.figure(figsize=(12, 7))

    for label, data in results_dict.items():
        mean_rewards = np.mean(data, axis=0)
        std_rewards = np.std(data, axis=0)

        # Smoothing
        mean_smooth = smooth(mean_rewards, window)
        std_smooth = smooth(std_rewards, window)
        episodes = np.arange(len(mean_smooth))

        plt.plot(episodes, mean_smooth, label=label, linewidth=2)
        plt.fill_between(episodes,
                         mean_smooth - std_smooth,
                         mean_smooth + std_smooth,
                         alpha=0.2)

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Episodio', fontsize=12)
    plt.ylabel('Reward Totale', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Grafico salvato: {filename}")


def plot_sensitivity(algorithm_fn, env_name, bins_list, param_name, param_values,
                     fixed_params, title, filename, num_episodes=2000,
                     seeds=(42, 123, 456), window=50):
    """Esegue analisi di sensibilità per un singolo iperparametro.

    Parametri
    ---------
    algorithm_fn : callable
        Funzione dell'algoritmo.
    env_name : str
        Nome dell'ambiente Gymnasium.
    bins_list : list[np.ndarray]
        Bin per la discretizzazione.
    param_name : str
        Nome dell'iperparametro da variare (es. 'alpha', 'gamma', 'epsilon').
    param_values : list
        Valori da testare per l'iperparametro.
    fixed_params : dict
        Valori fissi per gli altri iperparametri.
    title : str
        Titolo del grafico.
    filename : str
        Percorso del file di output (PNG).
    num_episodes : int
        Numero di episodi per run.
    seeds : tuple[int]
        Semi per la riproducibilità.
    window : int
        Dimensione della finestra per il smoothing.
    """
    results = {}
    for value in param_values:
        params = {**fixed_params, param_name: value}
        label = f"{param_name}={value}"
        print(f"  Running {label}...")
        rewards = run_experiment(algorithm_fn, env_name, bins_list,
                                 num_episodes=num_episodes, seeds=seeds,
                                 **params)
        results[label] = rewards

    plot_learning_curve(results, title, filename, window)


# ---------------------------------------------------------------------------
# Velocità di apprendimento (Speed of Learning)
# ---------------------------------------------------------------------------

def episodes_to_threshold(rewards_array, threshold, window=50):
    """Calcola il numero di episodi per raggiungere una soglia di reward.

    Per ogni seed, trova il primo episodio in cui la media mobile
    supera (o scende sotto, per reward negativi) la soglia.

    Parametri
    ---------
    rewards_array : np.ndarray di shape (n_seeds, n_episodes)
        Reward per episodio per ogni seed.
    threshold : float
        Soglia di reward da raggiungere.
    window : int
        Dimensione della finestra per la media mobile.

    Ritorna
    -------
    episodes_list : list[int | None]
        Numero di episodi per raggiungere la soglia per ogni seed.
        None se la soglia non è stata raggiunta.
    """
    episodes_list = []
    for seed_rewards in rewards_array:
        smoothed = smooth(seed_rewards, window)
        # Per reward negativi (MountainCar), "raggiungere" = superare la soglia
        reached = np.where(smoothed >= threshold)[0]
        if len(reached) > 0:
            # Aggiungi offset del window perché smooth riduce la lunghezza
            episodes_list.append(int(reached[0]) + window)
        else:
            episodes_list.append(None)
    return episodes_list


def compute_speed_metrics(rewards_array, thresholds, window=50):
    """Calcola metriche di velocità di apprendimento per più soglie.

    Parametri
    ---------
    rewards_array : np.ndarray di shape (n_seeds, n_episodes)
    thresholds : list[float]
        Soglie di reward da testare.
    window : int

    Ritorna
    -------
    metrics : dict[float, dict]
        Per ogni soglia: {
            'episodes': list[int|None],
            'mean': float|None,
            'std': float|None,
            'reached_ratio': float
        }
    """
    metrics = {}
    for thresh in thresholds:
        eps_list = episodes_to_threshold(rewards_array, thresh, window)
        valid = [e for e in eps_list if e is not None]
        metrics[thresh] = {
            'episodes': eps_list,
            'mean': np.mean(valid) if valid else None,
            'std': np.std(valid) if valid else None,
            'reached_ratio': len(valid) / len(eps_list),
        }
    return metrics


def plot_speed_of_learning(results_dict, thresholds, title, filename, window=50):
    """Plotta un grafico a barre della velocità di apprendimento.

    Per ogni algoritmo/configurazione, mostra il numero medio di episodi
    necessari per raggiungere ciascuna soglia.

    Parametri
    ---------
    results_dict : dict[str, np.ndarray]
        Dizionario {label: array di shape (n_seeds, n_episodes)}.
    thresholds : list[float]
        Soglie di reward da testare.
    title : str
        Titolo del grafico.
    filename : str
        Percorso del file di output (PNG).
    window : int
        Dimensione della finestra per la media mobile.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    labels = list(results_dict.keys())
    n_labels = len(labels)
    n_thresholds = len(thresholds)
    bar_width = 0.8 / n_thresholds
    x = np.arange(n_labels)

    for i, thresh in enumerate(thresholds):
        means = []
        stds = []
        for label in labels:
            metrics = compute_speed_metrics(results_dict[label], [thresh], window)
            m = metrics[thresh]
            means.append(m['mean'] if m['mean'] is not None else 0)
            stds.append(m['std'] if m['std'] is not None else 0)

        offset = (i - n_thresholds / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, means, bar_width, yerr=stds,
                      label=f"Soglia={thresh}", capsize=4, alpha=0.85)

        # Annota con "N/A" dove la soglia non è stata raggiunta
        for j, (mean_val, label) in enumerate(zip(means, labels)):
            if mean_val == 0:
                ax.text(x[j] + offset, 5, "N/A", ha='center', va='bottom',
                        fontsize=8, fontweight='bold', color='red')

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Algoritmo / Configurazione', fontsize=12)
    ax.set_ylabel('Episodi per raggiungere la soglia', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Grafico salvato: {filename}")


# ---------------------------------------------------------------------------
# Confronto tra algoritmi (stabilità + velocità in un unico grafico)
# ---------------------------------------------------------------------------

def plot_comparison(results_dict, title, filename, window=50):
    """Plotta il confronto tra algoritmi: learning curves.

    Combina in un unico grafico:
    - Le learning curve (media ± std) per mostrare la stabilità

    Parametri
    ---------
    results_dict : dict[str, np.ndarray]
        Dizionario {label_algoritmo: array di shape (n_seeds, n_episodes)}.
    title : str
        Titolo del grafico.
    filename : str
        Percorso del file di output (PNG).
    window : int
        Dimensione della finestra per il smoothing.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    colors = plt.cm.Set1.colors

    # --- Learning curves con bande di stabilità ---
    for i, (label, data) in enumerate(results_dict.items()):
        color = colors[i % len(colors)]
        mean_rewards = np.mean(data, axis=0)
        std_rewards = np.std(data, axis=0)

        mean_smooth = smooth(mean_rewards, window)
        std_smooth = smooth(std_rewards, window)
        episodes = np.arange(len(mean_smooth))

        ax.plot(episodes, mean_smooth, label=label, linewidth=2.5, color=color)
        ax.fill_between(episodes,
                        mean_smooth - std_smooth,
                        mean_smooth + std_smooth,
                        alpha=0.15, color=color)





    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Episodio', fontsize=12)
    ax.set_ylabel('Reward Totale (media mobile)', fontsize=12)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Grafico salvato: {filename}")


def plot_v_table_comparison(mc_v_table, td_v_table, env_label, bins_list, filename):
    """Genera heatmaps per confrontare visivamente le V-table di MC e TD(0)."""
    if env_label == "MountainCar":
        # MountainCar ha 2 dimensioni: posizione e velocità
        positions = bins_list[0]
        velocities = bins_list[1]
        
        # Inizializziamo le griglie (y=velocità, x=posizione)
        mc_grid = np.zeros((len(velocities), len(positions)))
        td_grid = np.zeros((len(velocities), len(positions)))
        
        for i, v in enumerate(velocities):
            for j, p in enumerate(positions):
                state = (p, v)
                mc_grid[i, j] = mc_v_table.get(state, 0.0)
                td_grid[i, j] = td_v_table.get(state, 0.0)
                
        xlabel = "Posizione"
        ylabel = "Velocità"
        extent = [positions[0], positions[-1], velocities[0], velocities[-1]]
        
    elif env_label == "CartPole":
        # CartPole ha 4 dimensioni: [pos, vel, angle, angular_vel]
        # Fissiamo pos e vel al valore più vicino a 0 (centro dei bin)
        positions = bins_list[0]
        velocities = bins_list[1]
        angles = bins_list[2]
        angular_velocities = bins_list[3]
        
        pos_idx = np.argmin(np.abs(positions - 0.0))
        vel_idx = np.argmin(np.abs(velocities - 0.0))
        fixed_pos = positions[pos_idx]
        fixed_vel = velocities[vel_idx]
        
        # Inizializziamo le griglie (y=vel_angolare, x=angolo)
        mc_grid = np.zeros((len(angular_velocities), len(angles)))
        td_grid = np.zeros((len(angular_velocities), len(angles)))
        
        for i, av in enumerate(angular_velocities):
            for j, a in enumerate(angles):
                state = (fixed_pos, fixed_vel, a, av)
                mc_grid[i, j] = mc_v_table.get(state, 0.0)
                td_grid[i, j] = td_v_table.get(state, 0.0)
                
        xlabel = "Angolo Asta (rad)"
        ylabel = "Velocità Angolare (rad/s)"
        extent = [angles[0], angles[-1], angular_velocities[0], angular_velocities[-1]]
    else:
        return

    # Plottiamo subplots: MC e TD(0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Determiniamo i limiti di colore per le due V-table in modo da usare la stessa scala
    vmin = min(mc_grid.min(), td_grid.min())
    vmax = max(mc_grid.max(), td_grid.max())
    if vmin == vmax:
        vmin, vmax = -1.0, 1.0

    im1 = axes[0].imshow(mc_grid, origin='lower', extent=extent, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Monte Carlo V-table ({env_label})")
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    fig.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(td_grid, origin='lower', extent=extent, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1].set_title(f"TD(0) V-table ({env_label})")
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(ylabel)
    fig.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Grafico confronto V-table salvato in: {filename}")

