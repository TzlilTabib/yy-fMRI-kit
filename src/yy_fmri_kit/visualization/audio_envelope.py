import matplotlib.pyplot as plt
import numpy as np

def plot_env_tr(env_tr, title="Envelope (TR resolution)"):
    plt.figure(figsize=(10, 3))
    plt.plot(env_tr, lw=2)
    plt.xlabel("TR")
    plt.ylabel("z-scored envelope")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_raw_envelope(env, sr, tmax=30, title="Raw audio envelope"):
    t = np.arange(len(env)) / sr
    mask = t <= tmax
    plt.figure(figsize=(12, 3))
    plt.plot(t[mask], env[mask], lw=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Envelope amplitude")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_env_with_onsets(out_run: dict, run_label: str, tmax=60):
    df = out_run["df_run"]
    env = out_run["env_run_raw"]
    sr = out_run["info"]["sr_used"]

    t = np.arange(len(env)) / sr
    mask = t <= tmax

    plt.figure(figsize=(12, 3))
    plt.plot(t[mask], env[mask], lw=0.7)
    for onset in df["onset_s"]:
        if onset <= tmax:
            plt.axvline(onset, color="k", alpha=0.1)
    plt.title(f"{run_label} – RAW envelope with event onsets")
    plt.xlabel("Time (s)")
    plt.ylabel("Envelope")
    plt.tight_layout()
    plt.show()
