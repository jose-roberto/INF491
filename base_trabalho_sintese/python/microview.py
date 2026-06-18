from scipy.signal import spectrogram
import numpy as np
import matplotlib.pyplot as plt
import music21 as m21

########################################################
def plot_sinais(
    ys: list[np.ndarray],
    labels: list[str]=None,
    sr: int=44100,
    fatia: int=None,
    titulo: str="Comparação de Sinais",
    figsize: tuple[int, int]=(12, 6)
) -> None:
    """
    Plota múltiplos sinais empilhados compartilhando o eixo x.

    Parâmetros:
    - ys: lista de arrays (sinais)
    - labels: lista de strings para eixo y
    - fs: frequência de amostragem (para gerar t)
    - fatia: quantidade de amostras a exibir
    - titulo: título da figura
    - figsize: tamanho da figura
    """

    n = len(ys)
    if n == 0:
        raise ValueError("Lista de sinais não pode ser vazia")

    if labels is None:
        labels = [f"Sinal {i}" for i in range(n)]

    if len(labels) != n:
        raise ValueError("Número de labels deve bater com número de sinais")

    # usa o tamanho do primeiro sinal como referência
    N = len(ys[0])

    # garante consistência
    for y in ys:
        if y.ndim != 1:
            raise ValueError("Todos os sinais devem ser 1D")
        if len(y) != N:
            raise ValueError("Todos os sinais devem ter o mesmo tamanho")

    if fatia is None:
        fatia = N
    else:
        fatia = min(fatia, N)

    # gera tempo
    t = np.arange(N) / sr

    # plot propriamente dito
    fig, axs = plt.subplots(n, 1, sharex=True, figsize=figsize)

    # caso n=1, axs não vem como lista
    if n == 1:
        axs = [axs]

    fig.suptitle(titulo)

    for i, (y, label) in enumerate(zip(ys, labels)):
        axs[i].plot(t[:fatia], y[:fatia])
        axs[i].set_ylabel(label)

    axs[-1].set_xlabel("Tempo (s)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

########################################################
def plot_envelope(t: np.ndarray, env: np.ndarray) -> None:
    """
    Plota um envelope (por exemplo, ADSR) ao longo do tempo.

    Parâmetros:
    - t   : vetor de tempo (numpy array)
    - env : vetor do envelope (mesmo tamanho de t)

    Retorna:
    - None (apenas exibe o gráfico)
    """

    # --------------------------
    # Validações básicas
    # --------------------------
    assert isinstance(t, np.ndarray), 't deve ser um numpy array'
    assert isinstance(env, np.ndarray), 'env deve ser um numpy array'
    assert len(t) > 0, 't não pode ser vazio'
    assert len(env) > 0, 'env não pode ser vazio'
    assert len(t) == len(env), 't e env devem ter o mesmo tamanho'

    # --------------------------
    # Plotagem
    # --------------------------
    plt.figure(figsize=(8, 3))

    plt.plot(t, env, linewidth=2)

    plt.title('Visualização do Envelope')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Amplitude')

    # grade leve para melhor leitura
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

########################################################
def plot_espectrograma(y, sr=44100):
    f, t, Sxx = spectrogram(y, sr)

    # evita log(0)
    Sxx = np.maximum(Sxx, 1e-10)

    Sxx_db = 10 * np.log10(Sxx)

    plt.pcolormesh(t, f, Sxx_db, shading='gouraud')
    plt.ylabel('Frequência (Hz)')
    plt.xlabel('Tempo (s)')
    plt.colorbar(label='dB')
    plt.show()

########################################################
def plot_fft(y, sr=44100, titulo="Espectro"):
    Y = np.fft.rfft(y)
    f = np.fft.rfftfreq(len(y), 1/sr)

    plt.figure(figsize=(10, 4))
    plt.plot(f, np.abs(Y))
    plt.title(titulo)
    plt.xlabel("Frequência (Hz)")
    plt.ylabel("Magnitude")
    plt.xlim(0, 5000)
    plt.tight_layout()
    plt.show()

########################################################
def plot_envelope_sinal(y, sr=44100, janela=1024):
    env = np.abs(y)
    env_smooth = np.convolve(env, np.ones(janela)/janela, mode='same')

    t = np.arange(len(y)) / sr

    plt.figure(figsize=(10, 4))
    plt.plot(t, env, alpha=0.3, label="Bruto")
    plt.plot(t, env_smooth, label="Envelope")
    plt.legend()
    plt.title("Envelope do Sinal")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Amplitude")
    # plt.tight_layout()
    plt.show()

########################################################
def plot_energia(buffers):
    energias = []
    for i in range(buffers.shape[0]):
        e = np.sum(buffers[i]**2)
        energias.append(e)

    plt.bar(range(len(energias)), energias)
    plt.title("Energia por Voz")
    plt.xlabel("Voz")
    plt.ylabel("Energia")
    plt.show()
    
########################################################
def freq2midi_int(f: float) -> int:
    return int(round(69 + 12 * np.log2(f / 440)))

def midi2note_name(midi: int) -> str:
    return m21.pitch.Pitch(midi).nameWithOctave

def plot_pianoroll(eventos, voz=0):
    midis = []

    # coleta valores MIDI
    for f, inicio, dur, amp in eventos:
        midi = freq2midi_int(f)
        midis.append(midi)
        plt.hlines(midi, inicio, inicio + dur, linewidth=1+3*amp, color=plt.cm.viridis(amp))

    # eixo Y com nomes de notas
    unique_midis = sorted(set(midis))
    labels = [midi2note_name(m) for m in unique_midis]

    plt.yticks(unique_midis, labels)

    plt.title(f"Piano Roll da voz {voz}")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Nota")
    plt.ylim(min(unique_midis)-1, max(unique_midis)+1)
    plt.grid()
    plt.show()

def plot_pianoroll_multivoz(lista_eventos, nomes_vozes=None):
    """
    lista_eventos: lista de listas de eventos
        [
            [(f, inicio, dur, amp), ...],  # voz 0
            [(f, inicio, dur, amp), ...],  # voz 1
            ...
        ]
    """

    plt.figure(figsize=(12, 6))

    all_midis = []
    cmap = plt.cm.get_cmap('tab10')  # cores distintas por voz

    for i, eventos in enumerate(lista_eventos):
        cor = cmap(i % 10)

        for f, inicio, dur, amp in eventos:
            midi = freq2midi_int(f)
            all_midis.append(midi)

            offset = i * 0.1

            plt.hlines(
                midi + offset,
                inicio,
                inicio + dur,
                linewidth=1 + 3 * amp,
                color=cor,
                alpha=0.8
            )

    # eixo Y
    unique_midis = sorted(set(all_midis))
    labels = [midi2note_name(m) for m in unique_midis]
    plt.yticks(unique_midis, labels)

    # limites
    if unique_midis:
        plt.ylim(min(unique_midis)-1, max(unique_midis)+1)

    # legenda
    if nomes_vozes:
        for i, nome in enumerate(nomes_vozes):
            plt.plot([], [], color=cmap(i % 10), label=nome)
        plt.legend()

    plt.title("Piano Roll (Multi-voz)")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Nota")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
