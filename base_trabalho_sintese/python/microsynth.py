import numpy as np
from scipy import signal

########################################################
#
# IMPLEMENTAÇÕES PRONTAS
#
########################################################
def gera_tempo(dur: float, sr: int) -> np.ndarray:
    N = round(sr * dur)
    return np.arange(N) / sr

########################################################
# função que encontra a frequência de uma nota a partir de seu número MIDI
def midi2freq(n: int) -> float:
    assert isinstance(n, (int)), 'Nota deve ser um inteiro'
    assert 0 <= n < 128, 'Nota deve estar entre [0, 127]'

    return 2**((n - 69)/12) * 440

########################################################
def fm_time_warp(
    y: np.ndarray,
    t: np.ndarray,
    f_m: float,
    I: float
) -> np.ndarray:
    """
    Aproxima FM aplicando warping temporal ao sinal y(t).

    Parâmetros:
    - y   : sinal original
    - t   : vetor de tempo
    - f_m : frequência da modulação (Hz)
    - I   : intensidade (SEGUNDOS de desvio no tempo)

    Retorna:
    - y_fm : sinal com efeito de vibrato (FM aproximado)
    """

    assert isinstance(y, np.ndarray)
    assert isinstance(t, np.ndarray)
    assert len(y) == len(t)

    # deslocamento temporal (warping)
    # I = deslocamento máximo em segundos
    tau = t + I * np.sin(2 * np.pi * f_m * t)

    # reamostragem por interpolação
    y_fm = np.interp(tau, t, y)

    return y_fm

########################################################
def filtro(
    y: np.ndarray,
    cutoff_low: float = None,
    cutoff_high: float = None,
    metodo: str = 'filtfilt',
    sr: int = 44100
) -> np.ndarray:

    if not isinstance(y, np.ndarray):
        raise ValueError("y deve ser um numpy array")

    if cutoff_low is not None and not isinstance(cutoff_low, (int, float)):
        raise ValueError("cutoff_low deve ser numérico")

    if cutoff_high is not None and not isinstance(cutoff_high, (int, float)):
        raise ValueError("cutoff_high deve ser numérico")

    if metodo not in ['lfilter', 'filtfilt']:
        raise ValueError("metodo deve ser 'lfilter' ou 'filtfilt'")

    if not isinstance(sr, int) or sr <= 0:
        raise ValueError("sr deve ser positivo")

    # --------------------------
    # Determina tipo de filtro (e também valida)
    # --------------------------
    if cutoff_low is None and cutoff_high is None:
        raise ValueError("Defina pelo menos um cutoff")

    if cutoff_low is not None and cutoff_high is not None:
        if not (0 < cutoff_low < cutoff_high < sr / 2):
            raise ValueError("Deve valer: 0 < cutoff_low < cutoff_high < sr/2")
        tipo = 'band'
    elif cutoff_low is not None:
        if not (0 < cutoff_low < sr / 2):
            raise ValueError("cutoff_low deve estar em (0, sr/2)")
        tipo = 'high'
    else:
        if not (0 < cutoff_high < sr / 2):
            raise ValueError("cutoff_high deve estar em (0, sr/2)")
        tipo = 'low'

    # --------------------------
    # Projeto do filtro
    # --------------------------
    if tipo == 'low':
        Wn = cutoff_high / (sr / 2)
    elif tipo == 'high':
        Wn = cutoff_low / (sr / 2)
    else:
        Wn = [cutoff_low / (sr / 2), cutoff_high / (sr / 2)]

    b, a = signal.butter(4, Wn, btype=tipo)

    # --------------------------
    # Aplicação
    # --------------------------
    if metodo == 'lfilter':
        sinal = signal.lfilter(b, a, y)
    else:
        sinal = signal.filtfilt(b, a, y)

    # normaliza
    maxS = np.max(np.abs(sinal))
    if maxS > 0:
        sinal /= maxS

    return sinal

########################################################
#
# IMPLEMENTAÇÕES A FAZER DO ZERO
#
########################################################

def freq2midi(f: float) -> int:
    
    assert isinstance(f, (int, float)), 'Frequência deve ser numérica'
    assert f > 0, 'Frequência deve ser positiva'

    return int(round(69 + 12 * np.log2(f / 440)))

########################################################

def fm(
    dur: float,
    f_c: float,
    f_m: float,
    I: float,
    tipo_fm: str = 'const',
    fase: float = 0.0,
    unidade_fase: str = 'graus',
    sr: int = 44100,
    retorna_t: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:

    assert isinstance(dur, (int, float)), 'dur deve ser um número'
    assert dur > 0, 'dur deve ser positiva'

    assert isinstance(f_c, (int, float)), 'f_c deve ser um número'
    assert f_c > 0, 'f_c deve ser positiva'

    assert isinstance(f_m, (int, float)), 'f_m deve ser um número'
    assert f_m >= 0, 'f_m deve ser não negativa'

    assert isinstance(I, (int, float)), 'I deve ser um número'
    assert isinstance(sr, int), 'sr deve ser um inteiro'

    assert tipo_fm in ('const', 'mult'), \
        "tipo_fm deve ser 'const' ou 'mult'"

    assert unidade_fase in ('graus', 'rad'), \
        "unidade_fase deve ser 'graus' ou 'rad'"

    t = gera_tempo(dur, sr)

    if unidade_fase == 'graus':
        fase = np.deg2rad(fase)

    if tipo_fm == 'mult':
        f_m = f_m * f_c

    y = np.sin(2 * np.pi * f_c * t + I * np.sin(2 * np.pi * f_m * t) + fase)

    return (t, y) if retorna_t else y

########################################################

def am(
    y: np.ndarray,
    f_mod: float,
    I: float = 0.5,
    sr: int = 44100
) -> np.ndarray:
    
    assert isinstance(y, np.ndarray), 'y deve ser um numpy array'
    assert isinstance(f_mod, (int, float)), 'f_mod deve ser um número'
    assert isinstance(I, (int, float)), 'I deve ser um número'
    assert isinstance(sr, int), 'sr deve ser um inteiro'

    t = gera_tempo(len(y)/sr, sr)

    mod = 1 + I * np.sin(2 * np.pi * f_mod * t)

    return y * mod


########################################################
#
# IMPLEMENTAÇÕES A EVOLUIR A PARTIR DO CÓDIGO PARCIAL
#
########################################################

def adsr(
    dur: float,
    sr: int,
    A: float = 0.1,
    D: float = 0.1,
    S: float = 0.6,
    R: float = 0.2
) -> np.ndarray:
    """
    Gera um envelope ADSR (Attack, Decay, Sustain, Release).

    Parâmetros:
    - dur: duração total do envelope, em segundos.
    - sr : taxa de amostragem, em amostras por segundo.
    - A  : duração do ataque, em segundos.
    - D  : duração do decaimento, em segundos.
    - S  : nível de sustentação, entre 0 e 1.
    - R  : duração da liberação, em segundos.

    Retorna:
    - env: array NumPy contendo o envelope ADSR.
    """

    # --------------------------
    # Validações básicas
    # --------------------------
    assert isinstance(dur, (int, float)) and dur > 0, "dur deve ser positiva"
    assert isinstance(sr, int) and sr > 0, "sr deve ser inteiro positivo"

    for name, val in [("A", A), ("D", D), ("R", R)]:
        assert isinstance(val, (int, float)) and val >= 0, f"{name} deve ser >= 0"

    assert isinstance(S, (int, float)) and 0 <= S <= 1, "S deve estar em [0, 1]"

    # Número total de amostras do envelope
    N = round(sr * dur)

    # --------------------------
    # TODO 1:
    # Converter A, D e R de segundos para amostras.
    # --------------------------

    A_n = int(A * sr)
    D_n = int(D * sr)
    R_n = int(R * sr)

    # --------------------------
    # TODO 2:
    # Calcular o número de amostras da fase de sustain.
    #
    # Atenção:
    # Caso A_n + D_n + R_n seja maior que N,
    # a implementação deve tratar esse caso para que
    # o envelope final não ultrapasse a duração esperada.
    # --------------------------

    sum_n = A_n + D_n + R_n
    if sum_n > N:

        scale = N / sum_n

        sum_n = int(A * scale) + int(A * scale) + int(A * scale)

    S_n = N - sum_n

    # --------------------------
    # TODO 3:
    # Construir as quatro fases do envelope:
    #
    # attack  : cresce de 0 até 1
    # decay   : decresce de 1 até S
    # sustain : permanece em S
    # release : decresce de S até 0
    #
    # Use operações vetorizadas com NumPy.
    # --------------------------

    attack = np.linspace(0, 1, A_n, endpoint=False) if A_n > 0 else np.array([])
    decay = np.linspace(1, S, D_n, endpoint=False) if D_n > 0 else np.array([])
    sustain = np.one(S_n) * S if S_n > 0 else np.array([])
    release = np.linspace(S, 0, R_n, endpoint=False) if D_n > 0 else np.array([])

    # --------------------------
    # TODO 4:
    # Concatenar as fases para formar o envelope final.
    # --------------------------

    env = np.concatenate([attack, decay, sustain, release])

    # --------------------------
    # TODO 5:
    # Garantir que o envelope tenha exatamente N amostras.
    # --------------------------

    if len(env) != N:
        env = env[:N]

    return env

########################################################

def sintetiza(
    f: float,
    forma: str = 'senoide',
    n: int = 5,
    dur: float = 2.0,
    sr: int = 44100,
    fase: float = 0.0,
    unidade_fase: str = 'graus',
    retorna_t: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Gera um sinal sintético periódico.

    Parâmetros:
    - f: frequência fundamental, em Hz.
    - forma: tipo de onda. Valores aceitos:
      'senoide', 'quadrada', 'triangular' ou 'dente'.
    - n: número máximo de harmônicos utilizados.
    - dur: duração do sinal, em segundos.
    - sr: taxa de amostragem, em amostras por segundo.
    - fase: fase inicial.
    - unidade_fase: unidade da fase, 'graus' ou 'rad'.
    - retorna_t: se True, retorna também o vetor de tempo.

    Retorna:
    - y, se retorna_t=False.
    - (t, y), se retorna_t=True.
    """

    # --------------------------
    # Validações básicas
    # --------------------------
    if not isinstance(f, (int, float)) or f <= 0:
        raise ValueError("Frequência deve ser positiva")

    if forma not in ['senoide', 'quadrada', 'triangular', 'dente']:
        raise ValueError("Forma de onda inválida")

    if not isinstance(n, int) or n <= 0:
        raise ValueError("Número de harmônicos deve ser inteiro positivo")

    if not isinstance(dur, (int, float)) or dur <= 0:
        raise ValueError("Duração deve ser positiva")

    if not isinstance(sr, int) or sr <= 0:
        raise ValueError("Taxa de amostragem deve ser positiva")

    if not isinstance(fase, (int, float)):
        raise ValueError("Fase deve ser numérica")

    if unidade_fase not in ['rad', 'graus']:
        raise ValueError("A unidade da fase deve ser 'rad' ou 'graus'")

    # --------------------------
    # Preparação
    # --------------------------
    t = gera_tempo(dur, sr)

    fase_rad = np.deg2rad(fase) if unidade_fase == 'graus' else fase

    Y = np.zeros_like(t)

    # --------------------------
    # TODO 1:
    # Gerar a forma de onda solicitada.
    #
    # A senoide deve ser gerada diretamente.
    #
    # As formas quadrada, triangular e dente devem ser
    # construídas por soma de componentes harmônicas.
    #
    # Atenção:
    # Não inclua harmônicos cuja frequência seja maior
    # ou igual à frequência de Nyquist, isto é, sr/2.
    # --------------------------

    if forma == 'senoide':
        # TODO: gerar senoide
        y = np.sin(2 * np.pi * f * t + fase_rad)

    elif forma == 'quadrada':
        # TODO: somar harmônicos ímpares
        for k in range(1, 2 * n, 2):

            if k * f >= sr / 2: 
                break

            y += (1 / k) * np.sin(2 * np.pi * k * f * t + fase_rad)

    elif forma == 'triangular':
        # TODO: somar harmônicos ímpares com queda mais rápida
        # e alternância de sinal
        for idx, k in enumerate(range(1, 2 * n, 2)):

            if k * f >= sr / 2: 
                break

            sinal = (-1)**idx

            y += sinal * (1 / k**2) * np.sin(2 * np.pi * k * f * t + fase_rad)

    elif forma == 'dente':
        # TODO: somar harmônicos inteiros
        for k in range(1, n + 1):

            if k * f >= sr / 2:
                break

            Y += (1 / k) * np.sin(2 * np.pi * k * f * t + fase_rad)

    # --------------------------
    # Normalização
    # --------------------------
    max_amp = np.max(np.abs(Y))
    if max_amp > 0:
        Y = Y / max_amp

    return (t, Y) if retorna_t else Y

########################################################

class Instrumento:

    def __init__(
        self,
        nome: str,
        forma: str,
        n_harm: int,
        adsr_params: tuple,
        fase: float = 0.0,
        unidade_fase: str = 'graus',
        am_params: dict = None,
        fm_params: dict = None,
        filtro_params: dict = None,
        efeito_params: dict = None
    ):
        self.nome = nome
        self.forma = forma
        self.n_harm = n_harm
        self.adsr_params = adsr_params
        self.fase = fase
        self.unidade_fase = unidade_fase
        self.am_params = am_params
        self.fm_params = fm_params
        self.filtro_params = filtro_params
        self.efeito_params = efeito_params

    def gerar_nota(
        self,
        f: float,
        dur: float,
        sr: int = 44100,
        amp: float = 1.0,
        retorna_t: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Gera uma nota sintetizada usando os parâmetros do instrumento.

        Parâmetros:
        - f: frequência da nota, em Hz.
        - dur: duração da nota, em segundos.
        - sr: taxa de amostragem.
        - amp: amplitude/intensidade da nota.
        - retorna_t: se True, retorna também o vetor de tempo.

        Retorna:
        - y ou (t, y), dependendo de retorna_t.
        """

        formas_validas = ['senoide', 'quadrada', 'triangular', 'dente', 'fm']

        if self.forma not in formas_validas:
            raise ValueError(f"Forma de onda inválida: {self.forma}")

        if self.forma == 'fm' and self.fm_params is None:
            raise ValueError("Parâmetros de FM devem ser informados quando forma='fm'")
        
        # --------------------------
        # TODO 1:
        # Gerar o sinal base.
        #
        # Se self.forma estiver entre:
        # 'senoide', 'quadrada', 'triangular' ou 'dente',
        # use a função sintetiza.
        #
        # Se self.forma == 'fm',
        # use a função fm.
        # --------------------------

        if self.forma in formas_validas:

            t, y = sintetiza(f, self.forma, self.n_harm, dur, sr, self.fase, self.unidade_fase, retorna_t)

        elif self.forma == 'fm':

            f_c = self.fm_params['f_c']
            f_m = self.fm_params['f_m']
            I_fm = self.fm_params['I']
            tipo_fm = self.fm_params['tipo_fm']

            t, y = fm(dur, f_c, f_m, I_fm, tipo_fm, self.fase, self.unidade_fase, sr, retorna_t)

        # --------------------------
        # TODO 2:
        # Aplicar envelope ADSR, caso self.adsr_params não seja None.
        # --------------------------

        if self.adsr_params is not None:

            A, D, S, R = self.adsr_params

            env = adsr(dur, sr, A, D, S, R) 

            y *= env  
            
        # --------------------------
        # TODO 3:
        # Aplicar modulação de amplitude, caso self.am_params não seja None.
        # --------------------------

        if self.am_params is not None:

            f_mod = self.am_params['f_mod']
            I_am = self.am_params['I']

            y = am(y, f_mod, I_am, sr)

        # --------------------------
        # TODO 4:
        # Aplicar FM por time warping, caso:
        #
        # - self.forma != 'fm'
        # - self.fm_params não seja None
        #
        # Nesse caso, use a função fm_time_warp.
        # --------------------------

        if self.forma != 'fm' and self.fm_params is not None:

            f_mod = self.fm_params['f_mod']
            I_fm = self.fm_params['I']

            y = fm_time_warp(y, t, f_mod, I_fm)

        # --------------------------
        # TODO 5:
        # Aplicar filtro, caso self.filtro_params não seja None.
        # --------------------------

        if self.filtro_params is not None:

            cutoff_low = self.filtro_params['cutoff_low']
            cutoff_high = self.filtro_params['cutoff_high']
            metodo = self.filtro_params['metodo']

            y = filtro(y, cutoff_low, cutoff_high, metodo, sr)

        # --------------------------
        # TODO 6 opcional:
        # Aplicar efeitos, caso self.efeito_params não seja None.
        #
        # A implementação de efeitos pode ser tratada como
        # evolução opcional do trabalho.
        # --------------------------

        # --------------------------
        # TODO 7:
        # Aplicar o fator de intensidade amp.
        # --------------------------

        y *= amp

        # --------------------------
        # TODO 8:
        # Retornar y ou (t, y), de acordo com retorna_t.
        # --------------------------
        return (t, y) if retorna_t else y

"""
Exemplo de uso:

piano = Instrumento(
    nome='Piano-like',
    forma='dente',      # onda naturalmente rica em harmônicos
    n_harm=10,          # quantidade moderada de harmônicos
    adsr_params=(
        0.01,           # Ataque rápido
        0.2,            # Decaimento moderado
        0.4,            # Sustain baixo
        0.3             # Liberação moderada
    ),
    fase=0,
    unidade_fase='graus',
    am_params=None,
    fm_params=None,
    filtro_params={
        'cutoff_high': 3000  # passa-baixa leve
    },
    efeito_params=None
)

# Gera uma nota C4 por 1 segundo.
# A nota MIDI 60 corresponde aproximadamente a 261.63 Hz.
t, y = piano.gerar_nota(
    f=midi2freq(60),
    dur=1.0,
    retorna_t=True
)
"""
