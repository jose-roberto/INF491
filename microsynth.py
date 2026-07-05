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

        sum_n = int(A_n * scale) + int(D_n * scale) + int(R_n * scale)

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
    sustain = np.ones(S_n) * S if S_n > 0 else np.array([])
    release = np.linspace(S, 0, R_n, endpoint=False) if R_n > 0 else np.array([])

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

# Envelope específico para a tabla
 
def _envelope_tabla(dur: float, sr: int,
                    A: float, D: float, S: float, R: float) -> np.ndarray:

    n_total = round(sr * dur)
    nA = round(sr * A)
    nD = round(sr * D)
    nR = round(sr * R)
    nS = max(0, n_total - nA - nD - nR)
 
    ataque  = np.linspace(0.0, 1.0, nA)
    decaim  = np.linspace(1.0, S,   nD)
    sustain = np.full(nS, S)
    release = np.linspace(S,   0.0, nR)
 
    env = np.concatenate([ataque, decaim, sustain, release])
 
    # Garante tamanho exato independentemente de arredondamentos
    if len(env) < n_total:
        env = np.pad(env, (0, n_total - len(env)))
    else:
        env = env[:n_total]
 
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
        Y = np.sin(2 * np.pi * f * t + fase_rad)

    elif forma == 'quadrada':
        # TODO: somar harmônicos ímpares
        for k in range(1, 2 * n, 2):

            if k * f >= sr / 2: 
                break

            Y += (1 / k) * np.sin(2 * np.pi * k * f * t + fase_rad)

    elif forma == 'triangular':
        # TODO: somar harmônicos ímpares com queda mais rápida
        # e alternância de sinal
        for idx, k in enumerate(range(1, 2 * n, 2)):

            if k * f >= sr / 2: 
                break

            sinal = (-1)**idx

            Y += sinal * (1 / k**2) * np.sin(2 * np.pi * k * f * t + fase_rad)

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

# Evolução Natural: Sintetizador Percussivo

def sintetiza_percussao(
    tipo: str,
    dur: float,
    amp: float = 1.0,
    f_tonal: float = None,
    sr: int = 44100
) -> np.ndarray:
   
    tipos_validos = ('bumbo', 'caixa', 'hihat', 'tabla')
    if tipo not in tipos_validos:
        raise ValueError(
            f"sintetiza_percussao: 'tipo' deve ser um de {tipos_validos}. "
            f"Recebido: '{tipo}'."
        )
    if dur <= 0:
        raise ValueError("sintetiza_percussao: 'dur' deve ser um valor positivo.")

    t = gera_tempo(dur, sr)
    n_amostras = len(t)

    # Bumbo 
    if tipo == 'bumbo':
        f0 = f_tonal if f_tonal is not None else 55.0   
        f1 = f0 * 0.12 # frequência final (12% da inicial)
 
        # frequência instantânea decrescente de forma exponencial
        dur_sweep = 0.18
        sweep_rate = np.log(f1 / f0) / dur_sweep
 
        f_inst = f0 * np.exp(sweep_rate * np.minimum(t, dur_sweep))
 
        # fase acumulada = integral numérica da freq. instantânea
        fase_acum = 2.0 * np.pi * np.cumsum(f_inst) / sr
        y_tonal = np.sin(fase_acum)
 
        # sub-harmônico em f0/2 adiciona graves profundos ("boom")
        fase_sub = 2.0 * np.pi * np.cumsum(f_inst / 2.0) / sr
        y_sub = np.sin(fase_sub) * 0.5
 
        # camada de ruído para o ataque percussivo ("click" da baqueta)
        ruido = np.random.randn(n_amostras) * 0.18
 
        # frequência instantânea decrescente de forma exponencial
        env_tonal = np.exp(-5.0 * t / dur)
        # sub decai um pouco mais rápido que o tonal principal
        env_sub   = np.exp(-7.0 * t / dur)
        # ruído some rápido (apenas no ataque)
        env_ruido = np.exp(-60.0 * t / dur)
 
        y = y_tonal * env_tonal + y_sub * env_sub + ruido * env_ruido

    # Caixa
    elif tipo == 'caixa':
        f0 = f_tonal if f_tonal is not None else 200.0

        # componente tonal com decaimento rápido
        y_tonal = np.sin(2.0 * np.pi * f0 * t)
        env_tonal = np.exp(-25.0 * t / dur)

        # ruído branco filtrado em passa-banda
        ruido = np.random.randn(n_amostras)
        b, a = signal.butter(4, [800 / (sr / 2), 8000 / (sr / 2)], btype='band')
        ruido_filtrado = signal.filtfilt(b, a, ruido)
        env_ruido = np.exp(-15.0 * t / dur)

        y = y_tonal * env_tonal * 0.4 + ruido_filtrado * env_ruido * 0.6

    # Hi-hat
    elif tipo == 'hihat':
        ruido = np.random.randn(n_amostras)

        f_corte = 7000.0
        b, a = signal.butter(4, f_corte / (sr / 2.0), btype='high')
        y_filtrado = signal.filtfilt(b, a, ruido)

        # Suavize o envelope para -12.0 ou -15.0 para dar mais corpo ao som
        env = np.exp(-15.0 * t / dur)
        y = y_filtrado * env

    # Tabla
    elif tipo == 'tabla':
        f0 = f_tonal if f_tonal is not None else 120.0

        # Camada 1: pitch bend exponencial descendente
        bend_semitones = 2.5
        bend_dur = min(0.08, dur * 0.3)
        bend_rate = (np.log(2) / 12) * bend_semitones / bend_dur
        excesso = (2 ** (bend_semitones / 12)) - 1.0
        f_inst = f0 * (1.0 + excesso * np.exp(-bend_rate * t))
        fase_acum = 2.0 * np.pi * np.cumsum(f_inst) / sr
        y_tonal = np.sin(fase_acum)

        # Camada 2: harmônicos afinados (2f e 3f)
        y_harm2 = np.sin(2.0 * np.pi * f0 * 2.0 * t) * 0.30
        y_harm3 = np.sin(2.0 * np.pi * f0 * 3.0 * t) * 0.15

        # Camada 3: ruído de membrana (transitório de ataque)
        ruido = np.random.randn(n_amostras)
        b, a = signal.butter(4, [300 / (sr / 2), 3000 / (sr / 2)], btype='band')
        ruido_filtrado = signal.filtfilt(b, a, ruido)
        env_ruido = np.exp(-35.0 * t / max(dur, 1e-6))

        # envelope ADSR da tabla
        A_t = min(0.005, dur * 0.05)
        D_t = min(0.060, dur * 0.20)
        R_t = min(0.150, dur * 0.30)
        S_n = 0.35
        env_tonal = _envelope_tabla(dur, sr, A_t, D_t, S_n, R_t)

        # mixagem das camadas
        y = (y_tonal        * 0.55
             + y_harm2      * 0.20
             + y_harm3      * 0.10
             + ruido_filtrado * env_ruido * 0.15)
        y = y * env_tonal

    peak = np.max(np.abs(y))
    if peak > 1e-9:
        y = y / peak

    return y * amp

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

        if self.forma == 'fm':
            
            f_c = self.fm_params['f_c']
            f_m = self.fm_params['f_m']
            I_fm = self.fm_params['I']
            tipo_fm = self.fm_params['tipo_fm']

            t, y = fm(dur, f_c, f_m, I_fm, tipo_fm, self.fase, self.unidade_fase, sr, True)
           
        elif self.forma in formas_validas:

            t, y = sintetiza(f, self.forma, self.n_harm, dur, sr, self.fase, self.unidade_fase, True)


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

            f_mod = self.am_params['f_m']
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

            f_mod = self.fm_params['f_m']
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

class InstrumentoPercussivo:

    def __init__(
        self,
        nome: str,
        tipo_perc: str,
        f_tonal: float = None,
        dur_padrao: float = 0.30
    ):

        tipos_validos = ('bumbo', 'caixa', 'hihat', 'tabla')
        if tipo_perc not in tipos_validos:
            raise ValueError(
                f"InstrumentoPercussivo: 'tipo_perc' deve ser um de "
                f"{tipos_validos}. Recebido: '{tipo_perc}'."
            )

        self.nome       = nome
        self.tipo_perc  = tipo_perc
        self.f_tonal    = f_tonal
        self.dur_padrao = dur_padrao

        self.forma         = 'percussivo'
        self.adsr_params   = None
        self.am_params     = None
        self.fm_params     = None
        self.filtro_params = None
        self.efeito_params = None

    def gerar_nota(
        self,
        f: float,
        dur: float,
        sr: int = 44100,
        amp: float = 1.0,
        retorna_t: bool = False
    ):

        dur_efetiva = max(dur, self.dur_padrao)
        f_ref = self.f_tonal if self.f_tonal is not None else f

        # Evolução C: modifica parâmetro tonal em função da velocity 
        if self.tipo_perc == 'tabla':
            f_ref = f_ref * (1.0 + 0.05 * amp)

        y = sintetiza_percussao(
            tipo=self.tipo_perc,
            dur=dur_efetiva,
            amp=amp,
            f_tonal=f_ref,
            sr=sr
        )

        if retorna_t:
            t = gera_tempo(dur_efetiva, sr)
            return t, y
        return y

    def __repr__(self):
        return (f"InstrumentoPercussivo(nome='{self.nome}', "
                f"tipo='{self.tipo_perc}', f_tonal={self.f_tonal})")

