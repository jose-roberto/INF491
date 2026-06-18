import music21 as m21
import numpy as np
from microsynth import Instrumento, midi2freq

########################################################
#
# IMPLEMENTAÇÃO PRONTA
#
########################################################

def freq2midi(f):
    p = m21.pitch.Pitch(f)

    return p.midi, p.nameWithOctave

########################################################
#
# IMPLEMENTAÇÕES A EVOLUIR A PARTIR DO CÓDIGO PARCIAL
#
########################################################

class ArqMIDI:
    def __init__(
        self,
        arq: str
    ) -> None:
        """
        Lê um arquivo MIDI e converte suas partes/vozes em listas de eventos.

        Cada evento tonal deve ser representado como uma tupla:

            (frequencia, inicio, duracao, amplitude)

        em que:
        - frequencia está em Hz;
        - inicio está em segundos;
        - duracao está em segundos;
        - amplitude está em [0, 1].
        """

        try:
            self._rep = m21.converter.parse(arq)
        except Exception as e:
            raise Exception(e.args)

        # --------------------------------------------------
        # Andamento da música
        # --------------------------------------------------
        tempos = list(
            self._rep.recurse().getElementsByClass(m21.tempo.MetronomeMark)
        )

        if tempos:
            self.bpm = tempos[0].number
        else:
            self.bpm = 120  # fallback padrão

        # segundos por beat
        self._spb = 60 / self.bpm

        # duração total da música, em segundos
        self.duracao = self._rep.duration.quarterLength * self._spb

        # --------------------------------------------------
        # Partes/vozes
        # --------------------------------------------------
        self.parts = []

        for i, part in enumerate(self._rep.parts):
            nome = part.partName or part.id or f'Voz {i}'

            parte = {
                'nome': nome,
                'eventos': [],
                'percussiva': False
            }

            self.parts.append(parte)

            for el in part.flatten().notes:

                # --------------------------------------------------
                # TODO 1:
                # Calcular o instante de início do evento em segundos.
                #
                # Observação:
                # el.offset representa o início em semínimas.
                # Use self._spb para converter esse valor para segundos.
                # --------------------------------------------------

                # inicio = ...

                # --------------------------------------------------
                # TODO 2:
                # Calcular a duração do evento em segundos.
                #
                # Observação:
                # el.duration.quarterLength representa a duração em semínimas.
                # Use self._spb para converter esse valor para segundos.
                # --------------------------------------------------

                # dur = ...

                # --------------------------------------------------
                # TODO 3:
                # Obter a velocity MIDI e convertê-la para amplitude.
                #
                # A velocity MIDI normalmente varia de 0 a 127.
                # A amplitude usada pelo sintetizador deve ficar em [0, 1].
                #
                # Caso a velocity não esteja disponível, use 64 como valor padrão.
                # --------------------------------------------------

                # vel = ...
                # amp = ...

                # --------------------------------------------------
                # TODO 4:
                # Tratar acordes.
                #
                # Se el for um m21.chord.Chord, cada nota do acorde
                # deve gerar um evento independente no formato:
                #
                #     (frequencia, inicio, duracao, amplitude)
                #
                # A frequência deve ser obtida a partir do número MIDI
                # de cada nota do acorde.
                # --------------------------------------------------

                if isinstance(el, m21.chord.Chord):
                    pass

                # --------------------------------------------------
                # TODO 5:
                # Tratar notas comuns.
                #
                # Se el for um m21.note.Note, gere um evento no formato:
                #
                #     (frequencia, inicio, duracao, amplitude)
                #
                # A frequência deve ser obtida a partir do número MIDI
                # da nota.
                # --------------------------------------------------

                elif isinstance(el, m21.note.Note):
                    pass

                # --------------------------------------------------
                # TODO 6:
                # Tratar eventos percussivos.
                #
                # Nesta versão básica do MicroDAW, eventos percussivos
                # serão apenas identificados, mas não sintetizados.
                #
                # Se el for um m21.note.Unpitched:
                # - marque a parte atual como percussiva;
                # - ignore o evento.
                # --------------------------------------------------

                elif isinstance(el, m21.note.Unpitched):
                    pass

            # --------------------------------------------------
            # TODO 7:
            # Ordenar os eventos da parte pelo instante de início.
            # --------------------------------------------------

            # self.parts[-1]['eventos'].sort(...)

    def getPartList(
        self
    ) -> list[tuple]:
        """
        Retorna uma lista com informações resumidas sobre as partes/vozes.

        Cada item da lista retornada possui o formato:

            (nome, percussiva, quantidade_de_eventos)

        em que:
        - nome é o nome da parte/voz;
        - percussiva indica se a parte contém eventos percussivos;
        - quantidade_de_eventos indica quantos eventos tonais foram extraídos.
        """

        parts = []

        for part in self.parts:
            parts.append(
                (
                    part['nome'],
                    part['percussiva'],
                    len(part['eventos'])
                )
            )

        return parts

    def getPart(
        self,
        chave: int
    ) -> list[tuple]:
        """
        Retorna a lista de eventos de uma parte/voz específica.

        Parâmetros:
        - chave: índice da parte/voz desejada.

        Retorno:
        - lista de eventos da parte selecionada.
        """

        if not isinstance(chave, int) or chave < 0:
            raise ValueError('A chave tem de ser um inteiro >= 0')

        if chave >= len(self.parts):
            raise ValueError(f'Voz {chave} não encontrada')

        return self.parts[chave]['eventos']

########################################################

class Player:
    def __init__(
        self
    ) -> None:
        """
        Inicializa o player do MicroDAW.

        O player é responsável por:
        - carregar um arquivo MIDI;
        - associar instrumentos às partes/vozes;
        - gerar buffers de áudio por voz;
        - mixar os buffers;
        - retornar a waveform final.
        """

        self.arqMIDI = None
        self._limpaEstr()

    def _limpaEstr(
        self
    ) -> None:
        """
        Reinicializa as estruturas internas do player.
        """

        self.instrumentos = []
        self.buffers = None
        self.sr = 0
        self.ganhos = []

    def setArq(
        self,
        nome_arq: str,
        sr: int = 44100
    ) -> None:
        """
        Define o arquivo MIDI a ser processado.

        Parâmetros:
        - nome_arq: caminho do arquivo MIDI.
        - sr: taxa de amostragem a ser usada no processamento.
        """

        self._limpaEstr()

        self.arqMIDI = ArqMIDI(nome_arq)

        try:
            self.setSr(sr)
        except ValueError as e:
            raise ValueError(e.args)

    def setSr(
        self,
        sr: int
    ) -> None:
        """
        Define a taxa de amostragem do player.

        Parâmetros:
        - sr: taxa de amostragem, em amostras por segundo.
        """

        if not isinstance(sr, int) or sr <= 0:
            raise ValueError('A taxa de amostragem deve ser um inteiro positivo')

        self.sr = sr

    def setInstrumentos(
        self,
        instrumentos: list
    ) -> None:
        """
        Associa instrumentos às partes/vozes do arquivo MIDI.

        Cada item da lista pode ser:
        - um objeto Instrumento;
        - ou uma tupla (instrumento, ganho).

        O número de instrumentos deve ser igual ao número de partes/vozes
        presentes no arquivo MIDI.
        """

        if self.arqMIDI is None:
            raise ValueError('Não há um arquivo para atribuir instrumentos')

        if len(instrumentos) != len(self.arqMIDI.parts):
            raise ValueError(
                f'São esperados {len(self.arqMIDI.parts)} instrumentos, '
                f'mas foram informados {len(instrumentos)}'
            )

        self.instrumentos = []
        self.ganhos = []

        # --------------------------------------------------
        # TODO 1:
        # Percorrer a lista de instrumentos recebida.
        #
        # Cada item pode ser:
        # - um Instrumento;
        # - uma tupla (Instrumento, ganho).
        #
        # Para cada item:
        # - verificar se o instrumento é válido;
        # - armazenar o instrumento em self.instrumentos;
        # - armazenar o ganho correspondente em self.ganhos.
        #
        # Caso algum item seja inválido, lançar ValueError.
        # --------------------------------------------------

        # --------------------------------------------------
        # TODO 2:
        # Criar os buffers de áudio.
        #
        # Deve ser criado um array NumPy bidimensional com:
        # - uma linha para cada parte/voz;
        # - colunas suficientes para armazenar a música inteira.
        #
        # Recomenda-se adicionar um pequeno tempo extra ao final do buffer,
        # para acomodar releases, filtros ou pequenas sobras de sinal.
        # --------------------------------------------------

        # self.buffers = ...

    def processa(
        self,
        canais: int = 2,
        verbose: bool = False
    ) -> np.ndarray:
        """
        Processa o arquivo MIDI e gera o áudio sintetizado.

        Parâmetros:
        - canais:
            0 -> retorna buffers separados por voz;
            1 -> retorna áudio mono;
            2 -> retorna áudio estéreo.
        - verbose: se True, imprime informações durante o processamento.

        Retorno:
        - waveform sintetizada, de acordo com o número de canais solicitado.
        """

        if self.arqMIDI is None:
            raise ValueError('Não há arquivo para processar')

        if not isinstance(canais, int) or canais < 0 or canais > 2:
            raise ValueError(
                'Os canais devem ser inteiros: '
                '0 -> todos, 1 -> mono ou 2 -> estéreo'
            )

        if not len(self.instrumentos):
            raise ValueError('Os instrumentos não foram definidos')

        if self.buffers is None:
            raise ValueError('Os buffers ainda não foram criados')

        if self.sr <= 0:
            raise ValueError(f'Taxa de amostragem inválida: {self.sr}')

        nparts = len(self.arqMIDI.parts)
        N = self.buffers.shape[1]

        # --------------------------------------------------
        # TODO 3:
        # Percorrer as partes/vozes do arquivo MIDI.
        #
        # Para cada parte:
        # - obter a lista de eventos;
        # - obter o instrumento correspondente;
        # - sintetizar cada nota;
        # - posicionar a nota no buffer da voz.
        # --------------------------------------------------

        # for i in range(nparts):
        #     eventos = ...
        #     instrumento = ...
        #
        #     for evento in eventos:
        #         f, inicio, dur, amp = evento
        #
        #         y = ...
        #
        #         ini = ...
        #         fim = ...
        #
        #         self.buffers[i][ini:fim] += ...

        # --------------------------------------------------
        # TODO 4:
        # Normalizar ou controlar a amplitude de cada voz,
        # caso a equipe adote essa estratégia.
        # --------------------------------------------------

        if verbose:
            print('Buffers preenchidos')

        return self.getWav(canais=canais)

    def getWav(
        self,
        canais: int = 2
    ) -> np.ndarray:
        """
        Retorna a waveform final a partir dos buffers processados.

        Parâmetros:
        - canais:
            0 -> retorna os buffers separados por voz;
            1 -> retorna uma waveform mono;
            2 -> retorna uma waveform estéreo.

        Retorno:
        - array NumPy contendo o áudio sintetizado.
        """

        if self.arqMIDI is None:
            raise ValueError('Não há arquivo para processar')

        if not isinstance(canais, int) or canais < 0 or canais > 2:
            raise ValueError(
                'Os canais devem ser inteiros: '
                '0 -> todos, 1 -> mono ou 2 -> estéreo'
            )

        if not len(self.instrumentos):
            raise ValueError('Os instrumentos não foram definidos')

        if self.buffers is None:
            raise ValueError('Os buffers ainda não foram criados')

        # --------------------------------------------------
        # TODO 5:
        # Aplicar os ganhos de cada voz.
        #
        # Atenção:
        # Evite modificar self.buffers diretamente nesta etapa.
        # Use uma cópia dos buffers para evitar reaplicar ganhos
        # em chamadas sucessivas de getWav.
        # --------------------------------------------------

        # buffers_mix = ...

        # --------------------------------------------------
        # TODO 6:
        # Se canais == 0, retornar os buffers separados por voz.
        # --------------------------------------------------

        # --------------------------------------------------
        # TODO 7:
        # Misturar as vozes em um único sinal mono.
        # --------------------------------------------------

        # buffer = ...

        # --------------------------------------------------
        # TODO 8:
        # Normalizar a mistura final para evitar clipping.
        # --------------------------------------------------

        # --------------------------------------------------
        # TODO 9:
        # Retornar:
        # - buffer mono, se canais == 1;
        # - buffer estéreo, se canais == 2.
        #
        # Para estéreo simples, a mesma waveform mono pode ser
        # duplicada nos dois canais.
        # --------------------------------------------------