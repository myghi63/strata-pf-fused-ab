"""Fixed prompt set for the STRATA_PF_FUSED A/B.

KL:   one request per prompt, max_tokens=1. The engine dumps the logits of the
      first window (STRATA_DUMP_FIRST_LOGITS), i.e. the distribution the first
      generated token comes from. That is the "first-token KL" the Strata docs
      report for PF_FUSED.
LOOP: greedy, 600 tokens, two repeats. These are the prompts that make a
      degenerate repetition loop visible if the kernel rounding pushes the
      model into one.
"""

KL = [
    {
        "id": "kl_en_consensus",
        "max_tokens": 4,
        "text": """Consensus in a crash-stop distributed system is the problem of getting a set of
processes to agree on a single value even though some of them may stop responding at any
moment. The classic result is that agreement needs a majority: with three processes, one may
fail and the remaining two can still decide; with two, a single failure leaves no majority and
progress stops. Paxos and Raft both build on this observation. A leader collects proposals,
stamps them with a term or ballot number, and replicates them to a quorum. A follower accepts a
proposal only if the proposal's term is at least as large as any term it has already seen, which
is what prevents an older leader from overwriting a decision that a newer quorum already made.
The subtle part is not the normal path but the recovery path. When a leader disappears, the
remaining nodes must elect a new one without splitting the cluster into two groups that each
believe they are in charge. Raft solves this with randomized election timeouts: a candidate
waits a random interval before asking for votes, so two candidates rarely start at the same
instant, and the first one to reach a majority becomes leader for the next term. Paxos solves it
with ballot numbers that are globally ordered, so a later ballot always wins. Neither protocol
removes the need for a durable log. Before a node acknowledges a write, the entry must be on
stable storage on a majority of nodes; otherwise a crash can erase the only copy of a value the
cluster already reported as committed. This is why the storage layer matters as much as the
protocol. A slow disk does not merely slow one node down; it lengthens the window in which a
leader can be replaced, and it increases the chance that two terms overlap. Production systems
therefore separate the consensus log from the application state machine: the log is small,
append-only and fsynced, while the state machine can be rebuilt from the log after a restart.
The cost is that the log grows without bound unless it is compacted, and compaction is where
most real bugs live, because a snapshot taken at the wrong index can silently drop an entry
that a follower still needs. The practical lesson is that consensus is easy to describe and hard
to operate: the algorithm is a few hundred lines, but the failure handling, the membership
changes, the disk assumptions and the snapshot protocol are what decide whether a cluster
survives a bad night.

Responda em um paragrafo: qual e a ideia central do texto acima?""",
    },
    {
        "id": "kl_pt_navegacao",
        "max_tokens": 4,
        "text": """A expansao maritima europeia do seculo XV nao foi um acontecimento unico, mas um
processo lento que combinou tecnicas, interesses comerciais e disputas politicas. Durante
seculos, o Mediterraneo foi o centro do comercio europeu, e as rotas para o Oriente passavam
por intermediarios que encareciam a pimenta, a seda e o acucar. Quando os turcos otomanos
consolidaram o controle do leste do Mediterraneo, a pressao sobre essas rotas aumentou, e as
monarquias ibericas passaram a financiar expedicoes que buscassem um caminho alternativo pelo
Atlantico. O resultado dependeu de uma serie de inovacoes que hoje parecem modestas: a
caravela, com velas latinas que permitiam navegar contra o vento; o astrolabio e a balestilha,
que davam uma estimativa da latitude; a carta portulana, que registrava a costa conhecida; e o
diario de bordo, que transformava a experiencia de cada viagem em dado reutilizavel. Nenhuma
dessas ferramentas era nova em si, mas a combinacao permitiu que uma tripulacao voltasse com
informacao suficiente para que a viagem seguinte fosse menos arriscada. A colonizacao das
ilhas atlanticas, primeiro a Madeira e os Acores, depois as Canarias e Cabo Verde, funcionou
como um laboratorio: nelas se testaram cultivos, formas de administracao e o uso do trabalho
escravizado antes de o modelo ser levado para o continente americano. A chegada a America em
1492 e a circum-navegacao de 1519-1522 nao foram acidentes isolados, mas etapas de uma
expansao que ja estava em curso e que tinha como motor a busca de metais preciosos, terras e
rotas comerciais. O impacto foi desigual. Para as coroas ibericas, significou receita e
prestigio; para as populacoes americanas, significou doenca, deslocamento e destruicao de
sistemas politicos; para a Africa, significou o inicio de um trafico de pessoas que duraria
seculos. A historia dessa expansao, portanto, nao cabe em uma narrativa de descoberta heroica:
ela e tambem a historia de como o conhecimento tecnico, o capital e a violencia se combinaram
para reorganizar o mundo.

Responda em um paragrafo: qual e a ideia central do texto acima?""",
    },
    {
        "id": "kl_code_asyncio",
        "max_tokens": 4,
        "text": """O asyncio do Python resolve um problema especifico: manter muitas operacoes de
entrada e saida em andamento sem criar uma thread para cada uma. A unidade de trabalho e a
corrotina, declarada com async def e executada por um event loop. Quando uma corrotina chama
await em algo que ainda nao esta pronto, ela suspende e devolve o controle ao loop, que pode
avancar em outra tarefa. Isso e cooperativo: nada e interrompido no meio de um calculo. Se uma
corrotina passa dez segundos somando numeros, o loop inteiro fica parado por dez segundos. Por
isso bibliotecas de rede, como aiohttp, expoem versoes assincronas das operacoes, e por isso o
codigo que faz trabalho pesado de CPU deve ir para um processo separado, nao para uma thread
que ainda compartilha o GIL.

Um exemplo minimo mostra a estrutura:

    import asyncio

    async def fetch(nome, atraso):
        await asyncio.sleep(atraso)
        return f"{nome} pronto"

    async def main():
        tarefas = [fetch("a", 0.3), fetch("b", 0.1), fetch("c", 0.2)]
        for resultado in asyncio.as_completed(tarefas):
            print(await resultado)

    asyncio.run(main())

O ponto importante e que as tres chamadas comecam praticamente juntas: o tempo total e o do
maior atraso, nao a soma. O event loop mantem uma fila de callbacks prontos, uma fila de
timers e um conjunto de descritores de arquivo registrados no epoll do sistema operacional.
Quando um descritor fica legivel, o loop acorda a corrotina que o esperava. Esse desenho tem
duas consequencias praticas. A primeira e que erros de bloqueio sao dificeis de ver: uma
chamada sincrona escondida dentro de uma corrotina parece correta e so aparece como latencia
quando o sistema esta sob carga. A segunda e que o cancelamento precisa ser tratado de
proposito: quando uma tarefa e cancelada, o await levanta CancelledError, e blocos finally
precisam liberar recursos sem mascarar a excecao. Uma implementacao cuidadosa propaga o
cancelamento, usa timeouts explicitos e evita criar tarefas que ninguem vai aguardar, porque
uma tarefa orfa continua consumindo memoria e pode manter o processo vivo depois que o
trabalho principal terminou.

Responda em um paragrafo: qual e a ideia central do texto acima?""",
    },
    {
        "id": "kl_math_svd",
        "max_tokens": 4,
        "text": """A decomposicao em valores singulares de uma matriz A, de dimensoes m por n,
escreve A = U S V^T, onde U e ortogonal de ordem m, V e ortogonal de ordem n, e S e diagonal
retangular com entradas nao negativas s_1 >= s_2 >= ... >= s_r, sendo r o posto de A. As
colunas de U sao os vetores singulares a esquerda, as de V os vetores singulares a direita, e
os valores singulares medem quanto a matriz estica cada direcao. A construcao mais direta usa a
matriz simetrica A^T A: seus autovetores formam V, e os autovalores sao os quadrados dos
valores singulares. Se A tem posto completo por colunas, entao A^T A e positiva definida e a
decomposicao e unica a menos de sinal. Quando ha valores singulares repetidos, os subespacos
correspondentes ficam definidos, mas os vetores dentro deles nao.

A utilidade pratica vem de truncar a decomposicao. Se mantivermos apenas os k maiores valores
singulares, obtemos A_k = U_k S_k V_k^T, que e a melhor aproximacao de posto k na norma de
Frobenius, no sentido de que nenhuma outra matriz de posto k tem erro menor. Esse resultado,
conhecido como teorema de Eckart-Young, explica por que a SVD aparece em compressao de imagens,
sistemas de recomendacao e reducao de dimensionalidade: em muitos conjuntos de dados reais, a
energia se concentra em poucos valores singulares, e descartar o resto remove ruido mais do que
remove sinal. O custo computacional tambem importa. A SVD completa de uma matriz densa m por n
custa da ordem de O(m n min(m, n)) operacoes, o que e proibitivo para matrizes muito grandes.
Por isso, na pratica, usa-se a SVD truncada por metodos iterativos, como Lanczos ou
randomized SVD, que calculam apenas os k primeiros fatores e custam O(m n k). A escolha de k,
porem, nao e apenas numerica: ela envolve um julgamento sobre quanto erro o problema tolera.
Em um sistema de recomendacao, k pequeno pode significar recomendacoes mais estaveis e menos
personalizadas; k grande pode capturar preferencias individuais e tambem ruido. A SVD, nesse
sentido, nao decide nada sozinha: ela oferece uma familia de aproximacoes, e a escolha de onde
cortar e parte da modelagem.

Responda em um paragrafo: qual e a ideia central do texto acima?""",
    },
    {
        "id": "kl_agent_transcript",
        "max_tokens": 4,
        "text": """O agente recebeu a tarefa de investigar por que o servico de pedidos estava
retornando erro 500 de forma intermitente. A primeira acao foi listar os arquivos do projeto e
procurar por pontos de entrada e por configuracao de banco de dados. Em seguida, leu o
manipulador HTTP que cria pedidos e observou que a transacao era aberta antes da validacao do
corpo da requisicao. Depois, consultou os logs das ultimas duas horas e encontrou tres
padroes: picos de latencia a cada dez minutos, um aumento de conexoes abertas no pool e um
erro de serializacao que aparecia apenas quando dois pedidos do mesmo cliente chegavam quase
ao mesmo tempo. A hipotese inicial foi que o pool de conexoes estava pequeno, mas o teste com
um limite maior nao reduziu os erros. A segunda hipotese foi que a validacao tardia mantinha a
transacao aberta durante uma chamada externa, e o tempo dessa chamada variava com o provedor de
pagamento. Para confirmar, o agente instrumentou a transacao com marcas de tempo e reproduziu o
cenario com duas requisicoes concorrentes em um ambiente local. O resultado mostrou que a
segunda requisicao esperava pela primeira, mas o tempo de espera era curto; o erro vinha de
outro lugar. Ao revisar o codigo de serializacao, o agente encontrou um campo opcional que era
lido sem verificacao de nulo e que aparecia apenas em pedidos com cupom de desconto. A
correcao foi pequena: validar o campo antes de usar e mover a abertura da transacao para depois
da validacao. Em seguida, o agente escreveu um teste que enviava dois pedidos simultaneos com
cupom e verificava que ambos retornavam 201. O teste falhou na primeira execucao por causa de
uma ordem de insercao, foi ajustado para comparar conjuntos em vez de listas, e passou. Por
fim, o agente registrou a causa raiz, o teste de regressao e uma nota sobre o tempo de espera
do provedor externo, porque a latencia continuava alta mesmo sem o erro. A conclusao foi que
havia dois problemas independentes: um bug de validacao que causava o 500 e uma chamada
externa lenta que causava os picos. O primeiro foi corrigido; o segundo foi encaminhado para
outra equipe, com os dados coletados durante a investigacao.

Responda em um paragrafo: qual e a ideia central do texto acima?""",
    },
    {
        "id": "kl_json_api",
        "max_tokens": 4,
        "text": """A API de reservas expoe tres recursos: salas, reservas e disponibilidade. Uma
sala tem identificador estavel, nome, capacidade e uma lista de equipamentos. Uma reserva
aponta para uma sala, tem inicio e fim em UTC, um responsavel e um estado que pode ser
pendente, confirmada ou cancelada. A disponibilidade nao e um recurso armazenado: ela e
calculada a partir das reservas confirmadas e pendentes, descontando os intervalos ocupados.
Toda escrita exige um cabecalho de idempotencia; se a mesma chave chegar duas vezes com o
mesmo corpo, a segunda chamada devolve a resposta original em vez de criar outra reserva. Se a
mesma chave chegar com um corpo diferente, a API responde 409 e nao altera nada. As listagens
sao paginadas por cursor, e nao por offset, porque reservas podem ser criadas entre duas
paginas e um offset fixo faria um item aparecer duas vezes ou desaparecer. O cursor codifica o
ultimo identificador visto e a direcao da ordenacao; ele nao e um numero sequencial e nao deve
ser interpretado pelo cliente.

Um erro segue o formato problem+json, com campos type, title, status, detail e instance. O
campo detail e destinado a pessoa que esta depurando a integracao e pode citar o nome do campo
invalido, mas nunca inclui dados pessoais. Limites de taxa sao comunicados por tres cabecalhos:
o numero de requisicoes restantes, o instante em que a janela reinicia e o limite total. Uma
resposta 429 inclui o tempo de espera recomendado e nao deve ser tratada como falha permanente
pelo cliente. A autenticacao usa um token de curta duracao e um token de renovacao; o token de
acesso nunca e aceito em parametros de URL, apenas no cabecalho Authorization, para que nao
apareca em logs de proxy. A versao da API faz parte do caminho e nao do cabecalho, porque
clientes diferentes podem precisar de versoes diferentes ao mesmo tempo durante uma migracao.
Mudancas compativeis, como adicionar um campo opcional, nao exigem nova versao; mudancas
incompativeis, como remover um campo ou mudar o tipo de um identificador, exigem. Por fim, o
horario de inicio e fim e sempre tratado como intervalo semiaberto: uma reserva que termina as
10h libera a sala para outra que comece as 10h, e essa regra precisa ser documentada porque e
a fonte mais comum de duvida entre integradores.

Responda em um paragrafo: qual e a ideia central do texto acima?""",
    },
]

LOOP = [
    {
        "id": "loop_count_300",
        "max_tokens": 700,
        "text": "Conte de 1 a 300, um numero por linha, sem pular nenhum.",
    },
    {
        "id": "loop_words_a",
        "max_tokens": 700,
        "text": "Liste 120 palavras em portugues que comecam com a letra 'a', uma por linha, sem repetir nenhuma.",
    },
    {
        "id": "loop_essay_reading",
        "max_tokens": 700,
        "text": ("Escreva uma redacao de pelo menos 600 palavras sobre a importancia da leitura "
                 "na formacao de uma pessoa. Desenvolva uma ideia por paragrafo."),
    },
    {
        "id": "loop_explain_compiler",
        "max_tokens": 700,
        "text": ("Explique em detalhe como funciona um compilador, da analise lexica a geracao "
                 "de codigo, com exemplos em cada etapa. Escreva pelo menos 500 palavras."),
    },
    {
        "id": "loop_table_40",
        "max_tokens": 700,
        "text": ("Gere uma tabela markdown com 40 linhas numeradas. Cada linha deve ter: numero, "
                 "nome de uma cidade brasileira e sua populacao aproximada."),
    },
    {
        "id": "loop_while_python",
        "max_tokens": 700,
        "text": ("Escreva um programa Python com um laco while que imprime os numeros de 1 a 100, "
                 "um por linha, e explique cada parte do codigo."),
    },
    {
        "id": "loop_tcp_steps",
        "max_tokens": 700,
        "text": ("Explique passo a passo como funciona o protocolo TCP, numerando as etapas. "
                 "Repita cada etapa duas vezes: primeiro em uma frase curta, depois em uma frase longa."),
    },
    {
        "id": "loop_translate_30",
        "max_tokens": 700,
        "text": ("Traduza a frase 'O livro esta sobre a mesa' para 30 idiomas diferentes. "
                 "Escreva uma linha por idioma, no formato 'idioma: traducao'."),
    },
    {
        "id": "loop_fibonacci",
        "max_tokens": 700,
        "text": ("Continue a sequencia de Fibonacci a partir de 1, 1, 2, 3, 5, 8 e escreva os "
                 "50 primeiros termos, um por linha, mostrando a soma que gera cada termo."),
    },
    {
        "id": "loop_poem_sea",
        "max_tokens": 700,
        "text": "Escreva um poema com 40 versos sobre o mar, numerando cada verso.",
    },
    {
        "id": "loop_process_50",
        "max_tokens": 700,
        "text": ("Descreva o processo de publicacao de um livro em 50 etapas numeradas, "
                 "da ideia inicial ate a distribuicao nas livrarias."),
    },
    {
        "id": "loop_json_50",
        "max_tokens": 700,
        "text": ("Gere um JSON valido com uma lista de 50 objetos. Cada objeto deve ter id, nome, "
                 "categoria e preco. Nao use comentarios."),
    },
    {
        "id": "loop_dialogue_40",
        "max_tokens": 700,
        "text": ("Escreva um dialogo com 40 falas entre duas pessoas que tentam decidir onde jantar. "
                 "Alterne as falas e numere cada uma."),
    },
    {
        "id": "loop_summarize_cycle",
        "max_tokens": 700,
        "text": ("Escolha um tema de sua preferencia. Escreva um resumo de tres frases, depois expanda "
                 "o resumo em um texto de dez frases, depois resuma o texto novamente em tres frases. "
                 "Repita esse ciclo cinco vezes, sempre partindo do resumo anterior."),
    },
]

# --- long-context variants -------------------------------------------------
# STRATA_PF_FUSED's prompt kernels only run on a streamed chunk of
# stream_all_min() tokens or more (1024 by default; src/prefill/prefill.cpp,
# fused_layout()).  A ~600-token prompt runs MMQ in BOTH arms, so the first run
# measured nothing (the first-token KL came out exactly 0.0 for all six).
# Prefixing the same neutral context pushes every prompt past the threshold and
# leaves the question / the loop instruction at the end, where it belongs.
_PAD_PARTS = []
for _p in KL:
    _txt = _p["text"]
    _cut = _txt.find("Responda em um paragrafo")
    _PAD_PARTS.append(_txt[:_cut].rstrip() if _cut > 0 else _txt)
PAD = "\n\n".join(_PAD_PARTS)


def _with_pad(items):
    return [{**p, "text": PAD + "\n\n" + p["text"]} for p in items]


KL_SHORT, LOOP_SHORT = KL, LOOP
KL, LOOP = _with_pad(KL_SHORT), _with_pad(LOOP_SHORT)
