# STRATA_PF_FUSED A/B

## Prompt speed (the KL requests)

| arm | prompts | prompt tokens | mean tok/s | median tok/s | mean prompt ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 6 | 3834-3893 | 1071.1 | 1072.1 | 3603.2 |
| 0 | 6 | 3834-3893 | 639.3 | 640.7 | 6037.1 |

fused on/off prompt speed ratio: **1.675x**

## Server loop detectors (whole server log of each arm)

| arm | reasoning loop, stop | reasoning loop, recover | one token repeated |
| --- | ---: | ---: | ---: |
| 1 | 2 | 0 | 0 |
| 0 | 2 | 0 | 0 |

## Repetition loops (short prompts)

### greedy0

| prompt | arm1 finish | arm1 tokens | arm1 signals | arm0 finish | arm0 tokens | arm0 signals |
| --- | --- | ---: | --- | --- | ---: | --- |
| loop_count_300 | stop | 1209 | - | stop | 1707 | - |
| loop_dialogue_40 | stop | 1408 | - | stop | 1250 | - |
| loop_essay_reading | stop | 5074 | - | stop | 5912 | - |
| loop_explain_compiler | stop | 2506 | - | stop | 2388 | - |
| loop_fibonacci | stop | 5859 | - | stop | 5668 | reasoning line x8 |
| loop_json_50 | stop | 3326 | - | stop | 3825 | - |
| loop_poem_sea | stop | 3255 | reasoning 6-gram x14 | stop | 2236 | - |
| loop_process_50 | stop | 3213 | - | stop | 3549 | - |
| loop_summarize_cycle | length | 6144 | - | stop | 6036 | reasoning 10-gram x7; answer line x10 |
| loop_table_40 | stop | 5261 | - | stop | 5258 | - |
| loop_tcp_steps | stop | 5491 | reasoning line x13; reasoning 10-gram x7 | stop | 4146 | - |
| loop_translate_30 | stop | 2650 | - | stop | 2426 | - |
| loop_while_python | stop | 462 | - | stop | 406 | - |
| loop_words_a | length | 2560 | server: the thinking repeated the same passages (coverage=1.000) at 2560 token; reasoning line x153; reasoning tail period 30; reasoning 10-gram x142; reasoning 6-gram x146 | length | 3584 | server: the thinking repeated the same passages (coverage=0.318) at 3584 token; reasoning line x15 |

### sampled1

| prompt | arm1 finish | arm1 tokens | arm1 signals | arm0 finish | arm0 tokens | arm0 signals |
| --- | --- | ---: | --- | --- | ---: | --- |
| loop_count_300 | stop | 1269 | - | stop | 3256 | - |
| loop_dialogue_40 | stop | 1440 | - | stop | 1330 | - |
| loop_essay_reading | stop | 5831 | - | stop | 5747 | - |
| loop_explain_compiler | stop | 2672 | - | stop | 3211 | - |
| loop_fibonacci | stop | 5674 | - | stop | 5859 | - |
| loop_json_50 | stop | 2759 | answer zlib 0.144 | stop | 2985 | answer zlib 0.147 |
| loop_poem_sea | stop | 2779 | - | stop | 1149 | - |
| loop_process_50 | stop | 2664 | - | stop | 2465 | - |
| loop_summarize_cycle | length | 6144 | - | stop | 5589 | answer line x10 |
| loop_table_40 | stop | 5247 | - | stop | 4084 | - |
| loop_tcp_steps | stop | 3912 | reasoning line x9 | stop | 4057 | - |
| loop_translate_30 | length | 3072 | server: the thinking repeated the same passages (coverage=0.284) at 3072 token | stop | 2609 | - |
| loop_while_python | stop | 429 | - | stop | 470 | - |
| loop_words_a | stop | 5041 | - | length | 3072 | server: the thinking repeated the same passages (coverage=0.398) at 3072 token; reasoning line x10 |

### Summary

| mode | arm | cut by the server's loop detector | text heuristics fired | hit max_tokens | wrote an answer |
| --- | --- | ---: | ---: | ---: | ---: |
| greedy0 | 1 | **1/14** | 3/14 | 2/14 | 13/14 |
| greedy0 | 0 | **1/14** | 3/14 | 1/14 | 13/14 |
| sampled1 | 1 | **1/14** | 2/14 | 2/14 | 13/14 |
| sampled1 | 0 | **1/14** | 3/14 | 1/14 | 13/14 |

The server's detector is the primary signal. The text heuristics are deliberately sensitive and also fire on tasks whose correct output is repetitive (a 50-item JSON list, a numbered poem, a Fibonacci table); read the flagged texts below before calling one a loop.

## Greedy repeatability (same prompt, same server, 3 requests)

| arm | identical pairs | common prefix (chars) 0-1 / 1-2 | draft accepted / drafted |
| --- | ---: | --- | --- |
| 1 | 0/2 | 222 / 648 | 234/255, 257/270, 203/217 |
| 0 | 0/2 | 222 / 222 | 251/261, 222/244, 231/239 |

The 2nd and 3rd requests reuse the cached prompt (conversation cache), so a difference between them comes from decoding, not from the prompt path. The draft acceptance counts differ between requests; speculative decoding is the likely source, but this run does not isolate it (`--drafting off` would).

## Long context (~187K-token prompt, sampled, up to 60K tokens)

| arm | prompt tokens | prompt tok/s | output tokens | decode tok/s | finish | notes found / right chapter | chapters missing / repeated | loop signals |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| 1 | 186578 | 1044 | 20427 | 45.0 | stop | 12/12, 12/12 | 0 / 0 | - |
| 0 | 186578 | 719 | 17728 | 43.3 | stop | 12/12, 12/12 | 0 / 0 | - |

## Outputs flagged as loops (last 500 characters)

### loop_fibonacci.greedy0 - arm0

Signals: reasoning line x8

```
2 = 5702887 + 9227465
24157817 = 9227465 + 14930352
39088169 = 14930352 + 24157817
63245986 = 24157817 + 39088169
102334155 = 39088169 + 63245986
165580141 = 63245986 + 102334155
267914296 = 102334155 + 165580141
433494437 = 165580141 + 267914296
701408733 = 267914296 + 433494437
1134903170 = 433494437 + 701408733
1836311903 = 701408733 + 1134903170
2971215073 = 1134903170 + 1836311903
4807526976 = 1836311903 + 2971215073
7778742049 = 2971215073 + 4807526976
12586269025 = 4807526976 + 7778742049
```

### loop_poem_sea.greedy0 - arm1

Signals: reasoning 6-gram x14

```
 e a lua puxa a água como um véu.
29. No porto, as cordas cheiram a viagem,
30. e o peixeiro conta moedas de bronze.
31. A criança vê o mar como miragem,
32. e o velho vê o mar como adeus.
33. O mar é espelho de céu e de dor,
34. onde o reflexo se quebra em mil pedaços.
35. A maré sobe, apaga a cor,
36. e deixa conchas como pequenos abraços.
37. No fundo, o navio dorme com peixes,
38. e o coral lhe faz coroa de cal.
39. O mar não guarda mágoas nem nomes,
40. mas leva o mundo em suas mãos de sal.
```

### loop_summarize_cycle.greedy0 - arm0

Signals: reasoning 10-gram x7; answer line x10

```
o crescer indefinidamente.
A compactação deve preservar entradas que ainda são necessárias aos seguidores.
Uma snapshot no índice errado pode causar perda silenciosa de dados.
O algoritmo é pequeno, mas a operação robusta depende de testes e monitoramento.

Resumo (3 frases):
Consenso distribuído depende de quóruns, termos ordenados e log durável.
Quóruns compartilhados impedem que decisões conflitantes sejam aceitas.
Operação cuidadosa evita que falhas, lentidão e compactação quebrem o sistema.
```

### loop_tcp_steps.greedy0 - arm1

Signals: reasoning line x13; reasoning 10-gram x7

```
o que os segmentos cheguem fora de ordem, duplicados ou atrasados, o TCP usa números de sequência para descartar duplicatas, reordenar os dados e entregar à aplicação um fluxo contínuo e sem lacunas.

11. Encerramento de conexão.
   - Curta: A conexão é fechada com FIN e ACK.
   - Longa: Cada lado envia um segmento FIN para indicar que não enviará mais dados, o outro lado confirma com ACK e pode enviar seu próprio FIN, permitindo que ambos encerrem a conexão de forma ordenada e liberem recursos.
```

### loop_words_a.greedy0 - arm1

Signals: server: the thinking repeated the same passages (coverage=1.000) at 2560 token; reasoning line x153; reasoning tail period 30; reasoning 10-gram x142; reasoning 6-gram x146

```
no.
62 afogar? no.
62 afogar? no.
62 afogar? no.
62 afogar? no.
62 afogar? no.
62 afogar? no.
62 afogar? no.
62 afogar? no.
62 afogar? no.
62 afogar? no.
62 afogar? no.
62 afogar? no.
62 afogar? no.
62 afogar? no.
62 afogar? no.
62 afogar? no.
62 afogar? no.
62 afogar? no.
62 afogar? no.
62 afogar? no.
62 afogar? no.
62 afogar? no.
62 afogar? no.
62 afogar? no.
62 afogar? no.
62 afogar? no.
62 afogar? no.
62 afogar? no.
62 afogar? no.
62 afogar? no.
62 afogar? no.
62 afogar? no.
62 afogar? no.
6
```

### loop_words_a.greedy0 - arm0

Signals: server: the thinking repeated the same passages (coverage=0.318) at 3584 token; reasoning line x15

```
no.
Maybe "acompanhar" no.
Let's use "acompanhar" no.
This is silly. Let's generate from categories:
- nouns: abacaxi, abacate, aba, abade, abajur, abano, abate, abelha, abismo, abraço, abuso, academia, acampamento, acaso, acento, aceno, acesso, acessório, acidental? adjective, acionista, acolhimento, acompanhamento, ação, acidez, acionamento? etc.
- verbs: abafar, abalar, abanar, abarcar, abater, abdicar, abjurar, aborrecer, abotoar, abraçar, abrandar, abrasar, abreviar, abrir, absorver, abster
```

### loop_json_50.sampled1 - arm1

Signals: answer zlib 0.144

```
categoria": "Vestuario", "preco": 82.20},
  {"id": 45, "nome": "Produto 45", "categoria": "Casa", "preco": 47.60},
  {"id": 46, "nome": "Produto 46", "categoria": "Alimentos", "preco": 16.45},
  {"id": 47, "nome": "Produto 47", "categoria": "Eletronicos", "preco": 165.75},
  {"id": 48, "nome": "Produto 48", "categoria": "Livros", "preco": 24.99},
  {"id": 49, "nome": "Produto 49", "categoria": "Vestuario", "preco": 99.00},
  {"id": 50, "nome": "Produto 50", "categoria": "Casa", "preco": 52.10}
]
```

### loop_json_50.sampled1 - arm0

Signals: answer zlib 0.147

```
 "categoria": "Livros", "preco": 75.00},
  {"id": 45, "nome": "Produto 45", "categoria": "Vestuário", "preco": 140.00},
  {"id": 46, "nome": "Produto 46", "categoria": "Casa", "preco": 200.00},
  {"id": 47, "nome": "Produto 47", "categoria": "Esporte", "preco": 115.00},
  {"id": 48, "nome": "Produto 48", "categoria": "Brinquedos", "preco": 65.00},
  {"id": 49, "nome": "Produto 49", "categoria": "Saúde", "preco": 100.00},
  {"id": 50, "nome": "Produto 50", "categoria": "Beleza", "preco": 45.00}
]
```

### loop_summarize_cycle.sampled1 - arm0

Signals: answer line x10

```
lacionar latência e erros.
Logs precisam conter contexto sem expor dados pessoais.
Testes de regressão precisam proteger a correção.
Testes de carga precisam revelar gargalos.
Contratos de API precisam ser estáveis.
Versionamento precisa preservar clientes antigos.
Documentação precisa explicar limites e estados.
A equipe precisa aprender com cada incidente.

Resumo (3 frases):
Sistemas confiáveis dependem de observação.
Testes sustentam mudanças seguras.
Operação cuidadosa reduz falhas futuras.
```

### loop_tcp_steps.sampled1 - arm1

Signals: reasoning line x9

```
ando os números de sequência, armazena temporariamente segmentos adiantados e entrega à aplicação apenas o fluxo contínuo de bytes que já foram confirmados e estão completos.

13. Frase curta: O encerramento usa segmentos FIN e ACK.
    Frase longa: Cada lado envia um segmento FIN para indicar que não enviará mais dados, o outro lado confirma com ACK e, após o fechamento completo e o tempo TIME_WAIT, a conexão é liberada para evitar que segmentos atrasados sejam confundidos com uma nova conexão.
```

### loop_translate_30.sampled1 - arm1

Signals: server: the thinking repeated the same passages (coverage=0.284) at 3072 token

```
en ligger på bordet.
8 Dinamarquês: Bogen ligger på bordet.
9 Norueguês: Boka ligger på bordet.
10 Islandês: Bókin er á borðinu.
11 Finlandês: Kirja on pöydällä.
12 Estoniano: Raamat on laual.
13 Letão: Grāmata ir uz galda.
14 Lituano: Knyga yra ant stalo.
15 Polonês: Książka jest na stole.
16 Tcheco: Kniha je na stole.
17 Húngaro: A könyv az asztalon van.
18 Esloveno: Knjiga je na mizi.
19 Croata: Knjiga je na stolu.
20 Romeno: Cartea este pe masă.
21 Grego: Το βιβλίο είναι πάνω στο τραπέζι.
22
```

### loop_words_a.sampled1 - arm0

Signals: server: the thinking repeated the same passages (coverage=0.398) at 3072 token; reasoning line x10

```
se "absurdo"? no.
Maybe "absurdo"? no.
Let's use "absurdo"? no.
Need list after absurdamente: "absurdo" no. Other words: "absurdo" no. "absurdo" no.
Let's use "absurdo" no.
Maybe "absurdo"? no.
Stop. Let's pick "absurdo"? no.
Use "absurdo"? no.
Okay 31: "absurdo"? no. Let's use "absurdo"? no.
I need other words: "absurdo" no. "absurdo" no.
Maybe "absurdo" no.
Let's use "absurdo"? no.
Could use "absurdo"? no.
Let's use "absurdo"? no.
Maybe I should not include absurd* too many. Use "absurdo" no.

```
