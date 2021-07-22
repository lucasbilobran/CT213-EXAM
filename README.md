# Exame de CT-213

Repositório contendo o projeto desenvolvido para o exame da disciplina CT-213: Inteligência Artificial para Robótica Móvel (Instituto Tecnológico de Aeronáutica) durante o primeiro semestre de 2021.

**Alunos**:
- Gianluigi Dal Toso
- Lucas Alberto Bilobran Lema

**Professor**:
- Marcos Ricardo Omena de Albuquerque Maximo

## <a name='TabeladeContedos'></a>Tabela de Conteúdos
<!-- vscode-markdown-toc -->
* [Descrição](#Descrio)
* [Instruções para execução](#Instruesparaexecuo)
	* [Pré-Requisitos](#Pr-Requisitos)
	* [Treinamento dos Modelos](#TreinamentodosModelos)
	* [Avaliação dos Modelos](#AvaliaodosModelos)
	* [Jogos de Atari](#JogosdeAtari)

<!-- vscode-markdown-toc-config
	numbering=false
	autoSave=true
	/vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->

---



## <a name='Descrio'></a>Descrição

Para complementar o aprendizado obtido durante as aulas teóricas da disciplina CT-213: Inteligência Artificial para Robótica Móvel, foram propostas ao longo do semestre, diversas atividades que consistiram de pequenos projetos no qual o aluno devia, a partir de um código fornecido previamente, finalizar a implementação escrevendo o código de inteligência artificial necessário para a correta execução do projeto, utilizando os conceitos aprendidos em aula. Para finalizar o semestre, foi proposto para o exame da disciplina a realização de um projeto livre, dessa vez sem código fornecido previamente, no qual os alunos ficaram encarregados de implementar uma técnica ou problema não implementado durante o curso.

Este repositório contém as implementações de código realizadas para o exame final da disciplina, sendo o problema escolhido para esta atividade a comparação dos algoritmos _SARSA_ e _Deep Q-Learning_, técnicas de aprendizado por reforço. Planejou-se aplicar ambas as técnicas na resolução de três problemas fornecidos pelo _OpenAI Gym_, sendo dois deles de controle: o primeiro sendo o equilíbrio de um pêndulo em cima de um carrinho (_cart pole_) e o segundo a subida de um carrinho em uma montanha (_mountain car_), este já resolvido nos laboratórios da disciplina. Por fim, o terceiro problema consistiria da resolução de um jogo de Atari, o jogo _Assault_.

## <a name='Instruesparaexecuo'></a>Instruções para execução

### <a name='Pr-Requisitos'></a>Pré-Requisitos
**Instalar as biblotecas do Python**
```bash
pip install -r requirements.txt
```

### <a name='TreinamentodosModelos'></a>Treinamento dos Modelos
Os arquivos `src/train_dqn.py` e `src/train_sarsa.py` são responsáveis por treinar utilizando os respectivos algoritmos. Os comandos para treinar o modelo são:

**Deep Q-Learning**:
```bash
cd src/
python train_dqn.py
```

**SARSA**:
```bash
cd src/
python train_sarsa.py
```

No início de cada um desses arquivos pode-se alterar (caso desejado) alguns parâmetros da execução:
```python
NUM_EPISODES = 50000 # Number of episodes used for training
RENDER = False  # If the Environment should be rendered

rom = 'CartPole-v1'
#rom = 'MountainCar-v0'
#rom = 'Assault-ram-v0'

fig_format = 'png'
# fig_format = 'eps'
# fig_format = 'svg'
```

Os modelos produzidos serão salvos no diretório `models/`.

### <a name='AvaliaodosModelos'></a>Avaliação dos Modelos
Os arquivos `src/evaluate_dqn.py` e `src/evaluate_sarsa.py` são responsáveis por avaliar os modelos gerados para cada algoritmo (salvos na pasta `models/`). Os comandos para avaliar os modelos são:

**Deep Q-Learning**:
```bash
cd src/
python evaluate_dqn.py
```

**SARSA**:
```bash
cd src/
python evaluate_sarsa.py
```

Assim como para o treinamento, no início de cada um desses arquivos pode-se alterar (caso desejado) alguns parâmetros da execução:
```python
NUM_EPISODES = 30 # Number of episodes used for training
RENDER = True  # If the Environment should be rendered

rom = 'CartPole-v1'
#rom = 'MountainCar-v0'
#rom = 'Assault-ram-v0'

fig_format = 'png'
# fig_format = 'eps'
# fig_format = 'svg'
```

### <a name='JogosdeAtari'></a>Jogos de Atari

TODO