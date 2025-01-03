# bassIR - Simulador de Impulse Response para Baixo

Este projeto simula a aplicação de Impulse Responses (IRs) em samples de baixo usando Python, Dash e Plotly. Ele permite que os usuários carreguem seus próprios arquivos de áudio de baixo (WAV ou MP3) e IRs (WAV), visualizem o espectrograma e a resposta de frequência do áudio original e processado, e ouçam o resultado da convolução.

## Funcionalidades

* **Carregamento de áudio:** Selecione samples de baixo pré-carregados ou carregue os seus próprios.
* **Carregamento de IR:** Carregue arquivos WAV de Impulse Response para simular diferentes ambientes acústicos.
* **Visualização:** Exibe o espectrograma e a resposta de frequência (SPL) do áudio original e processado. Os gráficos são interativos, permitindo zoom e pan.
* **Reprodução de áudio:** Reproduz o áudio original e o áudio processado com a IR.
* **Interface de usuário intuitiva:** Design de interface de usuário amigável, simulando pedais de efeito, construído com Dash Bootstrap Components.
* **Tema escuro:** Interface com tema escuro para melhor visualização.

## Instalação

1. **Clone o repositório:**

    git clone https://github.com/seu_usuario/bassIR.git

2. **Crie um ambiente virtual (recomendado):**

    python3 -m venv .venv
    source .venv/bin/activate  # No Linux/macOS
    .venv\Scripts\activate  # No Windows

3. **Instale as dependências:**

    pip install -r requirements.txt

4. Crie a pasta 'basslines' na raiz do projeto e coloque suas linhas de baixo nela.

## Uso
1. Execute o aplicativo:

    python app.py

2. Abra o aplicativo no seu navegador: Acesse o endereço exibido no terminal (geralmente http://127.0.0.1:8050/).
3. Selecione um arquivo de áudio de baixo no dropdown.
4. Carregue um arquivo de Impulse Response (WAV).
5. Visualize os gráficos e ouça o áudio original e processado.

## Tecnologias Utilizadas
Python: Linguagem de programação principal.
Dash: Framework web para construção de dashboards analíticos.
Plotly: Biblioteca para criação de gráficos interativos.
Librosa: Biblioteca para análise e processamento de áudio.
SciPy: Biblioteca para computação científica, usada para a convolução.
NumPy: Biblioteca para manipulação de arrays numéricos.
Dash Bootstrap Components: Componentes de interface do usuário baseados no Bootstrap.
Soundfile: Leitura e gravação de arquivos de áudio.

## Estrutura do Projeto
app.py: Arquivo principal do aplicativo Dash.
assets/style.css: Arquivo CSS personalizado para estilização.
basslines/: Diretório para armazenar os arquivos de áudio de baixo (WAV ou MP3).
requirements.txt: Arquivo com as dependências do projeto.

## Contribuição
Contribuições são bem-vindas! Sinta-se à vontade para abrir issues e pull requests.

## Licença
MIT, 2024.

## Autor
Eduardo Borges Filho @edborgesfh