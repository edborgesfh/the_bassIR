import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import dash
from dash import dcc, html, Input, Output, State
import librosa
import librosa.display
import os
import base64
import io
from scipy.signal import convolve
import dash_bootstrap_components as dbc
import soundfile as sf
from pathlib import Path
from functools import lru_cache


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY, 'src/assets/style.css'],
                suppress_callback_exceptions=True)
app.title = 'the_bassIR'
server = app.server

# Diretório dos arquivos de áudio
AUDIO_DIR = Path(__file__).parent / 'basslines'

# Labels para as frequências
freq_ticks = [20, 50, 70, 100, 150, 250, 500, 1000, 1500, 2000, 3000, 5000, 10000, 15000, 20000]

# Estilo do gráfico
grafico_config = {
    'plot_bgcolor': 'rgba(0, 0, 0, 0)',
    'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    'margin': {'pad': 0},
    'margin_b': 60,
    'font_color': '#d3d3d3',
}


# Função para cálculo das variáveis do áudio
def calculate_db(audio_data_or_filepath, sr, calibration_factor=1.0):
    if isinstance(audio_data_or_filepath, str):  # Verifica se é um caminho de arquivo
        y, sr = librosa.load(audio_data_or_filepath)
    elif isinstance(audio_data_or_filepath, np.ndarray):  # Verifica se são dados de áudio
        y = audio_data_or_filepath
    else:
        raise TypeError("Entrada inválida. Deve ser um caminho de arquivo (str) ou dados de áudio (np.ndarray).")

    n_fft = 2048
    hop_length = n_fft // 4
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)

    power = np.abs(stft)**2
    ref_power = np.mean(power)
    relative_spl_db = 10 * np.log10(power / ref_power) + calibration_factor

    freq = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    magnitude_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    time = librosa.times_like(magnitude_db, sr=sr, hop_length=hop_length)
    mean_spl_db = np.mean(relative_spl_db, axis=1)

    return freq, mean_spl_db, stft, magnitude_db, time, sr


# Função para carregamento dos áudios
@lru_cache(maxsize=32)
def load_audio(filepath):
    try:
        audio, sr = librosa.load(filepath)
        return audio, sr
    except Exception as e:
        print(f'Error loading audio: {e}')
        return None, None


# Função para plotar gráfico do Espectrograma
def create_spectrogram_figure(magnitude_db, freq, time):
    fig_spectrogram = px.imshow(magnitude_db,
                                y=freq, x=time,
                                color_continuous_scale=['#191919', '#FF5C00'],
                                aspect='auto', origin='lower',
                                labels=dict(x="Tempo (s)", y="Frequência (Hz)", color="Magnitude (dB)"),
                                )
    fig_spectrogram.update_layout(grafico_config,
                                  xaxis_title='Tempo (s)', yaxis_title='Frequências (Hz)')
    fig_spectrogram.update_yaxes(zeroline=False, type='log',
                                 tickvals=freq_ticks, ticktext=freq_ticks)

    return fig_spectrogram


# Função para plotar gráfico do SPL
def create_spl_figure(freq, mean_spl_db, name, color):
    fig_spl = px.line(log_x=True, title='Espectro de Potência',
                      labels={'x': 'Frequência (Hz)', 'y': 'Magnitude (dB Relative)'})
    fig_spl.add_trace(go.Scatter(x=freq, y=mean_spl_db, mode='lines', name=name, line=dict(color=color)))
    fig_spl.update_xaxes(range=[np.log10(20), np.log10(20000)],
                         tickvals=freq_ticks, ticktext=freq_ticks,
                         gridcolor='rgba(255, 255, 255, 0.04)')
    fig_spl.update_yaxes(gridcolor='rgba(255, 255, 255, 0.04)', zeroline=False)
    fig_spl.update_layout(grafico_config,
                          xaxis_title='Frequências (Hz)', yaxis_title='Magnitude (dB relativo)',
                          legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99))
    return fig_spl


# Função para decodificação do áudio
def audio_to_base64(audio, sr):
    audio_bytes = io.BytesIO()
    sf.write(audio_bytes, audio, sr, format='WAV')
    audio_bytes.seek(0)
    audio_base64 = base64.b64encode(audio_bytes.read()).decode('ascii')
    return f'data:audio/wav;base64,{audio_base64}'


# Callback principal
@app.callback(
    Output('spectrogram', 'figure'),
    Output('spl-graph', 'figure'),
    Output('audioinfo', 'children'),
    Output('irinfo', 'children'),
    Output('audio-player', 'src'),
    Output('ir-audio-player', 'src'),
    Input('audio-dropdown', 'value'),
    Input('upload-ir', 'contents'),
    State('upload-ir', 'filename'),
)
def update_graphs(selected_file, ir_contents, ir_filename):
    audio, sr = None, None
    fig_spectrogram = go.Figure()
    fig_spl = px.line()
    audio_src, ir_audio_src = None, None

    audioinfo_children = html.Div([html.P('No audio selected')], className='looper-div')
    irinfo_children = html.Div([html.P('No IR uploaded')], className='irloader-div')

    if selected_file:
        filepath = os.path.join(AUDIO_DIR, selected_file)
        audio, sr = load_audio(filepath)

        if audio is not None:

            duration = librosa.get_duration(y=audio, sr=sr)

            audioinfo_children = html.Div([
                html.P(f'Sample Rate: {sr} Hz'),
                html.P(f'Duração: {duration:.2f} sec')
            ], className='looper-div')

            audio_src = audio_to_base64(audio, sr)

            freq, mean_spl_db, stft, magnitude_db, time, sr = calculate_db(audio, sr)
            fig_spl = create_spl_figure(freq, mean_spl_db, 'SPL Original', '#FF5C00')
            fig_spectrogram = create_spectrogram_figure(magnitude_db, freq, time)

    if ir_contents and audio is not None:  # Process IR only if audio is loaded
        content_type, content_string = ir_contents.split(',')
        decoded = base64.b64decode(content_string)

        try:
            ir, sr_ir = librosa.load(io.BytesIO(decoded))
            if sr != sr_ir:
                ir = librosa.resample(ir, orig_sr=sr_ir, target_sr=sr)
                print(f"Nova taxa de amostragem do IR: {sr}")

            audio_ir = convolve(audio, ir, mode='same')
            audio_ir /= np.abs(audio_ir).max()
            audio /= np.abs(audio).max()

            freq_ir, mean_spl_db_ir, stft_ir, magnitude_db_ir, time_ir, sr_ir = calculate_db(audio_ir, sr)
            fig_spl.add_trace(go.Scatter(x=freq_ir, y=mean_spl_db_ir,
                                         mode='lines', name='SPL com IR', line=dict(color='#FFDF00')))

            irinfo_children = html.Div([
                html.P(f'{ir_filename}'),
                html.P(f'Sample Rate: {sr_ir} Hz'),
            ], className='irloader-div')

            ir_audio_src = audio_to_base64(audio_ir, sr)

        except Exception as e:
            print(f"Erro completo no carregamento/processamento do IR: {e}")  # Mais detalhes!
            import traceback
            traceback.print_exc()  # Imprime o traceback completo do erro

    return fig_spectrogram, fig_spl, audioinfo_children, irinfo_children, audio_src, ir_audio_src


# Callback para mostrar/esconder gráficos
@app.callback(
    Output('graphs-container', 'style'),
    Input('audio-dropdown', 'value')
)
def show_graphs(selected_file):
    if selected_file:
        return {'display': 'block'}  # Mostra os gráficos
    else:
        return {'display': 'none'}   # Esconde os gráficos


# Elementos do layout
titulo = html.H1(children='The_bassIR', className='audio-info'),

dropaudio = dcc.Dropdown(
    id='audio-dropdown',
    options=[{'label': f.name, 'value': f.name} for f in AUDIO_DIR.iterdir() if f.is_file() and
             f.suffix in ['.wav', '.mp3']],
    value=None,
    placeholder='Selecione um arquivo de áudio',
    className='dropdown',
)

audioinfo = html.Div(id='audioinfo')

audio_player = html.Div([
    html.Audio(id='audio-player', controls=True, className='audio-looper'),
    html.Div('bassIR — LOOPSTATION ', className='pedal-looper-nome'),
], style={'width': '100%'})

irloader = dcc.Upload(
    id='upload-ir',
    children=html.Div([html.P('Carregue um Impulse Response')]),
    className='upload-container',
    multiple=False
),

irinfo = html.Div(id='irinfo')

ir_audio_player = html.Div([
    html.Audio(id='ir-audio-player', controls=True, className='audio-irloader'),
    html.Div('bassIR — loader', className='pedal-irloader-nome'),
]),

grafico_spl = dcc.Graph(id='spl-graph', ),

grafico_spectrogram = dcc.Graph(id='spectrogram')

span = html.Div(
            [
                html.Br(),
                html.Span('Criado e desenvolvido por '),
                html.A(
                    'Eduardo Filho',
                    href='https://github.com/edborgesfh',
                    target='_blank',
                ),
            ], className='text-center'
)

# Layout dos pedais
pedal_looper = dbc.Card([
    dbc.CardBody([
        dbc.Row([
            dbc.Col([
                dbc.Row([
                    dbc.Col(dropaudio),
                ]),
                dbc.Row([
                    dbc.Col(audioinfo),
                ]),
                dbc.Row([
                    dbc.Col(audio_player),
                ]),
            ]),
        ]),
    ]),
], className='loop-cardbody'),

pedal_irloader = dbc.Card([
    dbc.CardBody([
        dbc.Row([
            dbc.Col([
                dbc.Row([
                    dbc.Col(irloader),
                ]),
                dbc.Row([
                    dbc.Col(irinfo),
                ]),
                dbc.Row([
                    dbc.Col(ir_audio_player),
                ]),
            ]),
        ]),
    ]),
], className='irloader-cardbody'),


# Linhas
linha_titulo = dbc.Row([
    dbc.Col(titulo),
])

linha_pedais = dbc.Row([
    dbc.Col(pedal_looper, lg=4, md=6, sm=12),
    dbc.Col(pedal_irloader, lg=4, md=6, sm=12)
], justify='center')

linha_graficos = html.Div(id='graphs-container', children=[
    dbc.Row([
        dbc.Col(
            dcc.Loading(
                    grafico_spl,
                    color="#d3d3d3", type="dot", fullscreen=False
            ),
        )
    ]),

    dbc.Row([
        dbc.Col(
            dcc.Loading(
                    grafico_spectrogram,
                    color="#d3d3d3", type="dot", fullscreen=False
            )
        )
    ]),
], style={'display': 'none'})

linha_span = dbc.Row([
    dbc.Col(span),
])

app.layout = dbc.Container([
    linha_titulo,
    linha_pedais,
    linha_graficos,
    linha_span,
], fluid=True)

if __name__ == '__main__':
    app.run_server(debug=True, port=8067)
