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
import uuid

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY, 'src/assets/style.css'])
server = app.server

# Configurações
AUDIO_DIR = Path(__file__).parent / 'basslines'
FREQ_TICKS = [20, 50, 70, 100, 150, 250, 500, 1000, 1500, 2000, 3000, 5000, 10000, 15000, 20000]
DEFAULT_COLOR = '#FF5C00'
IR_COLOR = '#FFDF00'

# Função para calcular e armazenar em cache os dados do áudio
audio_cache = {}

def process_audio(filepath_or_audio, sr=None):
    filepath = None
    if isinstance(filepath_or_audio, Path):
        filepath = filepath_or_audio
        cache_key = str(filepath)
    elif isinstance(filepath_or_audio, np.ndarray):
        cache_key = str(uuid.uuid4())  # Gere uma chave única para dados brutos
        sr = sr # Use sr fornecido
    else:
        raise TypeError("Input deve ser um caminho de arquivo ou dados de áudio numpy.")

    if cache_key in audio_cache:
        return audio_cache[cache_key]

    if filepath:
        y, sr = librosa.load(filepath)
    else:
        y = filepath_or_audio

    n_fft = 2048
    hop_length = n_fft // 4
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    power = np.abs(stft)**2
    ref_power = np.mean(power)
    relative_spl_db = 10 * np.log10(power / ref_power)
    freq = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    magnitude_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    time = librosa.times_like(magnitude_db, sr=sr, hop_length=hop_length)
    mean_spl_db = np.mean(relative_spl_db, axis=1)
    duration = librosa.get_duration(y=y, sr=sr)

    audio_data = {
        'y': y,
        'sr': sr,
        'freq': freq,
        'mean_spl_db': mean_spl_db,
        'magnitude_db': magnitude_db,
        'time': time,
        'duration': duration,
    }
    audio_cache[cache_key] = audio_data  # Armazene no cache
    return audio_data

# Estilo do gráfico
grafico_config = {
    'plot_bgcolor': 'rgba(0, 0, 0, 0)',
    'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    'margin': {'pad': 0},
    'margin_b': 60,
    'font_color': '#d3d3d3',
}

# Função para plotar gráfico do Espectrograma
def create_spectrogram_figure(magnitude_db, freq, time):
    fig_spectrogram = px.imshow(magnitude_db,
                                y=freq, x=time,
                                color_continuous_scale=['#191919', '#FF5C00'],
                                aspect='auto', origin='lower',
                                labels=dict(x="Tempo (s)", y="Frequência (Hz)", color="Magnitude (dB)"),
                                )
    fig_spectrogram.update_layout(grafico_config)
    fig_spectrogram.update_yaxes(zeroline=False, type='log',
                                 tickvals=FREQ_TICKS, ticktext=FREQ_TICKS)

    return fig_spectrogram


# Função para plotar gráfico do SPL
def create_spl_figure(freq, mean_spl_db, name, color):
    fig_spl = px.line(log_x=True, title='Espectro de Potência',
                      labels={'x': 'Frequência (Hz)', 'y': 'Magnitude (dB Relative)'})
    fig_spl.add_trace(go.Scatter(x=freq, y=mean_spl_db, mode='lines', name=name, line=dict(color=color)))
    fig_spl.update_xaxes(range=[np.log10(20), np.log10(20000)],
                         tickvals=FREQ_TICKS, ticktext=FREQ_TICKS,
                         gridcolor='rgba(255, 255, 255, 0.04)')
    fig_spl.update_yaxes(gridcolor='rgba(255, 255, 255, 0.04)', zeroline=False)
    fig_spl.update_layout(grafico_config,
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
        filepath = AUDIO_DIR / selected_file
        audio_data = process_audio(filepath) # Use o cache

        if audio_data:
            audioinfo_children = html.Div([
                html.P(f'Sample Rate: {audio_data["sr"]} Hz'),
                html.P(f'Duração: {audio_data["duration"]:.2f} sec')
            ], className='looper-div')

            audio_src = audio_to_base64(audio_data['y'], audio_data['sr'])

            fig_spl = create_spl_figure(audio_data['freq'], audio_data['mean_spl_db'], 'SPL Original', DEFAULT_COLOR)
            fig_spectrogram = create_spectrogram_figure(audio_data['magnitude_db'], audio_data['freq'], audio_data['time'])

    if ir_contents and audio_data:  # Process IR only if audio is loaded
        content_type, content_string = ir_contents.split(',')
        decoded = base64.b64decode(content_string)

        try:
            ir, sr_ir = librosa.load(io.BytesIO(decoded))

            if audio_data['sr'] != sr_ir:
                ir = librosa.resample(ir, orig_sr=sr_ir, target_sr=audio_data['sr'])


            audio_ir = convolve(audio_data['y'], ir, mode='same') # Use audio_data['y']
            audio_ir /= np.abs(audio_ir).max()
            audio /= np.abs(audio).max()

            freq_ir, mean_spl_db_ir, stft_ir, magnitude_db_ir, time_ir, sr_ir = process_audio(audio_ir, sr)
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
    options=[{'label': f.name, 'value': f.name} for f in AUDIO_DIR.iterdir()
                if f.is_file() and f.suffix in ['.wav', '.mp3']],
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
        ], className='loop-cardbody'),
    ]),
]),

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
        ], className='irloader-cardbody'),
    ]),
]),


# Linhas
linha_titulo = dbc.Row([
    dbc.Col(titulo),
])

linha_pedais = dbc.Row([
    dbc.Col(pedal_looper, width=4),
    dbc.Col(pedal_irloader, width=4)
], justify='center')

linha_graficos = html.Div(id='graphs-container', children=[
    dbc.Row([
        dbc.Col(
            dcc.Loading(
                grafico_spl, color='#d3d3d3',
            ),
        )
    ]),

    dbc.Row([
        dbc.Col(
            dcc.Loading(
                grafico_spectrogram, color='#d3d3d3',
            ),
        )
    ]),
], style={'display': 'none'})

app.layout = dbc.Container([
    linha_titulo,
    linha_pedais,
    linha_graficos
])

if __name__ == '__main__':
    app.run_server(debug=True, port=8067)
