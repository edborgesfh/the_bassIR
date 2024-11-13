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

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY, 'src/assets/style.css'])
server = app.server

# Diretório dos arquivos de áudio
AUDIO_DIR = 'src/basslines'

# Labels para as frequências
freq_ticks = [20, 50, 70, 100, 150, 200, 250, 300, 400, 500, 750, 1000, 1200,
              1500, 2000, 3000, 4000, 5000, 8000, 10000, 12000, 15000, 20000]

# Estilo do gráfico
grafico_config = {
    'plot_bgcolor': 'rgba(0, 0, 0, 0)',
    'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    'margin': {'pad': 0},
    'margin_b': 60,
    'font_color': '#d3d3d3',
}


# Função para cálculo das variáveis do áudio
def calculate_db(audio, sr):
    n = len(audio)
    frequencies = np.fft.fftfreq(n, 1/sr)[:n//2]
    yf = np.fft.fft(audio)[:n//2]
    magnitude = np.abs(yf)
    magnitude_db = 20 * np.log10(magnitude/np.max(magnitude))
    magnitude_db_relative = magnitude_db - np.max(magnitude_db)
    magnitude_db_normal = np.interp(magnitude_db_relative, (-90, 0), (-18, 18))
    return frequencies, magnitude_db_relative, magnitude_db_normal


# Função para carregamento dos áudios
def load_audio(filepath):
    try:
        audio, sr = librosa.load(filepath)
        return audio, sr
    except Exception as e:
        print(f'Error loading audio: {e}')
        return None, None


# Função para plotar gráfico do Espectrograma
def create_spectrogram_figure(audio, sr):
    stft = librosa.stft(audio)
    spectrogram = np.abs(stft)
    spectrogram_db = librosa.amplitude_to_db(spectrogram)
    fig_spectrogram = go.Figure(data=[go.Heatmap(z=spectrogram_db, colorscale=['#191919', '#FF5C00'])])
    fig_spectrogram.update_layout(grafico_config, title='Espectrograma',
                                  xaxis_title='Tempo (s)', yaxis_title='Frequência (Hz)')
    return fig_spectrogram


# Função para plotar gráfico do SPL
def create_spl_figure(frequencies, magnitude_db_relative, name, color):
    fig_spl = px.line(log_x=True, title='Espectro de Potência',
                      labels={'x': 'Frequência (Hz)', 'y': 'Magnitude (dB Relative)'})
    fig_spl.add_trace(go.Scatter(x=frequencies, y=magnitude_db_relative, mode='lines', name=name, line=dict(color=color)))
    fig_spl.update_xaxes(range=[np.log10(20), np.log10(20000)],
                         tickvals=freq_ticks, ticktext=freq_ticks,
                         gridcolor='rgba(255, 255, 255, 0.04)')
    fig_spl.update_yaxes(gridcolor='rgba(255, 255, 255, 0.04)')
    fig_spl.update_layout(grafico_config, yaxis_range=[-120, 0])
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
            fig_spectrogram = create_spectrogram_figure(audio, sr)

            audioinfo_children = html.Div([
                html.P(f'Sample Rate: {sr} Hz'),
                html.P(f'Duração: {duration:.2f} sec')
            ], className='looper-div')

            audio_src = audio_to_base64(audio, sr)

            frequencies, magnitude_db_relative, magnitude_db_normal = calculate_db(audio, sr)
            fig_spl = create_spl_figure(frequencies, magnitude_db_relative, 'SPL Original', '#FF5C00')

    if ir_contents and audio is not None:  # Process IR only if audio is loaded
        content_type, content_string = ir_contents.split(',')
        decoded = base64.b64decode(content_string)

        try:
            ir, sr_ir = librosa.load(io.BytesIO(decoded))
            if sr != sr_ir:
                ir = librosa.resample(ir, orig_sr=sr_ir, target_sr=sr)

            audio_ir = convolve(audio, ir, mode='same')
            audio_ir /= np.abs(audio_ir).max()
            audio /= np.abs(audio).max()

            frequencies_ir, magnitude_db_relative_ir, magnitude_db_normal_ir = calculate_db(audio_ir, sr)
            fig_spl.add_trace(go.Scatter(x=frequencies_ir, y=magnitude_db_relative_ir,
                                         mode='lines', name='SPL com IR', line=dict(color='#FFDF00')))

            irinfo_children = html.Div([
                html.P(f'{ir_filename}'),
                html.P(f'Sample Rate: {sr_ir} Hz'),
            ], className='irloader-div')

            ir_audio_src = audio_to_base64(audio_ir, sr)

        except Exception as e:
            print(f'Error loading or processing IR: {e}')

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
    options=[{'label': f, 'value': f} for f in os.listdir(AUDIO_DIR) if os.path.isfile(os.path.join(AUDIO_DIR, f))
             and (f.endswith('.wav') or f.endswith('.mp3'))],
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
