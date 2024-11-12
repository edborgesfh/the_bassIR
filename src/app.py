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
import scipy.signal
import dash_bootstrap_components as dbc
import soundfile as sf

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY, 'src/assets/style.css'])
server = app.server

# Lista de arquivos de áudio no diretório
AUDIO_DIR = 'basslines'

audio_files = [f for f in os.listdir(AUDIO_DIR) if os.path.isfile(os.path.join(AUDIO_DIR, f)) and
               (f.endswith('.wav') or f.endswith('.mp3'))]

# Labels para as frequências
freq_ticks = [20, 50, 70, 100, 150, 200, 250, 300, 400, 500, 750, 1000, 1200,
              1500, 2000, 3000, 4000, 5000, 8000, 10000, 12000, 15000, 20000]

# Estilo do gráfico
grafico_config = {
    # 'showlegend': False,
    'plot_bgcolor': 'rgba(0, 0, 0, 0)',
    'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    'margin': {'pad': 0},
    'margin_b': 60,
    'font_color': '#d3d3d3',
}

def calculate_db(audio, sr):
    n = len(audio)
    frequencies = np.fft.fftfreq(n, 1/sr)[:n//2]
    yf = np.fft.fft(audio)[:n//2]
    magnitude = np.abs(yf)
    magnitude_db = 20 * np.log10(magnitude/np.max(magnitude))
    magnitude_db_relative = magnitude_db - np.max(magnitude_db)
    magnitude_db_normal = np.interp(magnitude_db_relative, (-90, 0), (-18, 18))
    return frequencies, magnitude_db_relative, magnitude_db_normal

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
    audio_signal = None
    sr = None

    audioinfo_children = html.Div([
        html.P('Nenhum áudio selecionado'),
        html.P(''),
    ], className='looper-div')

    irinfo_children = html.Div([
        html.P('Nenhum IR carregado'),
        html.P(''),
    ], className='irloader-div')

    fig_spectrogram = go.Figure()
    fig_spl = px.line()

    audio_src = None
    ir_audio_src = None

    if selected_file:
        filepath = os.path.join(AUDIO_DIR, selected_file)
        try:
            audio_signal, sr = librosa.load(filepath)
            duration = librosa.get_duration(y=audio_signal, sr=sr)

            # Calcula o espectrograma
            S = librosa.feature.melspectrogram(y=audio_signal, sr=sr)
            S_dB = librosa.power_to_db(S, ref=np.max)
            fig_spectrogram = go.Figure(data=[go.Heatmap(z=S_dB, colorscale=['#191919', '#FF5C00'])])
            fig_spectrogram.update_layout(grafico_config,
                                          title=f'Espectrograma de {selected_file}',
                                          xaxis_title='Tempo (s)',
                                          yaxis_title='Frequência (Hz)')

            # Caracterizar o Div para as infomações do áudio
            audioinfo_children = html.Div([
                html.P(f'Sample Rate: {sr} Hz'),
                html.P(f'Duração: {duration:.2f} seg')
            ], className='looper-div')

            # Converte o áudio para formato base64 para o player
            audio_bytes = io.BytesIO()
            sf.write(audio_bytes, audio_signal, sr, format='WAV')
            audio_bytes.seek(0)
            audio_base64 = base64.b64encode(audio_bytes.read()).decode('ascii')
            audio_src = f'data:audio/wav;base64,{audio_base64}'

        except Exception as e:
            print(f'Erro ao carregar o áudio: {e}')

    if audio_signal is not None:
        frequencies, magnitude_db_relative, magnitude_db_normal = calculate_db(audio_signal, sr)

        fig_spl = px.line(log_x=True, title='Espectro de Potência',
                          labels={'x': 'Frequência (Hz)', 'y': 'Magnitude (dB Relativo)'})

        fig_spl.add_trace(go.Scatter(x=frequencies, y=magnitude_db_normal, mode='lines', name='SLP original',
                                     line=dict(color='#FF5C00'), yaxis='y1'))

        fig_spl.update_xaxes(range=[np.log10(20), np.log10(20000)],
                             tickvals=freq_ticks, ticktext=freq_ticks,
                             gridcolor='rgba(255, 255, 255, 0.04)'),
        fig_spl.update_yaxes(gridcolor='rgba(255, 255, 255, 0.04)')
        fig_spl.update_layout(grafico_config, yaxis_range=[-18, 18])

    if ir_contents:
        content_type, content_string = ir_contents.split(',')
        decoded = base64.b64decode(content_string)

        try:
            temp_filename = '../temp_ir.wav'
            with open(temp_filename, 'wb') as f:
                f.write(decoded)

            ir, sr_ir = librosa.load(io.BytesIO(decoded))
            if audio_signal is not None and sr != sr_ir:
                ir = librosa.resample(ir, orig_sr=sr_ir, target_sr=sr)

            if audio_signal is not None:
                audio_ir = scipy.signal.convolve(audio_signal, ir, mode='same')

                # Normalização pelo pico
                audio_ir = audio_ir / np.abs(audio_ir).max()
                audio_signal = audio_signal / np.abs(audio_signal).max()

                frequencies_ir, magnitude_db_relative_ir, magnitude_db_normal_ir = calculate_db(audio_ir, sr)
                fig_spl.add_trace(go.Scatter(x=frequencies_ir, y=magnitude_db_normal_ir,
                                             mode='lines', name='SLP com IR', line=dict(color='#FFDF00')))

            # Caracterizar o Div para as infomações do IR
            irinfo_children = html.Div([
                html.P(f'{ir_filename}'),
                html.P(f'Sample Rate: {sr_ir} Hz'),
                ], className='irloader-div',
            )

            # Converte o áudio com IR para base64
            ir_audio_bytes = io.BytesIO()
            sf.write(ir_audio_bytes, audio_ir, sr, format='WAV')  # Use soundfile.write
            ir_audio_bytes.seek(0)
            ir_audio_base64 = base64.b64encode(ir_audio_bytes.read()).decode('ascii')
            ir_audio_src = f'data:audio/wav;base64,{ir_audio_base64}'

        except Exception as e:
            print(f'Erro ao carregar ou processar a IR: {e}')

    return (fig_spectrogram, fig_spl,
            audioinfo_children, irinfo_children,
            audio_src, ir_audio_src,)

# Callback para mostrar/esconder gráficos
@app.callback(
    Output('graphs-container', 'style'),   # Output para controlar a visibilidade
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
    options=[{'label': f, 'value': f} for f in audio_files],
    value=None,
    placeholder='Selecione um arquivo de áudio',
    className='dropdown',
),

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
                grafico_spl, color='#FFDF00',
            ),
        )
    ]),

    dbc.Row([
        dbc.Col(
            dcc.Loading(
                grafico_spectrogram, color='#FFDF00',
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