# view.py
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from controller import SpamController
import dash_bootstrap_components as dbc

# Инициализация Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Путь к модели
model_path = "/Users/kozzze/Desktop/Учеба/Course_Work/2/code/model"
controller = SpamController(model_path)

# Хранилище истории запросов
history = []

# Описание страницы с улучшенным дизайном
app.layout = html.Div([
    dbc.Container([
        html.H1("Проверка Спама", className="text-center my-4"),
        dcc.Input(id='input-text', type='text', placeholder='Введите сообщение...',
                  style={'width': '100%', 'padding': '10px', 'border-radius': '5px', 'font-size': '18px'}),

        html.Br(),
        html.Br(),

        # Используем стиль для широкой кнопки
        dbc.Button('Проверить', id='submit-btn', n_clicks=0, color="primary", size="lg",
                   style={'width': '100%', 'padding': '10px'}),

        html.Div(id='output-prediction', className="my-4", style={
            'font-size': '20px', 'font-weight': 'bold', 'color': 'green', 'text-align': 'center'}),

        html.Br(),

        # История запросов
        html.H3("История запросов", className="text-center"),
        html.Div(id="history-output",
                 style={'font-size': '16px', 'margin-top': '10px', 'padding': '10px', 'border': '1px solid #ddd'}),

        html.Br(),
        # Кнопка очистки истории (с использованием стиля)
        dbc.Button('Очистить историю', id='clear-history-btn', n_clicks=0, color="danger", size="lg",
                   style={'width': '100%', 'padding': '10px'})
    ], style={'max-width': '600px', 'margin': 'auto'})
])


@app.callback(
    [Output('output-prediction', 'children'),
     Output('history-output', 'children')],
    [Input('submit-btn', 'n_clicks'),
     Input('clear-history-btn', 'n_clicks')],
    [dash.dependencies.State('input-text', 'value')]
)
def update_output(submit_clicks, clear_clicks, text):
    global history

    # Если нажали кнопку очистки истории
    if clear_clicks > 0:
        history = []  # Очистить историю
        return "", "История очищена."

    # Если нажали кнопку "Проверить"
    if submit_clicks > 0 and text:
        result = controller.get_prediction(text)

        # Добавить запрос в историю
        history.append({"text": text, "prediction": result})

        # Формирование истории
        history_list = [html.P(f"{i + 1}. {entry['text']} — {entry['prediction']}") for i, entry in enumerate(history)]

        return f"Результат: {result}", history_list

    return "", []


if __name__ == '__main__':
    app.run_server(debug=True)