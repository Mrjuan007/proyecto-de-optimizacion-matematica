import sys
from dash import Dash, dcc, html, Input, Output, callback
from dash.dash_table import DataTable
from pulp import LpVariable, LpProblem, LpMaximize
import plotly.graph_objs as go
import numpy as np

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = Dash(__name__)

# Estilos generales
gen_style = {
    'display': 'inline-block',
    'width': '20%',
    'margin': '10px',
    'padding': '15px',
    'font-size': '1.2em',
    'font-family': 'Product Sans, sans-serif',
    'color': "#FFFFFF",
    'background': '#444',
    'border': '1px solid #32bdbd',
    'border-radius': '8px',
    'box-shadow': '2px 2px 5px #888888',
    'text-align': 'center'
}

# Estilo del contenedor de salida
output_style = {
    'color': "#32bdbd",
    'font-family': 'Product Sans, sans-serif',
    'font-size': '1.5em',
    'margin-top': '20px',
    'text-align': 'center',
    'padding': '20px',
    'border': '2px solid #32bdbd',
    'border-radius': '10px',
    'background': '#333',
    'width': '60%',
    'margin': '20px auto',
    'box-shadow': '2px 2px 5px #888888'
}

# Estilo del gráfico
graph_style = {
    'margin-top': '40px',
    'width': '80%',
    'margin': '20px auto',
    'box-shadow': '2px 2px 5px #888888'
}

# Estilo del contenedor principal
main_container_style = {
    'backgroundColor': '#222',
    'padding': '40px',
    'min-height': '100vh',
    'border': '2px solid #32bdbd',
    'border-radius': '10px',
    'box-shadow': '5px 5px 15px #888888'
}

app.layout = html.Div(
    style=main_container_style,  # Aplicar estilo al contenedor principal
    children=[
        html.H2(
            'Maximización de Ganancias en Perfumes',
            style={
                'color': "#32bdbd",
                'font-family': 'Product Sans, sans-serif',
                'font-size': '3em',
                'text-align': 'center',
                'margin-bottom': '30px'
            }
        ),
        html.P(
            "Por favor, ingrese los valores por onza de los siguientes productos:",
            style={
                'color': "#32bdbd",
                'font-family': 'Product Sans, sans-serif',
                'font-size': '1.5em',
                'text-align': 'center',
                'margin-bottom': '40px'
            }
        ),
        html.Div(
            style={'display': 'flex', 'flex-wrap': 'wrap', 'justify-content': 'space-around'},
            children=[
                html.Div([
                    html.Label("Brutte Regular:", style={'color': '#32bdbd', 'font-size': '1.2em'}),
                    dcc.Input(
                        id="brutte_regular",
                        type="number",
                        placeholder="Brutte Regular",
                        style={**gen_style, 'width': '100%'}  # Añadir estilo para el ancho del input
                    )
                ], style={'width': '45%', 'text-align': 'center'}),  # Contenedor 1
                html.Div([
                    html.Label("Chanelle Regular:", style={'color': '#32bdbd', 'font-size': '1.2em'}),
                    dcc.Input(
                        id="chanelle_regular",
                        type="number",
                        placeholder="Chanelle Regular",
                        style={**gen_style, 'width': '100%'}  # Añadir estilo para el ancho del input
                    )
                ], style={'width': '45%', 'text-align': 'center'}),  # Contenedor 2
                html.Div([
                    html.Label("Brutte Luxury:", style={'color': '#32bdbd', 'font-size': '1.2em'}),
                    dcc.Input(
                        id="brutte_luxury",
                        type="number",
                        placeholder="Brutte Luxury",
                        style={**gen_style, 'width': '100%'}  # Añadir estilo para el ancho del input
                    )
                ], style={'width': '45%', 'text-align': 'center'}),  # Contenedor 3
                html.Div([
                    html.Label("Chanelle Luxury:", style={'color': '#32bdbd', 'font-size': '1.2em'}),
                    dcc.Input(
                        id="chanelle_luxury",
                        type="number",
                        placeholder="Chanelle Luxury",
                        style={**gen_style, 'width': '100%'}  # Añadir estilo para el ancho del input
                    )
                ], style={'width': '45%', 'text-align': 'center'})  # Contenedor 4
            ]
        ),
        html.Div(
            id="output",
            style=output_style
        ),
        html.Div(
            id="table-container",
            children=DataTable(id="result-table"),
            style=output_style
        ),
        html.Div(
            id="total-profit",
            style=output_style
        ),
        html.Div(
            id="materia-prima",
            style=output_style
        ),
        html.Div(
            id="horas-laboratorio",
            style=output_style
        ),
        dcc.Graph(id="perfume-production-chart", style=graph_style),  # Agregar un componente para mostrar el gráfico
        dcc.Graph(id="profitability-pie-chart", style=graph_style),  # Agregar un componente para mostrar el gráfico circular
        dcc.Graph(id="sensitivity-analysis-chart-brutte-regular", style=graph_style),  # Gráfico de análisis de sensibilidad para Brutte Regular
        dcc.Graph(id="sensitivity-analysis-chart-chanelle-regular", style=graph_style),  # Gráfico de análisis de sensibilidad para Chanelle Regular
        dcc.Graph(id="sensitivity-analysis-chart-brutte-luxury", style=graph_style),  # Gráfico de análisis de sensibilidad para Brutte Luxury
        dcc.Graph(id="sensitivity-analysis-chart-chanelle-luxury", style=graph_style),  # Gráfico de análisis de sensibilidad para Chanelle Luxury
        html.Div(
            id="funcion-objetivo",
            style=output_style
        )
    ]
)

@callback(
    Output("output", "children"),
    Output("total-profit", "children"),
    Output("materia-prima", "children"),
    Output("horas-laboratorio", "children"),
    Output("result-table", "data"),
    Output("result-table", "columns"),
    Output("perfume-production-chart", "figure"),  # Nuevo output para el gráfico
    Output("profitability-pie-chart", "figure"),  # Nuevo output para el gráfico circular
    Output("sensitivity-analysis-chart-brutte-regular", "figure"),  # Gráfico de análisis de sensibilidad para Brutte Regular
    Output("sensitivity-analysis-chart-chanelle-regular", "figure"),  # Gráfico de análisis de sensibilidad para Chanelle Regular
    Output("sensitivity-analysis-chart-brutte-luxury", "figure"),  # Gráfico de análisis de sensibilidad para Brutte Luxury
    Output("sensitivity-analysis-chart-chanelle-luxury", "figure"),  # Gráfico de análisis de sensibilidad para Chanelle Luxury
    Output("funcion-objetivo", "children"),  # Nuevo output para la función objetivo
    Input("brutte_regular", "value"),
    Input("chanelle_regular", "value"),
    Input("brutte_luxury", "value"),
    Input("chanelle_luxury", "value")
)
def update_output(prb, pcb, pbl, pcl):
    if prb is None or pcb is None or pbl is None or pcl is None:
        return "Por favor, complete todos los valores.", "", "", "", [], [], go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), ""

    # Initialize class
    model = LpProblem("Perfumes", LpMaximize)

    # Define Decision variables
    x = LpVariable("brutte_regular", lowBound=0, cat="Integer")
    y = LpVariable("chanelle_regular", lowBound=0, cat="Integer")
    z = LpVariable("brutte_luxury", lowBound=0, cat="Integer")
    v = LpVariable("chanelle_luxury", lowBound=0, cat="Integer")
    m = LpVariable("Materia Prima", lowBound=0, cat="Integer")

    # Coefficients from user input
    prb_coef = float(prb)
    pcb_coef = float(pcb)
    pbl_coef = float(pbl)
    pcl_coef = float(pcl)

    # Define Objective function
    objective_function = prb_coef * x + pcb_coef * y + (pbl_coef - 4) * z + (pcl_coef - 4) * v - 3 * m
    model += objective_function

    # Define Constraints
    model += m <= 4000
    model += m + 3 * z + 2 * v <= 6000
    model += x + z - 3 * m <= 0
    model += x + z - 3 * m >= 0
    model += y + v - 4 * m <= 0
    model += y + v - 4 * m >= 0
    model += x >= 0
    model += y >= 0
    model += z >= 0
    model += v >= 0
    model += m >= 0

    # Solve model
    model.solve()

    if model.status != 1:
        return "No se encontró una solución óptima.", "", "", "", [], [], go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), ""

    # Obtener valores de las variables
    x_value = x.value()
    y_value = y.value()
    z_value = z.value()
    v_value = v.value()
    m_value = m.value()

    total_profitp = prb_coef * x_value + pcb_coef * y_value + (pbl_coef - 4) * z_value + (pcl_coef - 4) * v_value - 3 * m_value

    output_text = f"La cantidad de onzas de los perfumes son:"
    materia_prima_text = f"Cantidad de materia prima utilizada: {m_value} libras"
    horas_laboratorio = m_value + 3 * z_value + 2 * v_value
    horas_laboratorio_text = f"Horas de laboratorio utilizadas: {horas_laboratorio}"

    # Datos para la tabla
    table_data = [
        {"Tipo de Perfume": "Brutte Regular", "Cantidad en onz": x_value},
        {"Tipo de Perfume": "Chanelle Regular", "Cantidad en onz": y_value},
        {"Tipo de Perfume": "Brutte Luxury", "Cantidad en onz": z_value},
        {"Tipo de Perfume": "Chanelle Luxury", "Cantidad en onz": v_value}
    ]
    table_columns = [{"name": col, "id": col} for col in ["Tipo de Perfume", "Cantidad en onz"]]

    # Crear el gráfico de barras
    perfume_production = {'Brutte Regular': x_value, 'Chanelle Regular': y_value, 'Brutte Luxury': z_value, 'Chanelle Luxury': v_value}
    bar_chart = go.Figure(data=[go.Bar(x=list(perfume_production.keys()), y=list(perfume_production.values()))])
    bar_chart.update_layout(
        title="Cantidad óptima a producir de cada producto",
        xaxis_title="Tipo de Perfume",
        yaxis_title="Cantidad Producida",
        plot_bgcolor='#444',
        paper_bgcolor='#333',
        font=dict(color='#32bdbd')
    )

    # Crear el gráfico circular
    total_revenue = prb_coef * x_value + pcb_coef * y_value + pbl_coef * z_value + pcl_coef * v_value
    brutte_regular_percentage = (prb_coef * x_value / total_revenue) * 100
    chanelle_regular_percentage = (pcb_coef * y_value / total_revenue) * 100
    brutte_luxury_percentage = (pbl_coef * z_value / total_revenue) * 100
    chanelle_luxury_percentage = (pcl_coef * v_value / total_revenue) * 100

    pie_chart = go.Figure(data=[go.Pie(labels=['Brutte Regular', 'Chanelle Regular', 'Brutte Luxury', 'Chanelle Luxury'],
                                        values=[brutte_regular_percentage, chanelle_regular_percentage, brutte_luxury_percentage, chanelle_luxury_percentage],
                                        textinfo='label+percent')])
    pie_chart.update_layout(
        title="Porcentaje de contribución a los ingresos",
        plot_bgcolor='#444',
        paper_bgcolor='#333',
        font=dict(color='#32bdbd')
    )

    # Mostrar la función objetivo
    funcion_objetivo_text = f"Función Objetivo: {prb_coef}*Brutte Regular + {pcb_coef}*Chanelle Regular + {pbl_coef}-4*Brutte Luxury + {pcl_coef}-4*Chanelle Luxury - 3*Materia Prima" 

    # Análisis de Sensibilidad
    sensitivity_results_brutte_regular = []
    sensitivity_results_chanelle_regular = []
    sensitivity_results_brutte_luxury = []
    sensitivity_results_chanelle_luxury = []

    # Definir un rango de valores para cada precio
    price_range_brutte_regular = np.linspace(0.5 * prb_coef, 1.5 * prb_coef, 10)  # Ajustar los límites y la cantidad de puntos en el rango
    price_range_chanelle_regular = np.linspace(0.5 * pcb_coef, 1.5 * pcb_coef, 10)
    price_range_brutte_luxury = np.linspace(0.5 * pbl_coef, 1.5 * pbl_coef, 10)
    price_range_chanelle_luxury = np.linspace(0.5 * pcl_coef, 1.5 * pcl_coef, 10)

    for price in price_range_brutte_regular:
        # Crear el modelo de optimización con el nuevo precio
        model = LpProblem("Perfumes", LpMaximize)
        x = LpVariable("brutte_regular", lowBound=0, cat="Integer")
        y = LpVariable("chanelle_regular", lowBound=0, cat="Integer")
        z = LpVariable("brutte_luxury", lowBound=0, cat="Integer")
        v = LpVariable("chanelle_luxury", lowBound=0, cat="Integer")
        m = LpVariable("Materia Prima", lowBound=0, cat="Integer")

        objective_function = price * x + pcb_coef * y + (pbl_coef - 4) * z + (pcl_coef - 4) * v - 3 * m
        model += objective_function

        model += m <= 4000
        model += m + 3 * z + 2 * v <= 6000
        model += x + z - 3 * m <= 0
        model += x + z - 3 * m >= 0
        model += y + v - 4 * m <= 0
        model += y + v - 4 * m >= 0
        model += x >= 0
        model += y >= 0
        model += z >= 0
        model += v >= 0
        model += m >= 0

        model.solve()

        if model.status == 1:
            x_value = x.value()
            y_value = y.value()
            z_value = z.value()
            v_value = v.value()
            m_value = m.value()

            total_profitbr = price * x_value + pcb_coef * y_value + (pbl_coef - 4) * z_value + (pcl_coef - 4) * v_value - 3 * m_value

            sensitivity_results_brutte_regular.append({
                'Precio Brutte Regular': price,
                'Ganancia Total': total_profitbr,
                'Cantidad Brutte Regular': x_value,
                'Cantidad Chanelle Regular': y_value,
                'Cantidad Brutte Luxury': z_value,
                'Cantidad Chanelle Luxury': v_value,
                'Materia Prima': m_value
            })

    for price in price_range_chanelle_regular:
        model = LpProblem("Perfumes", LpMaximize)
        x = LpVariable("brutte_regular", lowBound=0, cat="Integer")
        y = LpVariable("chanelle_regular", lowBound=0, cat="Integer")
        z = LpVariable("brutte_luxury", lowBound=0, cat="Integer")
        v = LpVariable("chanelle_luxury", lowBound=0, cat="Integer")
        m = LpVariable("Materia Prima", lowBound=0, cat="Integer")

        objective_function = prb_coef * x + price * y + (pbl_coef - 4) * z + (pcl_coef - 4) * v - 3 * m
        model += objective_function

        model += m <= 4000
        model += m + 3 * z + 2 * v <= 6000
        model += x + z - 3 * m <= 0
        model += x + z - 3 * m >= 0
        model += y + v - 4 * m <= 0
        model += y + v - 4 * m >= 0
        model += x >= 0
        model += y >= 0
        model += z >= 0
        model += v >= 0
        model += m >= 0

        model.solve()

        if model.status == 1:
            x_value = x.value()
            y_value = y.value()
            z_value = z.value()
            v_value = v.value()
            m_value = m.value()

            total_profitcr = prb_coef * x_value + price * y_value + (pbl_coef - 4) * z_value + (pcl_coef - 4) * v_value - 3 * m_value

            sensitivity_results_chanelle_regular.append({
                'Precio Chanelle Regular': price,
                'Ganancia Total': total_profitcr,
                'Cantidad Brutte Regular': x_value,
                'Cantidad Chanelle Regular': y_value,
                'Cantidad Brutte Luxury': z_value,
                'Cantidad Chanelle Luxury': v_value,
                'Materia Prima': m_value
            })

    for price in price_range_brutte_luxury:
        model = LpProblem("Perfumes", LpMaximize)
        x = LpVariable("brutte_regular", lowBound=0, cat="Integer")
        y = LpVariable("chanelle_regular", lowBound=0, cat="Integer")
        z = LpVariable("brutte_luxury", lowBound=0, cat="Integer")
        v = LpVariable("chanelle_luxury", lowBound=0, cat="Integer")
        m = LpVariable("Materia Prima", lowBound=0, cat="Integer")

        objective_function = prb_coef * x + pcb_coef * y + (price - 4) * z + (pcl_coef - 4) * v - 3 * m
        model += objective_function

        model += m <= 4000
        model += m + 3 * z + 2 * v <= 6000
        model += x + z - 3 * m <= 0
        model += x + z - 3 * m >= 0
        model += y + v - 4 * m <= 0
        model += y + v - 4 * m >= 0
        model += x >= 0
        model += y >= 0
        model += z >= 0
        model += v >= 0
        model += m >= 0

        model.solve()

        if model.status == 1:
            x_value = x.value()
            y_value = y.value()
            z_value = z.value()
            v_value = v.value()
            m_value = m.value()

            total_profitbl = prb_coef * x_value + pcb_coef * y_value + (price - 4) * z_value + (pcl_coef - 4) * v_value - 3 * m_value

            sensitivity_results_brutte_luxury.append({
                'Precio Brutte Luxury': price,
                'Ganancia Total': total_profitbl,
                'Cantidad Brutte Regular': x_value,
                'Cantidad Chanelle Regular': y_value,
                'Cantidad Brutte Luxury': z_value,
                'Cantidad Chanelle Luxury': v_value,
                'Materia Prima': m_value
            })

    for price in price_range_chanelle_luxury:
        model = LpProblem("Perfumes", LpMaximize)
        x = LpVariable("brutte_regular", lowBound=0, cat="Integer")
        y = LpVariable("chanelle_regular", lowBound=0, cat="Integer")
        z = LpVariable("brutte_luxury", lowBound=0, cat="Integer")
        v = LpVariable("chanelle_luxury", lowBound=0, cat="Integer")
        m = LpVariable("Materia Prima", lowBound=0, cat="Integer")

        objective_function = prb_coef * x + pcb_coef * y + (pbl_coef - 4) * z + (price - 4) * v - 3 * m
        model += objective_function

        model += m <= 4000
        model += m + 3 * z + 2 * v <= 6000
        model += x + z - 3 * m <= 0
        model += x + z - 3 * m >= 0
        model += y + v - 4 * m <= 0
        model += y + v - 4 * m >= 0
        model += x >= 0
        model += y >= 0
        model += z >= 0
        model += v >= 0
        model += m >= 0

        model.solve()

        if model.status == 1:
            x_value = x.value()
            y_value = y.value()
            z_value = z.value()
            v_value = v.value()
            m_value = m.value()

            total_profitcl = prb_coef * x_value + pcb_coef * y_value + (pbl_coef - 4) * z_value + (price - 4) * v_value - 3 * m_value

            sensitivity_results_chanelle_luxury.append({
                'Precio Chanelle Luxury': price,
                'Ganancia Total': total_profitcl,
                'Cantidad Brutte Regular': x_value,
                'Cantidad Chanelle Regular': y_value,
                'Cantidad Brutte Luxury': z_value,
                'Cantidad Chanelle Luxury': v_value,
                'Materia Prima': m_value
            })

    # Crear los gráficos de análisis de sensibilidad
    sensitivity_chart_brutte_regular = go.Figure(data=[
        go.Scatter(
            x=[result['Precio Brutte Regular'] for result in sensitivity_results_brutte_regular],
            y=[result['Ganancia Total'] for result in sensitivity_results_brutte_regular],
            mode='lines+markers',
            name='Ganancia Total'
        )
    ])
    sensitivity_chart_brutte_regular.update_layout(
        title="Análisis de Sensibilidad - Brutte Regular",
        xaxis_title="Precio Brutte Regular",
        yaxis_title="Ganancia Total",
        plot_bgcolor='#444',
        paper_bgcolor='#333',
        font=dict(color='#32bdbd')
    )

    sensitivity_chart_chanelle_regular = go.Figure(data=[
        go.Scatter(
            x=[result['Precio Chanelle Regular'] for result in sensitivity_results_chanelle_regular],
            y=[result['Ganancia Total'] for result in sensitivity_results_chanelle_regular],
            mode='lines+markers',
            name='Ganancia Total'
        )
    ])
    sensitivity_chart_chanelle_regular.update_layout(
        title="Análisis de Sensibilidad - Chanelle Regular",
        xaxis_title="Precio Chanelle Regular",
        yaxis_title="Ganancia Total",
        plot_bgcolor='#444',
        paper_bgcolor='#333',
        font=dict(color='#32bdbd')
    )

    sensitivity_chart_brutte_luxury = go.Figure(data=[
        go.Scatter(
            x=[result['Precio Brutte Luxury'] for result in sensitivity_results_brutte_luxury],
            y=[result['Ganancia Total'] for result in sensitivity_results_brutte_luxury],
            mode='lines+markers',
            name='Ganancia Total'
        )
    ])
    sensitivity_chart_brutte_luxury.update_layout(
        title="Análisis de Sensibilidad - Brutte Luxury",
        xaxis_title="Precio Brutte Luxury",
        yaxis_title="Ganancia Total",
        plot_bgcolor='#444',
        paper_bgcolor='#333',
        font=dict(color='#32bdbd')
    )

    sensitivity_chart_chanelle_luxury = go.Figure(data=[
        go.Scatter(
            x=[result['Precio Chanelle Luxury'] for result in sensitivity_results_chanelle_luxury],
            y=[result['Ganancia Total'] for result in sensitivity_results_chanelle_luxury],
            mode='lines+markers',
            name='Ganancia Total'
        )
    ])
    sensitivity_chart_chanelle_luxury.update_layout(
        title="Análisis de Sensibilidad - Chanelle Luxury",
        xaxis_title="Precio Chanelle Luxury",
        yaxis_title="Ganancia Total",
        plot_bgcolor='#444',
        paper_bgcolor='#333',
        font=dict(color='#32bdbd')
    )

    return (output_text, f"Ganancia Total: {total_profitp}", materia_prima_text, horas_laboratorio_text,
            table_data, table_columns, bar_chart, pie_chart, sensitivity_chart_brutte_regular, 
            sensitivity_chart_chanelle_regular, sensitivity_chart_brutte_luxury, sensitivity_chart_chanelle_luxury,
            funcion_objetivo_text)

if __name__ == "__main__":
    app.run_server(debug=True, host='0.0.0.0', port=9000)
    #app.run_server(debug=True)
    