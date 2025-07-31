import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots

FONT_FAMILY = "Palatino"

def plot_function_with_tangent_line(f, f_prime, x_range=(-10, 10), y_range=(-10, 10), title=None, font=None, xaxis_title='x', yaxis_title='y', show_axis_labels=None):
    """
    Plot function in 2D space.
    
    Args:
        f: function to plot
        f_prime: derivative of function to plot
        x_range: range of x-axis
        y_range: range of y-axis
        title: Optional title for the plot
        font: Optional font dictionary
        xaxis_title: Title for x-axis
        yaxis_title: Title for y-axis
        show_axis_labels: Whether to show axis labels
    
    Returns:
        plotly.graph_objects.Figure: The plotly figure object
    """

    fig = go.Figure()
    x = np.linspace(x_range[0], x_range[1], 100)
    y = f(x)
    y_prime = f_prime(x)
    x_slider=list(range(x_range[0], x_range[1] + 1))

    # Plot function
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='f(x)'))

    # Placeholder for tangent line (will be updated)
    tangent_line = go.Scatter(
        x=[], y=[],
        mode='lines',
        line=dict(color='red'),
        name='Tangent line'
    )
    fig.add_trace(tangent_line)

    tangent_point = go.Scatter(
        x=[], y=[],
        mode='markers',
        marker=dict(color='red', size=7),
        name='Tangent point'
    )
    fig.add_trace(tangent_point)
    
    slider_steps = []
    for x_point in x_slider:
        tan_slope = f_prime(x_point)
        y_point = f(x_point)
        x_tangent = np.array([x_point-3, x_point+3])
        y_tangent = tan_slope * (x_tangent - x_point) + y_point

        step = dict(
            method='update',
            args=[{
                'x': [x, x_tangent, [x_point]],
                'y': [y, y_tangent, [y_point]]
            }, {
                'title': f"Tangent at x = {x_point:.2f}"
            }],
        )
        slider_steps.append(step)

    # Slider layout
    sliders = [dict(
        active=0,
        steps=slider_steps,
        currentvalue={"prefix": "x = "},
        pad={"t": 50}
    )]

    layout_dict = dict(
        xaxis=dict(
            title=xaxis_title if show_axis_labels else '',
            range=x_range,
            gridcolor='#f0f0f0',
            zerolinecolor='gray',
            tickfont=dict(size=10)
        ),
        yaxis=dict(
            title=yaxis_title if show_axis_labels else '',
            range=y_range,
            gridcolor='#f0f0f0',
            zerolinecolor='gray',
            tickfont=dict(size=10)
        ),
        sliders=sliders,
        width=600,
        height=600,
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=font,
        margin=dict(l=50, r=50, t=50, b=50),
        showlegend=True
    )

    if title is not None:
        layout_dict['title'] = title

    fig.update_layout(layout_dict)
    return fig

def plot_functions(f_list, f_titles, crit_x=None, x_range=(-10, 10), y_range=(-10, 10), title=None, font=None, xaxis_title='x', yaxis_title='y', show_axis_labels=None):
    """
    Plot function in 2D space.
    
    Args:
        f: function to plot
        f_prime: derivative of function to plot
        crit_x: critical points of function to plot
        x_range: range of x-axis
        y_range: range of y-axis
        title: Optional title for the plot
        font: Optional font dictionary
        xaxis_title: Title for x-axis
        yaxis_title: Title for y-axis
        show_axis_labels: Whether to show axis labels
    
    Returns:
        plotly.graph_objects.Figure: The plotly figure object
    """
    fig = go.Figure()
    x = np.linspace(x_range[0], x_range[1], 100)
    for f, f_title in zip(f_list, f_titles):
        y = f(x)
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=f_title))
    if crit_x is not None:
        crit_y = [f_list[0](x) for x in crit_x]
        fig.add_trace(go.Scatter(x=crit_x, y=crit_y, mode='markers', marker=dict(color='blue', size=7), name='Critical Points'))
        fig.add_trace(go.Scatter(x=crit_x, y=[0 for _ in crit_x], mode='markers', marker=dict(color='red', size=7), name='Derivative = 0'))

    layout_dict = dict(
        xaxis=dict(
            title=xaxis_title if show_axis_labels else '',
            range=x_range,
            gridcolor='#f0f0f0',
            zerolinecolor='gray',
            tickfont=dict(size=10)
        ),
        yaxis=dict(
            title=yaxis_title if show_axis_labels else '',
            range=y_range,
            gridcolor='#f0f0f0',
            zerolinecolor='gray',
            tickfont=dict(size=10)
        ),
        width=600,
        height=600,
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=font,
        margin=dict(l=50, r=50, t=50, b=50),
        showlegend=True
    )

    if title is not None:
        layout_dict['title'] = title


    fig.update_layout(layout_dict)
    return fig



def plot_function_with_derivative_and_crits(f, f_prime, crit_x, x_range=(-10, 10), y_range=(-10, 10), title=None, font=None, xaxis_title='x', yaxis_title='y', show_axis_labels=None):
    """
    Plot function in 2D space.
    
    Args:
        f: function to plot
        f_prime: derivative of function to plot
        crit_x: critical points of function to plot
        x_range: range of x-axis
        y_range: range of y-axis
        title: Optional title for the plot
        font: Optional font dictionary
        xaxis_title: Title for x-axis
        yaxis_title: Title for y-axis
        show_axis_labels: Whether to show axis labels
    
    Returns:
        plotly.graph_objects.Figure: The plotly figure object
    """
    fig = go.Figure()
    x = np.linspace(x_range[0], x_range[1], 100)
    y = f(x)
    y_prime = f_prime(x)
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='f(x)'))
    fig.add_trace(go.Scatter(x=x, y=y_prime, mode='lines', name='f\'(x)'))
    crit_y = [f(x) for x in crit_x]
    fig.add_trace(go.Scatter(x=crit_x, y=crit_y, mode='markers', marker=dict(color='blue', size=7), name='Critical Points'))
    fig.add_trace(go.Scatter(x=crit_x, y=[0 for _ in crit_x], mode='markers', marker=dict(color='red', size=7), name='Derivative = 0'))

    layout_dict = dict(
        xaxis=dict(
            title=xaxis_title if show_axis_labels else '',
            range=x_range,
            gridcolor='#f0f0f0',
            zerolinecolor='gray',
            tickfont=dict(size=10)
        ),
        yaxis=dict(
            title=yaxis_title if show_axis_labels else '',
            range=y_range,
            gridcolor='#f0f0f0',
            zerolinecolor='gray',
            tickfont=dict(size=10)
        ),
        width=600,
        height=600,
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=font,
        margin=dict(l=50, r=50, t=50, b=50),
        showlegend=True
    )

    if title is not None:
        layout_dict['title'] = title


    fig.update_layout(layout_dict)
    return fig

def plot_functions_grid(f_list, f_titles, rows=2, cols=2, x_range=(-10, 10), y_range=(-10, 10), title=None, font=None, xaxis_title='x', yaxis_title='y', show_axis_labels=None):
    """
    Plot 2D functions in a grid.
    
    Args:
        f_list: list of functions to plot
        f_titles: list of titles for each function
        x_range: range of x-axis
        y_range: range of y-axis
        title: Optional title for the plot
        font: Optional font dictionary
        xaxis_title: Title for x-axis
        yaxis_title: Title for y-axis
        show_axis_labels: Whether to show axis labels
    
    Returns:
        plotly.graph_objects.Figure: The plotly figure object
    """
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=f_titles)
    x = np.linspace(x_range[0], x_range[1], 50)
    
    for r in range(rows):   
        for c in range(cols):
            f = f_list[r*cols + c]
            y = f(x)
            fig.add_trace(
                go.Scatter(x=x, y=y, mode='lines', name=f_titles[r*cols + c]),
                row=r + 1, col=c + 1)

    layout_dict = dict(
        width=600,
        height=600,
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=font,
        margin=dict(l=50, r=50, t=50, b=50),
        showlegend=True
    )

    for i in range(1, rows * cols + 1):
        axis_suffix = "" if i == 1 else str(i)
        layout_dict[f'xaxis{axis_suffix}'] = dict(
            title=xaxis_title if show_axis_labels else '',
            range=x_range,
            gridcolor='#f0f0f0',
            zerolinecolor='gray',
            tickfont=dict(size=10)
        )
        layout_dict[f'yaxis{axis_suffix}'] = dict(
            title=yaxis_title if show_axis_labels else '',
            range=y_range,
            gridcolor='#f0f0f0',
            zerolinecolor='gray',
            tickfont=dict(size=10)
        )
    

    if title is not None:
        layout_dict['title'] = title

    fig.update_layout(**layout_dict)
    return fig
