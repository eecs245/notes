import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

FONT_FAMILY = "Palatino"

def plot_function_with_tangent_line(f, f_prime, x_range=(-10, 10), y_range=(-10, 10), title=None, font=None, xaxis_title='x', yaxis_title='y', show_axis_labels=None, initial_x_point=-2, dtick=1):
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
    x_slider=list(np.arange(x_range[0], x_range[1] + 1, dtick))

    # Plot function
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='f(x)', line=dict(color='#3d81f6')))

    # Initialize tangent line and point with x = -2 (default position)
    initial_tan_slope = f_prime(initial_x_point)
    initial_y_point = f(initial_x_point)
    
    # Calculate tangent line with constant length
    tangent_length = 3  # Half-length of tangent line
    dx = tangent_length / np.sqrt(1 + initial_tan_slope**2)
    initial_x_tangent = np.array([initial_x_point - dx, initial_x_point + dx])
    initial_y_tangent = initial_tan_slope * (initial_x_tangent - initial_x_point) + initial_y_point

    tangent_line = go.Scatter(
        x=initial_x_tangent,
        y=initial_y_tangent,
        mode='lines',
        line=dict(color='orange'),
        name='Tangent line'
    )
    fig.add_trace(tangent_line)

    tangent_point = go.Scatter(
        x=[initial_x_point],
        y=[initial_y_point],
        mode='markers',
        marker=dict(color='orange', size=7),
        name='Tangent point'
    )
    fig.add_trace(tangent_point)
    
    slider_steps = []
    for i, x_point in enumerate(x_slider):
        tan_slope = f_prime(x_point)
        y_point = f(x_point)
        
        # Calculate tangent line with constant length
        tangent_length = 3  # Half-length of tangent line
        dx = tangent_length / np.sqrt(1 + tan_slope**2)
        x_tangent = np.array([x_point - dx, x_point + dx])
        y_tangent = tan_slope * (x_tangent - x_point) + y_point

        # Use original title if provided, otherwise show tangent info
        if title is not None:
            step_title = title
        else:
            step_title = f"Tangent line at x = {x_point}; slope = {np.round(tan_slope, 4)}"
        
        step = dict(
            method='update',
            args=[{
                'x': [x, x_tangent, [x_point]],
                'y': [y, y_tangent, [y_point]]
            }, {
                'title': {'text': step_title}
            }],
            label=f"x = {x_point}"
        )
        slider_steps.append(step)

    # Slider layout
    # Find index for x = -2 (default position)
    default_index = x_slider.index(initial_x_point)
    sliders = [dict(
        active=default_index,
        steps=slider_steps,
        currentvalue={"prefix": ""},
        pad={"t": 50}
    )]

    layout_dict = dict(
        xaxis=dict(
            title=xaxis_title if show_axis_labels else '',
            range=x_range,
            gridcolor='#f0f0f0',
            zerolinecolor='gray',
            tickfont=dict(size=10, family=FONT_FAMILY),
            title_font=dict(family=FONT_FAMILY)
        ),
        yaxis=dict(
            title=yaxis_title if show_axis_labels else '',
            range=y_range,
            gridcolor='#f0f0f0',
            zerolinecolor='gray',
            tickfont=dict(size=10, family=FONT_FAMILY),
            title_font=dict(family=FONT_FAMILY)
        ),
        sliders=sliders,
        width=600,
        height=600,
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(family=FONT_FAMILY) if font is None else font,
        margin=dict(l=50, r=50, t=50, b=50),
        showlegend=False
    )

    # Set initial title to match the default slider position
    if title is not None:
        layout_dict['title'] = title
    else:
        layout_dict['title'] = f"Tangent line at x = {initial_x_point}; slope = {np.round(initial_tan_slope, 4)}"

    fig.update_layout(layout_dict)
    return fig

def make_3D_surface(f, lim, xaxis_title, yaxis_title, zaxis_title, title):
    x1_range = np.linspace(-lim, lim, 100)
    x2_range = np.linspace(-lim, lim, 100)
    x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
    z_values = f(x1_grid, x2_grid)

    fig = go.Figure(data=[
        go.Surface(
            x=x1_grid,
            y=x2_grid,
            z=z_values,
            colorscale='RdBu_r',
            showscale=False,
        )
    ])
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(
                title=xaxis_title, gridcolor="#f0f0f0",
                showbackground=True, showline=True, linecolor="black", linewidth=1,
                tickfont=dict(family='Palatino', size=10),
                backgroundcolor="white"
            ),
            yaxis=dict(
                title=yaxis_title, gridcolor="#f0f0f0",
                showbackground=True, showline=True, linecolor="black", linewidth=1,
                tickfont=dict(family='Palatino', size=10),
                backgroundcolor="white"
            ),
            zaxis=dict(
                title=zaxis_title, gridcolor="#f0f0f0",
                showbackground=True, showline=True, linecolor="black", linewidth=1,
                tickfont=dict(family='Palatino', size=10),
                backgroundcolor="white"
            ),
            aspectratio=dict(x=1, y=1, z=1),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1))
        ),
        width=800,
        height=700,
        margin=dict(l=65, r=50, b=65, t=90),
        font=dict(family='Palatino', size=16, color="#222"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
    )
    return fig


def make_3D_contour(
    f,
    dfx1=None,
    dfx2=None,
    lim=2,
    xaxis_title="x1",
    yaxis_title="x2",
    title=None,
    with_gradient=False,
    grad_point=None,
    neg=False,
    contour_kwargs=None,
    norm_grad=False
):
    import numpy as np
    import plotly.graph_objects as go

    x1_range = np.linspace(-lim, lim, 100)
    x2_range = np.linspace(-lim, lim, 100)
    x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
    z_values = f(x1_grid, x2_grid)
    if contour_kwargs is None:
        contour_kwargs = {}

    base_contour = dict(
        z=z_values,
        x=x1_range,
        y=x2_range,
        colorscale='RdBu_r',
        showscale=False,
        contours=dict(
            showlabels=True,
            labelfont=dict(size=12, color='white', family='Palatino'),
        ),
    )
    base_contour.update(contour_kwargs)
    fig = go.Figure(data=[
        go.Contour(**base_contour)
    ])
    if with_gradient and dfx1 is not None and dfx2 is not None and grad_point is not None:
        x1_start, x2_start = grad_point
        # Use actual gradient magnitude for visual length, scaled by a factor
        grad_x, grad_y = dfx1(x1_start, x2_start), dfx2(x1_start, x2_start)
        grad_magnitude = (grad_x**2 + grad_y**2) ** 0.5
        # Scale factor to control overall arrow size
        if norm_grad:
            scale_factor = 0.2
        else:
            scale_factor = 1
        if grad_magnitude > 0:
            # Scale the actual gradient components by scale_factor
            scaled_grad_x = grad_x * scale_factor
            scaled_grad_y = grad_y * scale_factor
        else:
            scaled_grad_x, scaled_grad_y = 0, 0

        if neg:
            x1_end, x2_end = x1_start - scaled_grad_x, x2_start - scaled_grad_y
        else:
            x1_end, x2_end = x1_start + scaled_grad_x, x2_start + scaled_grad_y

        fig.add_trace(go.Scatter(
            x=[x1_start, x1_end], y=[x2_start, x2_end],
            mode='lines+markers',
            line=dict(color='red' if neg else 'gold', width=5),
            showlegend=False,
        ))

        # Adjust the arrow's tail so the arrowhead is at exactly the end of the gradient vector.
        # The arrow annotation should have (ax, ay) at the real 'start' and (x, y) at the tip.
        if neg:
            # Arrow from start (x1_start, x2_start) to tip (x1_end, x2_end), negative gradient
            annotation_x = x1_end
            annotation_y = x2_end
            annotation_ax = x1_start
            annotation_ay = x2_start
        else:
            # Arrow from start (x1_start, x2_start) to tip (x1_end, x2_end), positive gradient
            annotation_x = x1_end
            annotation_y = x2_end
            annotation_ax = x1_start
            annotation_ay = x2_start

        fig.add_annotation(
            x=annotation_x, y=annotation_y,
            ax=annotation_ax, ay=annotation_ay,
            xref='x', yref='y', axref='x', ayref='y',
            showarrow=True, arrowhead=5, arrowsize=1,
            arrowwidth=3, arrowcolor='red' if neg else 'gold'
        )
    if with_gradient and grad_point is not None:
        col = 'red' if neg else 'gold'
        text = 'Negative of the Gradient Vector' if neg else 'Gradient Vector'
        final_title = f'<span style="color:{col}"><b>{text}</b></span> at Point ({grad_point[0]}, {grad_point[1]})'
    else:
        final_title = title or ''
    fig.update_layout(
        title=final_title,
        xaxis=dict(
            title=xaxis_title, gridcolor="#f0f0f0",
            showline=True, linecolor="black", linewidth=1,
            tickfont=dict(family='Palatino', size=10)
        ),
        yaxis=dict(
            title=yaxis_title, gridcolor="#f0f0f0",
            showline=True, linecolor="black", linewidth=1,
            tickfont=dict(family='Palatino', size=10)
        ),
        width=800,
        height=700,
        margin=dict(l=65, r=50, b=65, t=90),
        showlegend=True,
        font=dict(family='Palatino', size=16, color="#222"),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def show_surface_and_contour_side_by_side(
    f,
    dfx1=None,
    dfx2=None,
    lim=2,
    xaxis_title="x₁",
    yaxis_title="x₂",
    zaxis_title="f(x₁, x₂)",
    surface_title="3D Surface",
    contour_title="Contour",
    with_gradient=False,
    grad_point=None,
    neg=False,
    contour_kwargs=None,
    width=800,
    height=1200,
):
    """
    Display both 3D surface and 2D contour stacked vertically (2 rows, 1 column) with shared stylings.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    surface_fig = make_3D_surface(
        f=f,
        lim=lim,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        zaxis_title=zaxis_title,
        title=surface_title,
    )
    contour_fig = make_3D_contour(
        f=f,
        dfx1=dfx1, dfx2=dfx2, lim=lim,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        title=contour_title,
        with_gradient=with_gradient,
        grad_point=grad_point,
        neg=neg,
        contour_kwargs=contour_kwargs,
    )
    subplot_fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(surface_title, contour_title),
        specs=[[{'type': 'surface'}], [{'type': 'xy'}]],
        vertical_spacing=0.13
    )
    # Transfer surface traces (3D plot)
    for trace in surface_fig.data:
        subplot_fig.add_trace(trace, row=1, col=1)
    for trace in contour_fig.data:
        subplot_fig.add_trace(trace, row=2, col=1)

    # Transfer contour annotations (gradient arrow)
    if hasattr(contour_fig, "layout") and "annotations" in contour_fig.layout:
        subplot_fig.layout.annotations += tuple(contour_fig.layout.annotations)

    # Layout: shared formatting
    subplot_fig.update_layout(
        height=height,
        width=width,
        margin=dict(l=40, r=40, b=60, t=100),
        font=dict(family='Palatino', size=16, color="#222"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
    )

    subplot_fig.update_scenes(
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        zaxis_title=zaxis_title,
        xaxis=dict(
            gridcolor="#f0f0f0", showbackground=True,
            showline=True, linecolor="black", linewidth=1,
            tickfont=dict(family='Palatino', size=10),
            backgroundcolor="white"
        ),
        yaxis=dict(
            gridcolor="#f0f0f0", showbackground=True,
            showline=True, linecolor="black", linewidth=1,
            tickfont=dict(family='Palatino', size=10),
            backgroundcolor="white"
        ),
        zaxis=dict(
            gridcolor="#f0f0f0", showbackground=True,
            showline=True, linecolor="black", linewidth=1,
            tickfont=dict(family='Palatino', size=10),
            backgroundcolor="white"
        ),
        aspectratio=dict(x=1, y=1, z=1),
        camera=dict(eye=dict(x=1.5, y=1.5, z=1))
    )
    subplot_fig.update_xaxes(
        title_text=xaxis_title, row=2, col=1,
        showline=True, linecolor="black", linewidth=1,
        showgrid=True, gridcolor="#f0f0f0",
        tickfont=dict(family='Palatino', size=10)
    )
    subplot_fig.update_yaxes(
        title_text=yaxis_title, row=2, col=1,
        showline=True, linecolor="black", linewidth=1,
        showgrid=True, gridcolor="#f0f0f0",
        tickfont=dict(family='Palatino', size=10)
    )

    return subplot_fig

def _gd_example_function(x1, x2):
    return 3 * np.sin(2 * x1) * np.cos(2 * x2) + x1**2 + x2**2


def _gd_example_gradients(x1, x2):
    grad_x = 6 * np.cos(2 * x1) * np.cos(2 * x2) + 2 * x1
    grad_y = -6 * np.sin(2 * x1) * np.sin(2 * x2) + 2 * x2
    return grad_x, grad_y


def _gd_example_grid(lim=2, resolution=100):
    x1_range = np.linspace(-lim, lim, resolution)
    x2_range = np.linspace(-lim, lim, resolution)
    x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
    z_values = _gd_example_function(x1_grid, x2_grid)
    return x1_range, x2_range, x1_grid, x2_grid, z_values


def _compute_gd_example_path(x1_start, x2_start, step_size, iterations):
    x1, x2 = x1_start, x2_start
    path_x, path_y = [x1], [x2]
    path_z = [_gd_example_function(x1, x2)]

    for _ in range(iterations):
        grad_x, grad_y = _gd_example_gradients(x1, x2)
        x1 = x1 - step_size * grad_x
        x2 = x2 - step_size * grad_y
        path_x.append(x1)
        path_y.append(x2)
        path_z.append(_gd_example_function(x1, x2))

    return path_x, path_y, path_z


def _gd_path_title(x1_start, x2_start, iteration=None, step_size=None):
    base_title = f'<b><span style="color:gold">Gradient Descent Path</span></b> from ({x1_start}, {x2_start})'
    if step_size is not None:
        base_title += f', Step Size = {step_size}'
    if iteration is None:
        return base_title
    return f'{base_title}<br><span style="font-size: 0.85em;">Iteration {iteration}</span>'


def show_gd_path_contour(x1_start=-0.5, x2_start=1, step_size=0.1, iterations=10):
    x1_range, x2_range, _, _, z_values = _gd_example_grid()

    # Initialize figure
    fig = go.Figure(data=[
        go.Contour(
            z=z_values, x=x1_range, y=x2_range,
            colorscale='RdBu_r',
            contours=dict(showlabels=True, labelfont=dict(size=12, color='white'))
        )
    ])

    path_x, path_y, _ = _compute_gd_example_path(x1_start, x2_start, step_size, iterations)

    # Add descent path to the plot
    fig.add_trace(go.Scatter(
        x=path_x, y=path_y, mode='lines+markers',
        line=dict(color='gold', width=3),
        marker=dict(size=8, color='gold'),
    ))

    title = _gd_path_title(x1_start, x2_start, step_size=step_size)

    # Update layout with Palatino font
    fig.update_layout(
        title=title, 
        xaxis_title='x1', 
        yaxis_title='x2',
        width=800, 
        height=700, 
        margin=dict(l=65, r=50, b=65, t=90),
        font=dict(family='Palatino'),
        xaxis=dict(
            tickfont=dict(family='Palatino', size=10),
            title_font=dict(family='Palatino')
        ),
        yaxis=dict(
            tickfont=dict(family='Palatino', size=10),
            title_font=dict(family='Palatino')
        ),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    return fig


def show_gd_path_contour_slider(x1_start=-0.5, x2_start=1, step_size=0.1, iterations=10):
    x1_range, x2_range, _, _, z_values = _gd_example_grid()
    path_x, path_y, _ = _compute_gd_example_path(x1_start, x2_start, step_size, iterations)

    fig = go.Figure(
        data=[
            go.Contour(
                z=z_values,
                x=x1_range,
                y=x2_range,
                colorscale='RdBu_r',
                showscale=False,
                contours=dict(showlabels=True, labelfont=dict(size=12, color='white')),
            ),
            go.Scatter(
                x=[path_x[0]],
                y=[path_y[0]],
                mode='lines+markers',
                line=dict(color='gold', width=3),
                marker=dict(size=8, color='gold'),
                showlegend=False,
            ),
            go.Scatter(
                x=[path_x[0]],
                y=[path_y[0]],
                mode='markers',
                marker=dict(color='orange', size=12),
                showlegend=False,
            ),
        ]
    )

    slider_steps = []
    for iteration_idx in range(len(path_x)):
        slider_steps.append(
            dict(
                method='update',
                args=[
                    {
                        'x': [path_x[:iteration_idx + 1], [path_x[iteration_idx]]],
                        'y': [path_y[:iteration_idx + 1], [path_y[iteration_idx]]],
                    },
                    {
                        'title': _gd_path_title(x1_start, x2_start, iteration=iteration_idx, step_size=step_size),
                    },
                    [1, 2],
                ],
                label=str(iteration_idx),
            )
        )

    fig.update_layout(
        title=_gd_path_title(x1_start, x2_start, iteration=0, step_size=step_size),
        xaxis_title='x1',
        yaxis_title='x2',
        width=800,
        height=700,
        margin=dict(l=65, r=50, b=65, t=90),
        font=dict(family='Palatino'),
        xaxis=dict(
            tickfont=dict(family='Palatino', size=10),
            title_font=dict(family='Palatino'),
        ),
        yaxis=dict(
            tickfont=dict(family='Palatino', size=10),
            title_font=dict(family='Palatino'),
        ),
        sliders=[
            dict(
                active=0,
                currentvalue={'prefix': 'Iteration: '},
                pad={'t': 35},
                steps=slider_steps,
            )
        ],
        paper_bgcolor='white',
        plot_bgcolor='white',
        showlegend=False,
    )

    return fig


def show_gd_path_surface(x1_start=-0.5, x2_start=1, step_size=0.1, iterations=10):
    x1_range, x2_range, x1_grid, x2_grid, z_values = _gd_example_grid()

    # Initialize figure
    fig = go.Figure(data=[
        go.Surface(
            x=x1_grid,
            y=x2_grid,
            z=z_values,
            colorscale='RdBu_r',
            contours=dict(
                z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project=dict(z=True))
            )
        )
    ])

    path_x, path_y, path_z = _compute_gd_example_path(x1_start, x2_start, step_size, iterations)
    
    # Add descent path to the plot
    fig.add_trace(go.Scatter3d(
        x=path_x, y=path_y, z=path_z, mode='lines+markers',
        line=dict(color='gold', width=2),
        marker=dict(size=8, color='gold')
    ))

    title = _gd_path_title(x1_start, x2_start, step_size=step_size)

    # Update layout with Palatino font, white background, and custom gridlines
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='x1',
            yaxis_title='x2',
            zaxis_title='f(x1, x2)',
            aspectratio=dict(x=1, y=1, z=1),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1)),
            xaxis=dict(
                gridcolor='#f0f0f0',
                zerolinecolor='#f0f0f0',
                title_font=dict(family="Palatino"),
                tickfont=dict(family="Palatino", size=10),
                showbackground=True, showline=True, linecolor="black", linewidth=1,
                backgroundcolor="white"
            ),
            yaxis=dict(
                gridcolor='#f0f0f0',
                zerolinecolor='#f0f0f0',
                title_font=dict(family="Palatino"),
                tickfont=dict(family="Palatino", size=10),
                showbackground=True, showline=True, linecolor="black", linewidth=1,
                backgroundcolor="white"
            ),
            zaxis=dict(
                gridcolor='#f0f0f0',
                zerolinecolor='#f0f0f0',
                title_font=dict(family="Palatino"),
                tickfont=dict(family="Palatino", size=10),
                showbackground=True, showline=True, linecolor="black", linewidth=1,
                backgroundcolor="white"
            ),
        ),
        width=800,
        height=700,
        margin=dict(l=65, r=50, b=65, t=90),
        font=dict(family="Palatino"),
        paper_bgcolor='white',
        plot_bgcolor='white',
    )

    return fig


def display_paths(x1_start=-0.5, x2_start=1, step_size=0.1, iterations=10):
    # Call the functions that return Plotly figures
    fig1 = show_gd_path_contour(x1_start, x2_start, step_size, iterations)
    fig2 = show_gd_path_surface(x1_start, x2_start, step_size, iterations)
    
    # Extract traces and titles
    traces1 = fig1.data
    traces2 = fig2.data
    title1 = fig1.layout.title.text if fig1.layout.title.text else "Plot 1"
    title2 = fig2.layout.title.text if fig2.layout.title.text else "Plot 2"
    
    # Create a subplot figure
    fig = make_subplots(rows=2, cols=1, specs=[
        [{"type": "contour"}],
        [{"type": "surface"}]
    ], shared_xaxes=True, shared_yaxes=True)
    
    # Add traces to the new figure
    for trace in traces1:
        fig.add_trace(trace, row=1, col=1)
    for trace in traces2:
        fig.add_trace(trace, row=2, col=1)
    
    # Update layout with Palatino title font
    fig.update_layout(
        title=dict(
            text=title1,
            font=dict(family="Palatino")
        ),
        width=1600,
        height=700,
        scene=dict(
            xaxis_title='x1',
            yaxis_title='x2',
            zaxis_title='f(x1, x2)',
            aspectratio=dict(x=1, y=1, z=1),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1))
        ),
        showlegend=False
    )
    
    # Show the figure
    return fig
