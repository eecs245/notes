"""Shared interactive plane style, matching Math 124 Chapter 2.4.

WebGL labels use italic HTML, combining vector arrows, and HTML scripts;
LaTeX belongs in notebook prose/captions, not Scatter3d text.
"""
from itertools import product, combinations
import numpy as np
import plotly.graph_objects as go

BLUE, ORANGE, PINK = '#3d81f6', 'orange', '#d81a60'
GRAY = '#9ca3af'
FONT = 'Palatino Linotype, Palatino, serif'


def vector_label(letter, sub=None, sup=None):
    label = f'<i>{letter}</i>⃗'
    if sub is not None:
        label += f'<sub>{sub}</sub>'
    if sup is not None:
        label += f'<sup>{sup}</sup>'
    return label


def line3(fig, start, end, color=BLUE, width=5, dash='solid'):
    points = np.asarray([start, end], float)
    fig.add_trace(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode='lines', line=dict(color=color, width=width, dash=dash),
        showlegend=False, hoverinfo='skip'))


def label3(fig, position, label, color=BLUE, size=20):
    fig.add_trace(go.Scatter3d(x=[position[0]], y=[position[1]], z=[position[2]],
        mode='text', text=[label], textfont=dict(family=FONT, color=color, size=size),
        showlegend=False, hoverinfo='skip'))


def vector3(fig, end, label='', color=BLUE, start=(0, 0, 0), offset=None):
    start, end = np.asarray(start, float), np.asarray(end, float)
    length = np.linalg.norm(end - start)
    if length == 0:
        return
    span = np.ptp(fig.layout.scene.xaxis.range)
    tip = 0.035 * span
    direction = (end - start) / length
    line3(fig, start, end - min(tip * 0.3, length * 0.1) * direction, color, width=7)
    fig.add_trace(go.Cone(x=[end[0]], y=[end[1]], z=[end[2]],
        u=[direction[0]], v=[direction[1]], w=[direction[2]],
        anchor='tip', sizemode='absolute', sizeref=tip,
        colorscale=[[0, color], [1, color]], showscale=False, hoverinfo='skip'))
    if label:
        offset = np.asarray(offset if offset is not None else (0.025*span, 0.025*span, 0.035*span))
        label3(fig, end + offset, label, color)


def plane_figure(bounds=(-6, 7), axis_titles=('<i>x</i>', '<i>y</i>', '<i>z</i>'), camera=None):
    axis = dict(range=list(bounds), showbackground=True, backgroundcolor='white',
                gridcolor='#e5e7eb', zerolinecolor=GRAY, showspikes=False,
                tickfont=dict(size=12), nticks=7)
    fig = go.Figure()
    fig.update_layout(autosize=True, height=520, showlegend=False,
        paper_bgcolor='white', plot_bgcolor='white', font=dict(family=FONT, size=16, color='black'),
        margin=dict(l=0, r=0, t=15, b=0), scene=dict(bgcolor='white',
        xaxis=dict(title=axis_titles[0], **axis), yaxis=dict(title=axis_titles[1], **axis),
        zaxis=dict(title=axis_titles[2], **axis), aspectmode='cube',
        camera=dict(eye=camera or dict(x=1.6, y=-2.1, z=1.3))))
    for direction in np.eye(3):
        line3(fig, bounds[0]*direction, bounds[1]*direction, GRAY, width=2)
    return fig


def plane_polygon(normal, d, bounds):
    """Intersect a plane with a finite box; handles vertical planes too."""
    normal = np.asarray(normal, float)
    if np.linalg.norm(normal) == 0:
        raise ValueError('A plane needs a nonzero normal.')
    vertices = np.array(list(product(*bounds)), dtype=float)
    points = []
    for a, b in combinations(vertices, 2):
        if np.count_nonzero(a != b) != 1:
            continue
        fa, fb = a @ normal - d, b @ normal - d
        if abs(fa) < 1e-9:
            points.append(a)
        if abs(fb) < 1e-9:
            points.append(b)
        if fa * fb < 0:
            points.append(a + fa / (fa - fb) * (b - a))
    if not points:
        raise ValueError('Plane does not intersect the plotting box.')
    points = np.unique(np.round(points, 10), axis=0)
    center = points.mean(axis=0)
    u = points[0] - center
    u /= np.linalg.norm(u)
    v = np.cross(normal / np.linalg.norm(normal), u)
    angles = np.arctan2((points-center) @ v, (points-center) @ u)
    return points[np.argsort(angles)]


def add_plane(fig, normal, d=0, color=BLUE, opacity=0.24, name=None):
    # Use the original gray-axis extents, so adding another plane cannot
    # enlarge the next patch recursively. Solve for the most stable coordinate.
    bounds = np.array([[fig.data[j][axis][0], fig.data[j][axis][1]]
                       for j, axis in enumerate(('x', 'y', 'z'))], dtype=float)
    normal = np.asarray(normal, dtype=float)
    if np.linalg.norm(normal) == 0:
        raise ValueError('A plane needs a nonzero normal.')
    dependent = int(np.argmax(np.abs(normal)))
    free = [j for j in range(3) if j != dependent]
    inset = bounds[:, 0] + 0.07 * np.ptp(bounds, axis=1)
    outset = bounds[:, 1] - 0.07 * np.ptp(bounds, axis=1)
    p = np.zeros((4, 3))
    p[:, free[0]] = [inset[free[0]], outset[free[0]], outset[free[0]], inset[free[0]]]
    p[:, free[1]] = [inset[free[1]], inset[free[1]], outset[free[1]], outset[free[1]]]
    p[:, dependent] = (d - p @ normal) / normal[dependent]
    # Show every corner with padding instead of cutting the patch at z limits.
    for j, axis in enumerate(('xaxis', 'yaxis', 'zaxis')):
        current = getattr(fig.layout.scene, axis)
        pad = 0.07 * np.ptp(bounds[j])
        current.range = [min(current.range[0], p[:, j].min() - pad),
                         max(current.range[1], p[:, j].max() + pad)]
    fig.update_scenes(aspectmode='data')
    group = f'plane-{len(fig.data)}'
    fig.add_trace(go.Mesh3d(x=p[:,0], y=p[:,1], z=p[:,2],
        i=[0]*(len(p)-2), j=list(range(1,len(p)-1)), k=list(range(2,len(p))),
        color=color, opacity=opacity, flatshading=True,
        lighting=dict(ambient=1, diffuse=0, specular=0),
        name=name, legendgroup=group, showlegend=bool(name), hoverinfo='skip'))
    q = np.vstack([p, p[0]])
    fig.add_trace(go.Scatter3d(x=q[:,0], y=q[:,1], z=q[:,2], mode='lines',
        line=dict(color=color, width=2), legendgroup=group, showlegend=False, hoverinfo='skip'))
    return fig


def surface_border(x, y, z):
    coords = [np.asarray(v) for v in (x,y,z)]
    return [v[[0,0,-1,-1,0], [0,-1,-1,0,0]].tolist() for v in coords]


def style_surface_planes(fig, indices, colors=None, height=520):
    """Style selected planar Surface traces, including slider-driven borders.

    Curved surfaces, data coordinates, and the original trace indices are kept.
    """
    colors = colors or [BLUE]*len(indices)
    n = len(fig.data)
    for idx, color in zip(indices, colors):
        trace = fig.data[idx]
        trace.update(colorscale=[[0,color],[1,color]], opacity=0.24,
                     showscale=False, hoverinfo='skip',
                     lighting=dict(ambient=1, diffuse=0, specular=0))
        x,y,z = surface_border(trace.x,trace.y,trace.z)
        group = trace.legendgroup or f'plane-{idx}'
        trace.legendgroup = group
        fig.add_trace(go.Scatter3d(x=x,y=y,z=z, mode='lines',
            scene=trace.scene, visible=trace.visible, legendgroup=group,
            line=dict(color=color,width=2), showlegend=False, hoverinfo='skip'))

    # Update all slider states and axis-switch buttons, including nested sliders.
    def patch_layout(value):
        if isinstance(value, list):
            for item in value: patch_layout(item)
        elif isinstance(value, dict):
            if value.get('method') in ('update', 'restyle') and 'args' in value:
                update = value['args'][0]
                if isinstance(update, dict):
                    if all(k in update and len(update[k]) == n for k in ('x','y','z')):
                        borders = [surface_border(*(update[k][idx] for k in ('x','y','z'))) for idx in indices]
                        for j,k in enumerate(('x','y','z')):
                            update[k] = list(update[k]) + [border[j] for border in borders]
                    if 'visible' in update and len(update['visible']) == n:
                        update['visible'] = list(update['visible']) + [update['visible'][idx] for idx in indices]
            for item in value.values(): patch_layout(item)
    layout = fig.layout.to_plotly_json()
    patch_layout(layout)
    fig.layout = layout
    fig.update_layout(width=None, autosize=True, height=height,
        paper_bgcolor='white', plot_bgcolor='white',
        font=dict(family=FONT, size=16, color='black'))
    fig.update_scenes(bgcolor='white',
        xaxis=dict(backgroundcolor='white',gridcolor='#e5e7eb',zerolinecolor=GRAY,showspikes=False),
        yaxis=dict(backgroundcolor='white',gridcolor='#e5e7eb',zerolinecolor=GRAY,showspikes=False),
        zaxis=dict(backgroundcolor='white',gridcolor='#e5e7eb',zerolinecolor=GRAY,showspikes=False))
    return fig


def projection_figure(y, basis, candidates=None, notation='o'):
    """Column-space example with the same colors as the surrounding equations."""
    y, basis = np.asarray(y), np.asarray(basis)
    fig = plane_figure(bounds=(-3,6), camera=dict(x=1,y=-1.3,z=1))
    add_plane(fig, np.cross(basis[:,0],basis[:,1]))
    vector3(fig,y,vector_label('y'),ORANGE,offset=(0.25,0.3,0.35))
    label3(fig,(3.5,-1.4,1.1),'colsp(<i>X</i>)',BLUE)
    if candidates is None:
        vectors = [basis[:,0], -basis[:,1], -3.5*basis[:,0]+4*basis[:,1]]
        for j,v in enumerate(vectors,1):
            vector3(fig,v,vector_label('x',sup=f'({j})'),BLUE,
                    offset=(-0.3,0.15,0.25) if j==2 else (0.3,0.2,0.35))
    else:
        for j,p in enumerate(candidates):
            p = np.asarray(p)
            if j:
                plabel = vector_label('p',sup='′')+' = <i>X</i>'+vector_label('w',sup='′')
                elabel = vector_label('e',sup='′')
            elif notation == '*':
                plabel = vector_label('p')+' = <i>X</i>'+vector_label('w',sup='*')
                elabel = vector_label('e')
            else:
                plabel = vector_label('p',sub='o')+' = <i>X</i>'+vector_label('w',sub='o')
                elabel = vector_label('e',sub='o')
            vector3(fig,p,'','#004d40')
            label3(fig, (1.1,3.1,-0.65) if j==0 else (2.5,-0.6,1), plabel,'#004d40')
            line3(fig,p,y,PINK,4,'dash')
            label3(fig,(p+y)/2 + np.array([0.4,0,0.1]),elabel,PINK)
            if j==0 and np.isclose(np.dot(y-p,p),0) and np.linalg.norm(p)>0:
                a, b = -p/np.linalg.norm(p)*0.35, (y-p)/np.linalg.norm(y-p)*0.35
                line3(fig,p+a,p+a+b,GRAY,2)
                line3(fig,p+a+b,p+b,GRAY,2)
    return fig
