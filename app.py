import numpy as np
import dash
from dash import dcc, html, Output, Input
import plotly.graph_objs as go
import os

# 你的参数
features = ['WaveAtt-Net\nscore', "Intratumoral\nartery", "\"Capsule\"\nappearance", "Corona\nenhancement"]
coefs = [4.239346, -0.422436, -0.193177, -0.05293]
intercept = -3.036185
feature_ranges = [
    (0, 1),
    (0, 1),
    (0, 1),
    (0, 1)
]
feature_labels = [('0', '1'), ('absent', 'present'), ('absent', 'present'), ('absent', 'present')]

# 分数缩放
contribs = np.abs(np.array(coefs) * (np.array([r[1] - r[0] for r in feature_ranges])))
max_contrib = max(contribs)
score_scale = 100 / max_contrib
total_score_max = sum([abs(c * (r[1]-r[0])) * score_scale for c, r in zip(coefs, feature_ranges)])

def calc_points(values):
    points = [coef * (-(v - r[0]) if coef < 0 else (v - r[0])) * score_scale for coef, v, r in zip(coefs, values, feature_ranges)]

    # points = [coef * (v - r[0]) * score_scale for coef, v, r in zip(coefs, values, feature_ranges)]
    total = sum(points)
    return points, total

def calc_probability(total_points):
    lp = intercept + total_points / score_scale
    prob = 1/(1 + np.exp(-lp))
    return prob

def get_prob_scale():
    probs = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85]
    prob_scores = [score_scale * (np.log(p/(1-p)) - intercept) for p in probs]
    return probs, prob_scores

# 构造Dash页面（DL score为滑块，其余为单选）
app = dash.Dash(__name__)

def feature_input(i):
    labels = feature_labels[i]
    if i == 0:
        # DL score为滑块
        return html.Div([
            html.Label(features[i], style={'font-weight':'bold'}),
            dcc.Slider(
                id=f'feat-{i}',
                min=0, max=1, step=0.01, value=0,
                marks={0: '0', 0.5: '0.5', 1: '1'},
                tooltip={"placement": "bottom", "always_visible": True},
                included=False,
            )
        ], style={'marginRight': '30px', 'display': 'inline-block', 'width': '300px', 'font-size':'18px'})
    else:
        return html.Div([
            html.Label(features[i], style={'font-weight':'bold'}),
            dcc.RadioItems(
                id=f'feat-{i}',
                options=[
                    {'label': f"{labels[1]}", 'value': 0}, # 因为absent才加分
                    {'label': f"{labels[0]}", 'value': 1}
                ],
                value=0,
                inline=True
            )
        ], style={'marginRight': '30px', 'display':'inline-block', 'font-size':'18px'})

app.layout = html.Div([
    html.Div([
        # 标题区
        html.Div(
            html.H2('Interactive Nomogram', style={'textAlign': 'center', 'margin': 0}),
            style={
                'background': '#e3f1fb',      # 标题区色块
                'borderTopLeftRadius': '16px',
                'borderTopRightRadius': '16px',
                'padding': '18px 0 10px 0',  # 上右下左
                'marginBottom': '0',
                'borderBottom': '1px solid #d0d0d0'
            }
        ),
        # 控件区
        html.Div(
            [feature_input(i) for i in range(len(features))],
            style={
                'background': '#f7fafe',        # 控件区色块
                'padding': '22px 0 18px 0',
                'borderBottomLeftRadius': '16px',
                'borderBottomRightRadius': '16px',
                'textAlign': 'center'
            }
        ),
        dcc.Graph(id='nomogram-graph', config={'displayModeBar': False}),
        # html.Div(id='score-output', style={'fontSize': 22, 'textAlign':'center', 'marginTop':'10px'})
    ],style={
        'margin-bottom':'20px',
        'text-align':'center',          # 让所有输入块顶部对齐
        'verticalAlign': 'top',
        'maxWidth': '1100px',           # 控制宽度
        'margin': '40px auto',          # 居中
        'background': '#fff',           # 白色背景
        'border': '2px solid #d0d0d0',  # 灰色边框
        'borderRadius': '18px',         # 圆角
        'boxShadow': '0 8px 24px 0 rgba(0,0,0,0.10)',  # 阴影
        'padding': '36px 36px 24px 36px',              # 内边距
        })
])

@app.callback(
    Output('nomogram-graph', 'figure'),
    # Output('score-output', 'children'),
    [Input('feat-0', 'value'),
     Input('feat-1', 'value'),
     Input('feat-2', 'value'),
     Input('feat-3', 'value')]
)
def update_nomogram(dl_score, f1, f2, f3):
    feat_values = [dl_score, f1, f2, f3]
    points, total = calc_points(feat_values)
    prob = calc_probability(total)
    # 画图
    fig = go.Figure()
    y0 = len(features) + 4
    ygap = 0.5

    x_axis_move = 4

    # Points总标尺
    fig.add_trace(go.Scatter(
        x=[0+x_axis_move, 100+x_axis_move], y=[y0, y0],
        mode='lines', line=dict(color='black', width=2),
        showlegend=False  # 这一行！
    ))
    for pt in np.linspace(0, 100, 11):
        fig.add_trace(go.Scatter(
            x=[pt+x_axis_move, pt+x_axis_move], y=[y0, y0+0.1], mode='lines', line=dict(color='black', width=2), showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[pt+x_axis_move], y=[y0+0.15], mode='text', text=[f'{int(pt)}'], textfont=dict(size=16), textposition='top center', showlegend=False
        ))
    fig.add_trace(go.Scatter(x=[-15], y=[y0], mode='text', text=['Points'], textfont=dict(size=16, family='Arial', color='black',), showlegend=False))

    # 各特征尺及当前值
    for i, (feat, coef, r, labels) in enumerate(zip(features, coefs, feature_ranges, feature_labels)):
        y = y0 - (i+1)*ygap
        max_score = abs(coef * (r[1] - r[0])) * score_scale
        # 标尺
        fig.add_trace(go.Scatter(x=[0+x_axis_move, max_score+x_axis_move], y=[y, y], mode='lines', line=dict(color='black', width=2), showlegend=False))
        fig.add_trace(go.Scatter(x=[0+x_axis_move, 0+x_axis_move], y=[y, y-0.1], mode='lines', line=dict(color='black', width=2), showlegend=False))
            
        if i == 0:
            fig.add_trace(go.Scatter(x=[max_score+x_axis_move, max_score+x_axis_move], y=[y, y-0.1], mode='lines', line=dict(color='black', width=2), showlegend=False))
            for pt in np.linspace(0, max_score, 11):
                ratio = pt / max_score
                fig.add_trace(go.Scatter(x=[pt+x_axis_move, pt+x_axis_move], y=[y, y-0.06], mode='lines', line=dict(color='black', width=2), showlegend=False))
                fig.add_trace(go.Scatter(x=[pt+x_axis_move], y=[y-0.1], mode='text', text=[f'{ratio:.1f}'], textfont=dict(size=14), textposition='bottom center', showlegend=False))
        else:
            fig.add_trace(go.Scatter(x=[max_score+x_axis_move, max_score+x_axis_move], y=[y, y+0.1], mode='lines', line=dict(color='black', width=2), showlegend=False))
        
        if coef < 0:
            fig.add_trace(go.Scatter(x=[0+x_axis_move-0.5], y=[y-0.15], mode='text', text=[labels[1]], textfont=dict(size=14), showlegend=False, textposition='middle center'))
            fig.add_trace(go.Scatter(x=[max_score+x_axis_move+0.5], y=[y+0.15], mode='text', text=[labels[0]], textfont=dict(size=14), showlegend=False, textposition='middle center'))
            # 当前点
            val = -feat_values[i]
            this_score = (coef * (val - r[0])) * score_scale
            fig.add_trace(go.Scatter(x=[this_score+x_axis_move], y=[y], mode='markers', marker=dict(size=12, color='blue'), showlegend=False))
        else:   
            # fig.add_trace(go.Scatter(x=[0+x_axis_move-0.5], y=[y-0.18], mode='text', text=[labels[0]], textfont=dict(size=14), showlegend=False, textposition='middle center'))
            # fig.add_trace(go.Scatter(x=[max_score+x_axis_move+0.5], y=[y+0.18], mode='text', text=[labels[1]], textfont=dict(size=14), showlegend=False, textposition='middle center'))
            # 当前点
            val = feat_values[i]
            this_score = (coef * (val - r[0])) * score_scale
            fig.add_trace(go.Scatter(x=[this_score+x_axis_move], y=[y], mode='markers', marker=dict(size=12, color='blue'), showlegend=False))
       
        # 特征名
        fig.add_trace(go.Scatter(x=[-18], y=[y], mode='text', text=[feat.replace('\n', '<br>')], textfont=dict(size=16, color='black'), showlegend=False, textposition='middle right'))

    # Total Points 尺
    y_total = y0 - (len(features)+1)*ygap
    fig.add_trace(go.Scatter(x=[0+x_axis_move, total_score_max+x_axis_move], y=[y_total, y_total], mode='lines', line=dict(color='black', width=3), showlegend=False))
    fig.add_trace(go.Scatter(x=[0+x_axis_move, 0+x_axis_move], y=[y_total, y_total-0.1], mode='lines', line=dict(color='black', width=2), showlegend=False))
    fig.add_trace(go.Scatter(x=[total_score_max+x_axis_move, total_score_max+x_axis_move], y=[y_total, y_total-0.1], mode='lines', line=dict(color='black', width=2), showlegend=False))
    # 总分名
    fig.add_trace(go.Scatter(x=[-18], y=[y_total], mode='text', text=['Total Points'], textfont=dict(size=16), showlegend=False, textposition='middle right'))
    # 分刻度
    for pt in np.linspace(0, total_score_max, int(total_score_max)):
        if int(pt) % 10 ==0 or pt == total_score_max:
            fig.add_trace(go.Scatter(x=[pt+x_axis_move, pt+x_axis_move], y=[y_total, y_total-0.1], mode='lines', line=dict(color='black', width=2), showlegend=False))
            fig.add_trace(go.Scatter(x=[pt+x_axis_move, pt+x_axis_move], y=[y_total-0.15, y_total-0.15], mode='text', text=[int(pt)], textfont=dict(size=12), showlegend=False, textposition='bottom center'))
        elif int(pt) % 2 ==0:
            fig.add_trace(go.Scatter(x=[pt+x_axis_move, pt+x_axis_move], y=[y_total, y_total-0.06], mode='lines', line=dict(color='black', width=2), showlegend=False))
    # 当前总分
    fig.add_trace(go.Scatter(x=[total+x_axis_move], y=[y_total], mode='markers', marker=dict(size=12, color='red'), showlegend=False))
    fig.add_trace(go.Scatter(x=[total+x_axis_move], y=[y_total+0.06], mode='text', text=[f'{total:.1f}'], textfont=dict(size=16, color='red'), showlegend=False, textposition='top center'))

    # 概率尺
    y_prob = y0 - (len(features)+2)*ygap
    probs, prob_scores = get_prob_scale()
    fig.add_trace(go.Scatter(x=[min(prob_scores)+x_axis_move, max(prob_scores)+x_axis_move], y=[y_prob, y_prob], mode='lines', line=dict(color='black', width=3), showlegend=False))
    fig.add_trace(go.Scatter(x=[-18], y=[y_prob], mode='text', text=['Risk of response'], textfont=dict(size=16), showlegend=False, textposition='middle right'))
    for s, p in zip(prob_scores, probs):
        if 0 <= s <= total_score_max:
            fig.add_trace(go.Scatter(
                x=[s+x_axis_move, s+x_axis_move], y=[y_prob-0.1, y_prob], mode='lines', line=dict(color='black', width=2), showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=[s+x_axis_move], y=[y_prob-0.15], mode='text', text=[f'{p:.2f}'], textfont=dict(size=16), showlegend=False, textposition='bottom center'
            ))
    # 当前概率点
    fig.add_trace(go.Scatter(x=[total+x_axis_move], y=[y_prob], mode='markers', marker=dict(size=12, color='green'), showlegend=False))
    fig.add_trace(go.Scatter(x=[total+x_axis_move], y=[y_prob+0.06], mode='text', text=[f'{calc_probability(total):.2%}'], textfont=dict(size=16, color='green'), showlegend=False, textposition='top center'))

    # 美化
    fig.update_layout(
        # width=1400,  # 变宽
        height=400+(len(features)-2)*50,
        yaxis=dict(showticklabels=False, range=[y_prob-0.8, y0+0.8], zeroline=False),
        xaxis=dict(showticklabels=False, range=[-18, total_score_max+8], zeroline=False),
        margin=dict(l=0, r=0, t=20, b=20),
        plot_bgcolor='white'
    )

    return fig #, f"Total Points: <b>{total:.1f}</b> | Predicted Risk: <span style='color:green'><b>{prob:.2%}</b></span>"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 8080)), debug=False)
