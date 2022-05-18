import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

from app import app
from apps import estimation_app


app.layout = html.Div(children=[
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),
])

first_card = dbc.Card(
    [
        dbc.CardImg(src="/assets/example.jpeg", top=True),
        dbc.CardBody(
            [
                html.H4("About", className="card-title"),
                html.P(
                    "The problem of engagement estimation is the problem of assessing the level of a person's interest" 
                    "in the process taking place (in particular, a video meeting). "
                    "Usually it is solved by finding the face, assessing the head posture and direction of the gaze.",
                    className="card-text",
                ),
                # dbc.Button("Proof of concept", color="primary", href=""),
            ]
        ),
    ],
    style={
        "width": "20rem",
        "margin-left": "200px",
        "margin-top": "50px"},
)


second_card = dbc.Card(
    [
        dbc.CardImg(src="/assets/user_guide.jpg", top=True),
        dbc.CardBody(
            [
                html.H4("User manual", className="card-title"),
                html.P(
                    "To run our system push the button below",
                    className="card-text",
                ),
                dbc.Button("Run Estimation", color="primary", href='/apps/estimation_app'),
            ]
        ),
    ],
    style={
        "width": "20rem",
        "margin-top": "50px"},
)

cards = dbc.Row([dbc.Col(first_card, width="auto"),
                 dbc.Col(second_card, width="auto"),])

index_page =  html.Div(children=[
    dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Github Link", href='https://github.com/Lolik-Bolik/engagement_estimation_system')),
            dbc.NavItem(dbc.NavLink("Run Estimation", href='/apps/estimation_app')),
        ],
        brand="Engagement Estimation",
        brand_href="#",
        color="primary",
        dark=True,
    ),
    cards
],
    style={
        "background-image": 'url(/assets/background_2.png)',
        "background-repeat": "no-repeat",
        "background-position": "center",
        "background-size": "cover",
        "position": "fixed",
	    "min-height": "100%",
	    "min-width": "100%",})


@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    print(pathname)
    if pathname == '/apps/estimation_app':
        return estimation_app.layout
    else:
        return index_page


if __name__ == '__main__':
    app.run_server(debug=True)