from typing import List

from dash import html, dcc


def addSideBarInfoLabel(id: str, classWrapper="info-label"):
    """
    Adds a dash Label with the two ids id_main and id_hint for the main text and the hint text
    :param id: ID of the dash component
    :param classWrapper: CSS Class for wrapper
    :return: Dash Element
    """
    return html.Div(className=classWrapper, children=[
        html.Div(id=id + '_main'),
        html.Div(id=id + '_hint', className='hint'),
    ])


def addInputWithTitle(id: str, name: str, type: str, val:any, step: float = None, classWrapper="margin-100-bottom",
                      classInput="fl-grow pad-25-top", persistence=True, **input_kwargs):

    return html.Div(className=classWrapper + " fl-grow input-container", children= [
        html.Label(children=name, className="pad-25-bottom"),
        dcc.Input(id=id, type="number", value=val, className= classInput + "margin-100-right", step=step, persistence=persistence, **input_kwargs)
    ])

def addSliderWithTitle(id: str, name: str, range: List, step: float, classWrapper="margin-100-bottom",
                       classSlider="fl-grow pad-25-top", persistence=True, value=None, **kwargs):
    """
    Adds a dash Slider with 2 handles
    :param id: ID of the dash component
    :param name: Label for the slider
    :param range: List with 2 entries, min and max
    :param step: Stepping of slider
    :param classWrapper: CSS Class for wrapper
    :param classSlider: CSS Class for Slider
    :param persistence: Dash persistence, i.e. if value is stored in session

    :return: Dash Element
    """
    v = value if value is not None else [range[0], range[1]]
    return html.Div(className=classWrapper+" slider-container", children=[
        html.Label(children=name),
        dcc.RangeSlider(className=classSlider + ' range-slider', min=range[0], max=range[1], step=step,
                        value=v, id=id,
                        persistence=True,
                        tooltip={"placement": "bottom", "always_visible": True},
                        marks=None,
                        **kwargs)
    ])


def addDropDown(id: str, name: str, choices, classWrapper="margin-100-bottom", classDCC="fl-grow pad-25-top",
                persistence=True, initial = None, **kwargs):
    """
    Adds a dash Dropdown
    :param id: Dropdown id
    :param name: Label name
    :param choices: Choices as a list or key-value dictionary
    :param classWrapper: CSS class of wrapper
    :param classDCC: CSS class of Dropdown
    :param persistence: Dash persistence, i.e. if value is stored in session
    :param kwargs: Any additional arguments to pass to dcc.Dropdown (see documentation)
    :return:
    """
    if initial is None and "Multi" in kwargs and kwargs["Multi"]:
        initial = []
    elif len(choices) > 0:
        #check if choices is a dictionary
        if isinstance(choices, dict):
            initial = list(choices.keys())[0] if initial is None else initial
        elif isinstance(choices, list):
            initial = choices[0] if initial is None else initial


    return html.Div(className=classWrapper+" dropdown-container", children=[
        html.Strong(name),
        dcc.Dropdown(choices, value=initial, className=classDCC, id=id, persistence=persistence, **kwargs)
    ])
def addChecklist(id: str, name: str, choices, classWrapper="margin-100-bottom", classDCC="fl-grow pad-25-top",
                persistence=True, initial = None, **kwargs):
    """
    Adds a dash Checklist
    :param id: Checklist id
    :param name: Label name
    :param choices: Choices as a list or key-value dictionary
    :param classWrapper: CSS class of wrapper
    :param classDCC: CSS class of Dropdown
    :param persistence: Dash persistence, i.e. if value is stored in session
    :param kwargs: Any additional arguments to pass to dcc.Dropdown (see documentation)
    :return:
    """
    if initial is None: initial = []

    return html.Div(className=classWrapper+" checklist-container", children=[
        html.Strong(name),
        dcc.Checklist(choices, value=initial, className=classDCC, id=id, persistence=persistence, **kwargs)
    ])
