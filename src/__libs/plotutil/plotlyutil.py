from typing import List, Dict, Union, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
import plotly.figure_factory as ff
def drawCross(fig:go.Figure, x,y,w,h, name:str, col="black", markerProps=None,lineProps=None):
    """Draws a marker at x/y and two lines +/- w and h of it. For visualizing mean and std of distributions on a scatter plot for example"""
    if markerProps is None:
        markerProps = dict(marker_color=col, marker_size=20, marker_symbol="cross")
    if lineProps is None:
        lineProps = dict(color=col, width=1, dash="dash")

    #draw a cross at the weighted mean
    fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers", name=name,**markerProps))
    #draw lines indicating mean and max variance
    fig.add_trace(go.Scatter(x=[x,x], y=[y - h, y+h], mode="lines", line=lineProps,showlegend=False))
    fig.add_trace(go.Scatter(x=[x - w,x + w], y=[y, y], mode="lines", line=lineProps,showlegend=False))

def convertFloatMapToColorMap(inp:np.ndarray, colorscale, fillnanValue = np.nan):
    arr = inp.copy()
    #norm 0-1
    arr = (arr - np.nanmin(arr)) / (np.nanmax(arr) - np.nanmin(arr))

    arr = np.reshape(arr, (-1, 1))
    nanmask = np.isnan(arr)
    flatNoNan = arr[~nanmask]

    #will contain N x 3
    pred = np.array(px.colors.sample_colorscale(colorscale, arr[~nanmask], colortype="tuple"))

    res = np.ones((len(arr),3)) * fillnanValue
    res[~nanmask[:,0],:] = pred

    return np.reshape(res, inp.shape + (3,))

def drawDistMeanLine(fig, x,y:np.array, col="black", lineProps=None ):
    """Draws a line at the mean of the distribution"""
    if lineProps is None:
        lineProps = dict(color=col, width=1)

    distMeanX = np.sum(x * y) / np.sum(y)
    #approximate value of y at distMeanX
    distMeanY = np.interp(distMeanX,x,y)

    fig.add_trace(go.Scatter(x=[distMeanX,distMeanX], y=[0, distMeanY], mode="lines+markers", marker_symbol="circle", marker_size=10, line=lineProps,showlegend=False))
    return fig

def plotScatterWithMean(data:pd.DataFrame, axis:List[str], labels:Dict, **scatterArgumens):
    """
    Draws a scatter plot with the mean +/- std as a cross
    :param data: Dataframe with columns axis[0] and axis[1]
    :param axis: The columns
    :param labels: Dictionary with axis names , optional
    :return:
    """

    fig = go.Figure()
    x = axis[0]
    y = axis[1]

    opt = dict(marker_size=3, marker_color="blue", showlegend=False,mode="markers")
    opt.update(scatterArgumens)

    fig.add_trace(go.Scatter(x=data[x], y=data[y], **opt))

    drawCross(fig, data[x].mean(), data[y].mean(), data[x].std(), data[y].std(),
                         "Mean Change")

    # add x and y axis labels
    if labels is not None:
        fig.update_xaxes(title_text=str(labels[x]))
        fig.update_yaxes(title_text=str(labels[y]))

    fig.show()

def getDendrogramFromFeatureVectors(featureVectors:pd.DataFrame, fig, r=None,c=None, orientation="bottom" ):
    fig2 = ff.create_dendrogram(featureVectors.to_numpy(),
                                color_threshold=0.5,
                                labels=featureVectors.index,
                                orientation=orientation,
                                linkagefun=lambda distmat: hierarchy.ward(squareform(distmat)))

    if fig is not None:
        fig.add_traces(fig2.data, rows=r, cols=c)
        propCopy = ["ticktext","tickvals"]
        if orientation == "bottom":
            fig.update_xaxes({p:fig2.layout["xaxis"][p] for p in propCopy},row=r,col=c)
        else:
            fig.update_yaxes({p:fig2.layout["yaxis"][p] for p in propCopy},row=r,col=c)
        #remove legend
        fig.update_traces(showlegend=False, row=r, col=c)

    return fig2

def getDendrogramFromCorrelationMatrix(corrMatrix:Union[np.ndarray,pd.DataFrame], fig, labels = None, r=None,c=None, orientation="bottom" ):
    """Creates a dendrogram from a correlation matrix using Ward's linkage"""
    if isinstance(corrMatrix,pd.DataFrame):
        X = corrMatrix.to_numpy()
    else:
        X = corrMatrix

    fig2 = ff.create_dendrogram(X,
                                color_threshold=0.5,
                                labels=corrMatrix.columns if labels is None else labels,
                                distfun=lambda x: 1 - np.abs(x),
                                orientation = orientation,
                                linkagefun=lambda distmat: hierarchy.ward(squareform(distmat)))

    if fig is not None:
        fig.add_traces(fig2.data, rows=r, cols=c)
        propCopy = ["ticktext","tickvals"]
        fig.update_xaxes({p:fig2.layout["xaxis"][p] for p in propCopy},row=r,col=c)
        #remove legend
        fig.update_traces(showlegend=False, row=r, col=c)

    return fig2

def initSubplots(title, cols:int,widthsOrRows:Union[int,List[int]],titles:List[str]=None,share:str = None, plotIDsNeedingSecYAxis:List[int] = None,**spargs):
    """Shorthand for initiating plotly subplots where some plots span multiple columns
    :param cols: Number of columns
    :param widthsOrRows: Widths in columns for each plot 2 for 2 columns, cannot be larger than cols or integer which is then just the number of rows
    :param titles: Titles for each plot
    :param share: "x" or "y" or "xy" to share axes
    """

    rows = int(np.ceil(np.sum(widthsOrRows) / cols)) if isinstance(widthsOrRows, list) else widthsOrRows
    args = {"cols": cols,
            "shared_xaxes": True if str(share).find("x") >= 0 else False,
            "shared_yaxes": True if str(share).find("y") >= 0 else False,
            "rows": rows,
            "subplot_titles": titles
            }

    specs = []
    widths = widthsOrRows if isinstance(widthsOrRows, list) else [1]*rows*cols

    for w in widths:
        specs.append({"colspan": w})
        if w > 1:
            specs += [None] * (w - 1)

    if plotIDsNeedingSecYAxis is not None:
        for i in plotIDsNeedingSecYAxis:
            specs[i]["secondary_y"] = True

    #use numpy to rearrange specs
    args["specs"] = np.array(specs).reshape((args["rows"],cols)).tolist()

    pos = []
    poss = []
    for r,row in enumerate(args["specs"]):
        for c,col in enumerate(row):
            if col is not None:
                pos += [{"row": r+1, "col": c+1}]
                poss += [{"rows": r+1, "cols": c+1}]


    fig = make_subplots(**args,**spargs)
    fig.update_layout(title_text=title)
    return fig,pos,poss

def drawLinesWithErrorbars():
    def plotForVars(v=PredictiveVariableSet.Full):
        fd2 = fd.loc[collapsedData.Variables == v.name]

        f = px.scatter(fd2, x="NoiseRed", y="Mean", color="Classifier",
                       title=f"Model Scores Means (for 'Equal' NoiseBalance and {_ccm.name} )",
                       range_y=[0.5, 1],
                       color_discrete_map=GlobalParams.modelColMapping(),
                       labels={"Mean": "Mean AUC score across all species", "NoiseRed": "Noise Reduction Fraction",
                               "Variables": "Variable Set"})
        # draw Q25 and Q75 as error bars
        fgr = fd2.groupby(["Classifier", "NoiseRed"])
        for i, ((c, nr), d) in enumerate(fgr):
            ##add vertical line as shape
            xp = nr
            f.add_shape(type="line", x0=xp, x1=xp, y0=d.Q25.min(), y1=d.Q75.max(),
                        line=dict(color=modelCols[c], width=1), opacity=1)

        f = _showScoreScatter(f)
        return f

__basePrintWidth = 1000
__basePrintHeight = 600
def saveAsPrint(path:Optional[str], f:go.Figure, h:any = "100%", w:any = "100%", tickfontx = 14, tickfonty=14,fontsize=14, noLegend:bool = False, **layoutargs):

    if isinstance(w,str) and "%" in w:
        w = __basePrintWidth * int(w.replace("%","")) / 100
    if isinstance(h,str) and "%" in h:
        h = __basePrintHeight * int(h.replace("%","")) / 100

    #update template to white
    f.update_layout(template="simple_white")
    #make size of x and y axis larger
    #change Font to Open Sans
    f.update_layout(font=dict(size=fontsize, family="Open Sans" ), height=h, width=w)
    f.update_xaxes(tickfont=dict(size=tickfontx))
    f.update_yaxes(tickfont=dict(size=tickfonty))
    #update the font size of the title
    f.update_layout(title_font_size=fontsize)
    #remove titles
    f.update_layout(title=None)

    if noLegend:
        f.update_layout(showlegend=False)

    f.update_layout(**layoutargs)



    if path is not None:
        print(f'Saving {path.split("/")[-1]}')
        f.update(layout_margin = dict(t=0,b=0,l=0,r=0))
        f.write_image(path)

        #run a terminal utilitiy on the path to copy it into clipboard
        imgContent = str(f.to_image(format="svg"))
        import subprocess
        subprocess.run(["pbcopy"], text=True, input=imgContent)


    return f
