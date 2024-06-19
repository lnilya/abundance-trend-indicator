import inspect
from math import ceil
from typing import List, Union, Literal, Tuple, Dict, TypeVar

import numpy as np
import pandas as pd
from plotly.graph_objs import Figure, Layout, Scatter, Histogram, Heatmap, Contour
from plotly.subplots import make_subplots

from src.__libs.plotutil import initSubplots
from plotly import express as px
from plotly import graph_objects as go


def _get_kwargs():
    frame = inspect.currentframe().f_back
    keys, _, _, values = inspect.getargvalues(frame)
    kwargs = {}
    for key in keys:
        if key != 'self':
            kwargs[key] = values[key]
    return kwargs

class BarGraphSettings:
    bargap:float = None
    barmode:str = None

    def toDict(self):
        return {k:v for k,v in self.__dict__.items() if v is not None}



PlotlyTrace = TypeVar('PlotType', bound=[Contour, go.Bar, go.Barpolar,go.Scatter,go.Histogram, go.Pie, go.Sankey, go.Scatter3d])

class PlotlySubplot:
    """Class storing some data for a subplot in a plotly graph."""

    title:str = None
    xlabel:str = None
    ylabel:str = None
    ylabelSec:str = None
    _traces:List[Figure]
    _traceList:List[PlotlyTrace]
    _secAxis:List[bool]
    _axesLimits:Dict[str,Tuple]
    _legend:Dict[str,any]
    _layoutAdditions:List

    def __init__(self, title:str = None, secYAxis:bool = False):
        self.title = title
        self.secYAxis = secYAxis
        self._traces = []
        self._secAxis = []
        self._axesLimits ={}
        self._legend ={}
        self._layoutAdditions = [BarGraphSettings()]
        self._traceList = []

    def _getSettings(self,type):
        for s in self._layoutAdditions:
            if isinstance(s,type):
                return s
        return None

    @property
    def bar(self)->BarGraphSettings:
        s = self._getSettings(BarGraphSettings)
        return s

    @property
    def typesOfTraces(self):
        """Returns the Plotly Traces that have been added to this graph. You have to use these types when retrieving using getTrace or tr methods.
        This will ensure that you get the correct code completion hints."""
        return [type(t).__name__ for t in self._traceList]

    def getTrace(self, tracenum, castTo:PlotlyTrace)->PlotlyTrace:
        """Returns the trace at the given index and casts it to the given type."""
        trace = self._traceList[tracenum]
        if castTo is None: return trace
        # noinspection PyTypeHints
        if not isinstance(trace, castTo):
            raise TypeError(f"The trace at position {tracenum} is of type {type(trace).__name__}. Pass this as an argument to getTrace/tr intead of {castTo.__name__}")
        # if not any(isinstance(trace, t) for t in self.valid_types):
        #     raise TypeError(f"Expected object of type {self.valid_types}, but got {type(trace).__name__}")
        return trace
    def tr(self, castTo:PlotlyTrace=None)->PlotlyTrace:
        """Convenience for getting the last trace added to this graph."""
        return self.getTrace(-1,castTo)
    def setLegends(self,names:List[str] = None):
        """Sets the legend titles if needed. Pass None to disable legends.
        If an element is a string it will be assigned to the trace at the same index.
        Trace indices are assigned by the order at which traces are added."""
        if names is None:
            self._legend = {"showlegend":False}
        else:
            for i, n in enumerate(names):
                if n is None:
                    self._traceList[i].showlegend = False
                else:
                    self._traceList[i].showlegend = True
                    self._traceList[i].name = n


    def setAxesLimis(self, x:Tuple = None, primaryY:Tuple = None, secondaryY:Tuple = None):
        self._axesLimits["x"] = x
        self._axesLimits["primaryY"] = primaryY
        self._axesLimits["secondaryY"] = secondaryY

    def setLabels(self, x:str = None, primaryY:str = None, secondaryY:str = None, title:str = None):
        if x is not None: self.xlabel = x
        if primaryY is not None: self.ylabel = primaryY
        if secondaryY is not None: self.ylabelSec = secondaryY
        if title is not None: self.title = title

    def hasSecYAxis(self):
        for b in self._secAxis:
            if b: return True
        return False
    def plot(self,fig,row=None,col=None):
        pos = {"row":row,"col":col}
        if row is None or col is None:
            pos = {}
        for i, t in enumerate(self._traces):
            if hasattr(t,"data"): #Plotly Express objects e.g. px.line

                #Some sort of plotly express bug here. The xaxis for the traces gets set correctly only for the first trace
                #when using add_traces. Instead make a loop and use add_trace

                for tr in t.data:
                    fig.add_trace(tr,**pos,secondary_y=self._secAxis[i] if self.hasSecYAxis() else None)

                #copy some of the layout
                #overwrite properties except anchor and domain (needed for subplots)
                xaxis = t.layout["xaxis"]
                xaxis.domain = xaxis.anchor = None
                yaxis = t.layout["yaxis"]
                yaxis.domain = yaxis.anchor = None
                fig.update_xaxes(xaxis,**pos)
                fig.update_yaxes(yaxis,secondary_y = self._secAxis[i], **pos)

                #Update all other elements of the layout that are not axes
                fl:Layout = fig.layout
                t.layout.xaxis = t.layout.yaxis = t.layout.template = None
                fl.update(t.layout)
                # fig.update_layout(t.layout)

            else: #Graph objects e.g. go.Scatter
                fig.add_trace(t,**pos,secondary_y=self._secAxis[i] if self.hasSecYAxis() else None)

            #Y Axes title
            if self._secAxis[i]: fig.update_yaxes(title_text=self.ylabelSec,secondary_y=True,**pos)
            else: fig.update_yaxes(title_text=self.ylabel,**pos)

        # X Axis title
        fig.update_xaxes(title_text=self.xlabel,**pos)

        #Axes limits
        if self._axesLimits.get("x") is not None: fig.update_xaxes(range=self._axesLimits["x"],**pos)
        if self._axesLimits.get("primaryY") is not None: fig.update_yaxes(range=self._axesLimits["primaryY"],**pos)
        if self._axesLimits.get("secondaryY") is not None: fig.update_yaxes(range=self._axesLimits["secondaryY"],secondary_y=True,**pos)

        #legend
        fig.update_traces(self._legend,**pos)

        #Merge all additional global settings into a layout dictionary
        layout = {}
        for s in self._layoutAdditions:
            layout.update(s.toDict())

        fig.update_layout(**layout)

    def _addLastTraceToTraceList(self):
        d = self._traces[-1]
        if hasattr(d,"data"): self._traceList += d.data
        else: self._traceList += [d]

    def _onDataAdded(self,other,secY:bool):
        self._traces.append(other)
        self._secAxis.append(secY)
        self._addLastTraceToTraceList()
        if isinstance(other, Figure): #when adding plotly express
            if other.layout.title is not None:
                self.title = other.layout.title["text"]
            if other.layout.xaxis.title.text:
                self.xlabel = other.layout.xaxis.title.text
            if other.layout.yaxis.title.text:
                if secY: self.ylabelSec = other.layout.yaxis.title.text
                else: self.ylabel = other.layout.yaxis.title.text

        return self
    def __ipow__(self, other):
        """Use **= to add to the seconday Y axis"""
        return self._onDataAdded(other,True)
    def __imul__(self, other):
        """Use *= to add to the primary Y axis"""
        return self._onDataAdded(other,False)

class SubplotSettings:
    vertical_spacing:float = None
    horizontal_spacing:float = None
class PlotlyGraph:
    """Wrapper for plotly to easier deal with subplots and styles."""

    _subplots:List[PlotlySubplot]
    def __init__(self,title=""):
        self.title = title
        self.fig = None
        self._subplots = []
        self._layoutAdditions = [SubplotSettings()]
        pass

    def __len__(self):
        return len(self._subplots)
    def __getitem__(self, item):
        while item >= len(self._subplots):
            self._subplots.append(PlotlySubplot())

        return self._subplots[item]

    def newPlot(self)->PlotlySubplot:
        return self[len(self._subplots)]

    def _getSettings(self, type):
        for s in self._layoutAdditions:
            if isinstance(s, type):
                return s
        return None

    @property
    def subplots(self) -> SubplotSettings:
        s = self._getSettings(SubplotSettings)
        return s

    def __setitem__(self, index, value):
        while index >= len(self._subplots):
            self._subplots.append(PlotlySubplot())
        self._subplots[index] = value


    def show(self,cols:int = 1,widthsOrRows:Union[int,List[int]] = None,share:Literal["x","y","xy"] = None, height=None,print=False,subplotArgs:Dict = {}, layoutArgs:Dict = {}, returnOnly = False):
        if len(self._subplots) == 0: return

        if len(self._subplots) > 1: #need subplots
            numCols = cols
            if widthsOrRows is None:
                widthsOrRows = int(ceil(len(self._subplots) / numCols))

            titles = [s.title for s in self._subplots]
            if isinstance(widthsOrRows,int):
                plotIDsWithSecAxis = [i for i,s in enumerate(self._subplots) if s.hasSecYAxis()]
            else:
                c = [0]
                for w in widthsOrRows: c.append(c[-1]+w)
                plotIDsWithSecAxis = [c[i] for i,s in enumerate(self._subplots) if s.hasSecYAxis()]

            spargs = self.subplots.__dict__
            spargs.update(subplotArgs)
            self.fig,pos,poss = initSubplots(self.title,numCols, widthsOrRows, titles, share, plotIDsWithSecAxis,**spargs)

            #trace all the subplots onto the figure at the right position
            for i, s in enumerate(self._subplots):
                s.plot(self.fig,pos[i]["row"],pos[i]["col"])
        else:
            secY = self._subplots[0].hasSecYAxis()
            #For some reasong when using secY axes plotly defaults to using rows and columns, which requires to create a subplot, even for 1 plot.
            self.fig = make_subplots(subplot_titles=[self._subplots[0].title], specs=[[{"secondary_y": secY}]])
            self._subplots[0].plot(self.fig)
            self.fig.update_layout(title_text=self.title)

        if height is not None:
            self.fig.update_layout(height=height)

        #simple_white layout
        if print:
            self.fig.update_layout(template="simple_white",
                                   font=dict(size=26))


        self.fig.update_layout(**layoutArgs)

        if not returnOnly:
            self.fig.show()
        return self.fig


if __name__ == '__main__':



    x = np.linspace(1,10,200)
    y1 = np.sin(x)
    y2 = np.cos(x)
    df = pd.concat([pd.DataFrame({"x":x,"y":y1,"t":"Sin"}),pd.DataFrame({"x":x,"y":y2,"t":"Cos"})])
    pg = PlotlyGraph("Test")
    pg[0] *= px.line(df, x="x",y="y",color="t",title="Sine and Cosine")
    pg.show()
    k = 0
    pg[0] **= go.Scatter(x=[1,2,3,4,5],y=[0,0.5,0,-0.5,0],mode="markers",name="Test")


    pg[0].setLabels("X V", "Left Y", "Right Y", "Sine and Cosine")
    df = pd.DataFrame({"Type":["A","B","C","A","B","C"],"Color":["R","R","R","G","G","G"],"TypeCount":[1,2,3,3,2,1],})
    df = pd.DataFrame({"Type":["A","B","C","A","B","C"],"Color":["R","R","R","G","G","G"],"TypeCount":[1,2,3,3,2,1],})
    pg = PlotlyGraph("Test")
    pg[0] *= px.bar(df, x="Type",y="TypeCount",color="Color")
    pg[1] *= px.bar(df, x="Type",y="TypeCount",color="Color")
    pg.show(2)
    k = 0

    pg.show(1) #show the graph with 2 columns and shared x axis
    k = 0