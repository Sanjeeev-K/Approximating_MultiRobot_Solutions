# TODO: Should be able to plot both instances and may be solutions on top if
# possible
import plotly.offline as py
import plotly.graph_objs as go
from TO_InstanceReader import InstanceReader
from TO_SolutionReader import Solution_Reader
from InstancePlotter import Instance_Plotter
import os
import numpy as np
import random as rnd


class SolutionPlotter:

    def __init__(self, instance_string, path_to_data_folder=os.getcwd()):
        self.instance_string = instance_string
        self.instance = InstanceReader(instance_string[2:])
        self.sol = Solution_Reader(instance_string)
        self.instance_plotter_obj = Instance_Plotter(instance_string[2:])
        self.compute_path_lengths()

    def edgeTrace(self, S_loc, arcsInOrder):
        colors = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)',
                  'rgb(44, 160, 44)', 'rgb(214, 39, 40)',
                  'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
                  'rgb(227, 119, 194)', 'rgb(127, 127, 127)',
                  'rgb(188, 189, 34)', 'rgb(23, 190, 207)']
        edge_trace = go.Scatter(
            x=[],
            y=[],
            text=[],
            line=dict(width=1, color=colors[rnd.randint(
                0, len(colors) - 1)], dash='dash'),
            hoverinfo='none',
            showlegend=True,
            mode='lines')

        edge_info_trace = go.Scatter(
            text=[],
            x=[],
            y=[],
            mode='markers',
            hoverinfo='text',
            # name = 'Edge Info',
            showlegend=False,
            marker=dict(
                size=12,
                symbol='pentagon-open-dot',
                color='mistyrose',
                line=dict(
                    # color='rgba(217, 217, 217, 0.14)',
                    width=0.5
                ),
                opacity=0.8
            )
        )

        N_loc = {**self.instance.T_loc, **self.instance.D_loc, **self.instance.S_loc, **self.instance.E_loc}
        for arc in arcsInOrder:
            x0, y0 = N_loc.get(arc[0])
            x1, y1 = N_loc.get(arc[1])
            edge_trace['x'] += tuple([x0, x1])
            edge_trace['y'] += tuple([y0, y1])

        allNodesReq = self.instance.T + self.instance.D
        for n0 in allNodesReq:
            for n1 in allNodesReq:
                if n0 != n1:
                    x0, y0 = N_loc.get(n0)
                    x1, y1 = N_loc.get(n1)
                    edge_info_trace['x'] += tuple([(x0 + x1) / 2])
                    edge_info_trace['y'] += tuple([(y0 + y1) / 2])
                    edge_info_trace['text'] += tuple(
                        ["Weight: " + "{0:.2f}".format(np.linalg.norm(np.array([x0, y0]) - np.array([x1, y1])))])
        return edge_trace, edge_info_trace


    def drawArena(self, modelName='', remainingFuel=0):
        task_trace = self.instance_plotter_obj.taskNodesTrace(remainingFuel)
        start_trace = self.instance_plotter_obj.startNodesTrace(self.instance.D_loc)
        # end_trace = endNodesTrace(E_loc)
        data = [task_trace, start_trace]

        for k in self.sol.arcsInOrder:
            edge_trace, node_info_trace = self.edgeTrace(self.instance.D_loc, self.sol.arcsInOrder[k])
            edge_trace.name = str(k)
            data.append(edge_trace)
            data.append(node_info_trace)

        layout = go.Layout(
            title='{}: {} robot(s), {} task(s), {} depot(s), delta={}, Tmax={}<br>{}<br>Seed: {}<br>Total Length: {:.2f}'
            .format(modelName, len(self.instance.K), len(self.instance.T), len(self.instance.D), self.instance.delta,
                    self.instance.T_max, 
                    "<br>".join("{}:{} ({:.2f})".format(k, v, self.path_length[k]) for k, v in self.sol.arcsInOrder.items()), 
                    self.instance.thisSeed, self.total_length),
            hovermode='closest',
            xaxis=dict(
                title='X-Coord',
                range=[-100, 100]
                # ticklen= 5,
                # zeroline= False,
                # gridwidth= 2,
            ),
            yaxis=dict(
                title='Y-Coord',
                range = [-100, 100],
                scaleanchor="x", 
                scaleratio=1
                # ticklen= 5,
                # gridwidth= 2,
            ),
            margin=dict(t=250),
            showlegend=True

        )
        fig = go.Figure(data=data, layout=layout)
        return fig

    def create_plot_and_show(self):
        fig = self.drawArena(self.instance_string)
        # Add unfilled circle to show the Arena
        fig.update_layout(shapes=[go.layout.Shape(
                    type="circle",
                    xref="x",
                    yref="y",
                    x0=100,
                    y0=100,
                    x1=-100,
                    y1=-100,
                    line_color="Black",
                    line_width=1
                )])
        py.plot(fig,filename = 'temp-plot'+str(self.instance_string[-1]))
        # fig.write_image("fig1.png")
    
    def compute_path_lengths(self):
        self.path_length = {k:0 for k in self.instance.K}
        self.total_length = 0
        for k in self.instance.K:
            for arc in self.sol.arcsInOrder[k]:
                self.path_length[k]+=self.instance.c[arc]
            self.total_length+=self.path_length[k]



def main():
    instance_prefix = 'D1T10Delta300Tmax600Iter'
    #no_of_instances = 10
    iter_no_list = [0]
    r_list = [2]
    # iter_no_list = [i for i in range(no_of_instances)]
    # list_of_formulations = [1,2,3,4,5,6,7,8]    
    list_of_formulations = [3]  
    for i in list_of_formulations:
        for j in iter_no_list:
            for r in r_list:
                instance_name = 'F'+str(i)+'R'+str(r) + instance_prefix+str(j)
                # instance_name = 'F'+str(i)+instance_prefix+str(j)
                plot = SolutionPlotter(instance_name)
                plot.create_plot_and_show()


if __name__ == "__main__":
    main()
