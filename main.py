import numpy as np
import sys
from gui import Window, SolverWorker, WorkerSignals
from PyQt5.QtWidgets import QApplication
from heat_transfer import generate_image
worker = None
p_i = p_e = p = flow = 0
shape = ''

def init_globes():
    global update_num
    update_num = 1

def new_input():
    init_globes()

def handle_initial_power(val):
    global p_i
    p_i = val

def handle_end_power(val):
    global p_e
    p_e = val

def handle_period(val):
    global p
    p = val

def handle_flow_rate(val):
    global flow
    flow = val

def handle_shape(val):
    global shape
    shape = val
    
def solve_start():
    global p_i, p_e, p, flow, shape, worker
    if worker is not None and worker.isRunning():
        main_window.interupt()
        worker.should_cancel()
        worker.wait()
        worker=None
    try:
        p_i, p_e, p, flow = map(float, (p_i, p_e, p, flow))
    except:
        worker = None
        main_window.incorrect_vals_chosen()
        return
    if (10 <= p_i <= 10000) and (10 <= p_e <= 10000) and (1 <= flow <= 10) \
        and (1 <= p <= 60) and (shape != ''):
        worker = SolverWorker(p_i, p_e, p, shape, flow)
        worker.signals.progress.connect(main_window.updates)
        worker.signals.finished.connect(make_image)
        worker.start()
        main_window.correct_vals_chosen(p_i, p_e, p, flow, shape)
    else:
        worker = None
        main_window.incorrect_vals_chosen()
    
def make_image(data):
    if data is None:
        return
    main_window.plot_widget.show()
    main_window.cbar_panel.show()
    main_window.chf_graph.show()

    main_window.anim = generate_image(
        main_window.plot_widget,
        main_window.cbar_panel,
        main_window.chf_graph,
        *data
    )

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = Window()
    init_globes()
    main_window.Initial_Power.connect(handle_initial_power)
    main_window.End_Power.connect(handle_end_power)
    main_window.Period.connect(handle_period)
    main_window.Flow_Rate.connect(handle_flow_rate)
    main_window.Shape_Entered.connect(handle_shape)
    main_window.Submit.connect(solve_start)
    main_window.New_data.connect(init_globes)
    main_window.show()
    app.exec_()