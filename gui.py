import sys
import inspect
import numpy as np
import PyQt5  as qt
from PyQt5.QtWidgets import QApplication, QLabel, QCalendarWidget, \
        QLineEdit, QMainWindow, QWidget, QPushButton, QShortcut, \
        QVBoxLayout, QHBoxLayout, QAction, QFileDialog, QComboBox
from functools import partial
from PyQt5.QtCore import pyqtSignal, Qt, QObject, QThread
from PyQt5.QtGui import QKeySequence, QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pyqtgraph as pg
from heat_transfer import main

title = '2D Heat Transfer in PWR Fuel Rod'
ran_at_least_once = False

class WorkerSignals(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal(object)
    

class SolverWorker(QThread):

    def __init__(self, p_i, p_e, p, shape, flow):
        super().__init__()
        self.p_i = p_i
        self.p_e = p_e
        self.p = p
        self.shape = shape
        self.flow = flow
        self.signals = WorkerSignals()
        self.update_num = 1
        self._cancel = False

    def should_cancel(self):
        self._cancel = True

    def run(self):
        def progress_cb():
            self.signals.progress.emit(self.update_num)
            self.update_num += 1
        
        def cancel():
            if self._cancel:
                raise RuntimeError()

        try:
            results = main(self.p_i, self.p_e, self.p, self.shape, self.flow, progress_cb, cancel)
        except:
            self.signals.finished.emit(None)
            return
        self.signals.finished.emit(results)

class Window(QMainWindow):
    Shape_Entered = pyqtSignal(str)
    Initial_Power = pyqtSignal(str)
    End_Power = pyqtSignal(str)
    Flow_Rate = pyqtSignal(str)
    Period = pyqtSignal(str)
    Submit = pyqtSignal()
    Quit = pyqtSignal()
    Next = pyqtSignal()
    New_data = pyqtSignal()
    def __init__(self):
        super().__init__()
        self.setWindowTitle(title)
        self.setup_labels()
        self.setup_canvas()
        
        screen = QApplication.primaryScreen()
        rect = screen.availableGeometry()
        self.setGeometry(rect)
        
    def setup_labels(self):
        self.update1 = QLabel('', self)
        self.update1.hide()
        self.update2 = QLabel('', self)
        self.update2.hide()
        self.update3 = QLabel('', self)
        self.update3.hide()
        self.update4 = QLabel('', self)
        self.update4.hide()
        self.update5 = QLabel('', self)
        self.update5.hide()
        self.update6 = QLabel('', self)
        self.update6.hide()
        self.check_vals = QLabel('', self)
        self.check_vals.hide()
        self.current_values = QLabel('', self)
        self.current_values.hide()
        """Create the initial static labels."""
        self.initial1 = QLabel('This program can take 5 user inputs:', self)
        self.initial1.setFont(QFont("Arial", 14))
        self.initial1.resize(400, 30)
        self.initial1.move(725, 425)
        self.initial1.show()

        self.initial2 = QLabel('    1) Initial Power', self)
        self.initial2.setFont(QFont("Arial", 14))
        self.initial2.resize(400, 30)
        self.initial2.move(725, 455)
        self.initial2.show()

        self.initial3 = QLabel('    2) Final Power', self)
        self.initial3.setFont(QFont("Arial", 14))
        self.initial3.resize(400, 30)
        self.initial3.move(725, 485)
        self.initial3.show()
        
        self.initial4 = QLabel('    3) Reactor Period', self)
        self.initial4.setFont(QFont("Arial", 14))
        self.initial4.resize(400, 30)
        self.initial4.move(725, 515)
        self.initial4.show()
        
        self.initial5 = QLabel('    4) Flux Profile', self)
        self.initial5.setFont(QFont("Arial", 14))
        self.initial5.resize(400, 30)
        self.initial5.move(725, 545)
        self.initial5.show()
        
        self.initial6 = QLabel('    5) Average Flow Rate', self)
        self.initial6.setFont(QFont("Arial", 14))
        self.initial6.resize(400, 30)
        self.initial6.move(725, 575)
        self.initial6.show()
        
        self.initial7 = QLabel('It will then solve a 2-D heat transfer equation in time from initial', self)
        self.initial7.setFont(QFont("Arial", 14))
        self.initial7.resize(600, 30)
        self.initial7.move(725, 605)
        self.initial7.show()
        
        self.initial8 = QLabel('to final power and display the resulting temperature field at ~10 fps.', self)
        self.initial8.setFont(QFont("Arial", 14))
        self.initial8.resize(600, 30)
        self.initial8.move(725, 635)
        self.initial8.show()

        self.next = QPushButton('Understood', self)
        self.next.move(915, 685)
        self.next.show()
        self.next.clicked.connect(self.Main_screen)
        
    def Main_screen(self):
        self.initial1.clear()
        self.initial2.clear()
        self.initial3.clear()
        self.initial4.clear()
        self.initial5.clear()
        self.initial6.clear()
        self.initial7.clear()
        self.initial8.clear()
        self.next.close()
        
        self.P_i = QLineEdit(self)
        self.P_i.setFocus()
        self.P_i.setPlaceholderText('Initial Power')
        self.P_i.setFont(QFont("Arial", 14))
        self.P_i.resize(300, 50)
        self.P_i.move(1500, 600)
        self.P_i.show()
        
        self.P_e = QLineEdit(self)
        self.P_e.setFocus()
        self.P_e.setPlaceholderText('Final Power')
        self.P_e.setFont(QFont("Arial", 14))
        self.P_e.resize(300, 50)
        self.P_e.move(1500, 660)
        self.P_e.show()
        
        self.period = QLineEdit(self)
        self.period.setFocus()
        self.period.setPlaceholderText('Period')
        self.period.setFont(QFont("Arial", 14))
        self.period.resize(300, 50)
        self.period.move(1500, 720)
        self.period.show()
        
        self.flow = QLineEdit(self)
        self.flow.setFocus()
        self.flow.setPlaceholderText('Avg. Flow Velocity')
        self.flow.setFont(QFont("Arial", 14))
        self.flow.resize(300, 50)
        self.flow.move(1500, 780)
        self.flow.show()
        
        self.shape = QComboBox(self)
        self.shape.addItem('Shape')
        self.shape.addItems(['Uniform', 'Middle', 'Inlet', 'Outlet'])
        self.shape.model().item(0).setEnabled(False)
        self.shape.setFont(QFont("Arial", 14))
        self.shape.resize(100, 50)
        self.shape.move(1500, 840)
        self.shape.show()
        
        self.info = QLabel('Image will go here after submission/solve.', self)
        self.info.setFont(QFont("Arial", 14))
        self.info.resize(500, 30)
        self.info.move(300, 525)
        self.info.show()
        
        self.submission = QPushButton('Submit', self)
        self.submission.setFont(QFont("Arial", 14))
        self.submission.move(1615, 900)
        self.submission.resize(100, 50)
        self.submission.show()
        self.submission.clicked.connect(self.values_chose, Qt.UniqueConnection)
        
        self.initial_range = QLabel('Range: 10-10000 MW', self)
        self.initial_range.setFont(QFont("Arial", 14))
        self.initial_range.move(1300, 600)
        self.initial_range.resize(200, 50)
        self.initial_range.show()
        
        self.end_range = QLabel('Range: 10-10000 MW', self)
        self.end_range.setFont(QFont("Arial", 14))
        self.end_range.move(1300, 660)
        self.end_range.resize(200, 50)
        self.end_range.show()
        
        self.period_range = QLabel('Range: 1-60 s', self)
        self.period_range.setFont(QFont("Arial", 14))
        self.period_range.move(1365, 720)
        self.period_range.resize(150, 50)
        self.period_range.show()
        
        self.flow_range = QLabel('Range: 1-10 m/s', self)
        self.flow_range.setFont(QFont("Arial", 14))
        self.flow_range.move(1345, 780)
        self.flow_range.resize(150, 50)
        self.flow_range.show()
        
        self.initial_units = QLabel('MW', self)
        self.initial_units.setFont(QFont("Arial", 14))
        self.initial_units.move(1815, 600)
        self.initial_units.resize(150, 50)
        self.initial_units.show()
        
        self.end_units = QLabel('MW', self)
        self.end_units.setFont(QFont("Arial", 14))
        self.end_units.move(1815, 660)
        self.end_units.resize(150, 50)
        self.end_units.show()
        
        self.period_units = QLabel('seconds', self)
        self.period_units.setFont(QFont("Arial", 14))
        self.period_units.move(1815, 720)
        self.period_units.resize(150, 50)
        self.period_units.show()
        
        self.flow_units = QLabel('m/s', self)
        self.flow_units.setFont(QFont("Arial", 14))
        self.flow_units.move(1815, 780)
        self.flow_units.resize(150, 50)
        self.flow_units.show()
        
        self.inputs = QLabel('Inputs:', self)
        self.inputs.setFont(QFont("Arial", 14))
        self.inputs.move(1500, 540)
        self.inputs.resize(150, 50)
        self.inputs.show()
        
    def values_chose(self):
        self.update1.clear()
        self.update2.clear()
        self.update3.clear()
        self.update4.clear()
        self.update5.clear()
        self.update6.clear()
        self.check_vals.clear()
        self.current_values.clear()
        
        if hasattr(self, 'anim') and self.anim is not None:
            try:
                self.anim.event_source.stop()
                self.anim._stop()
            except Exception:
                pass
            self.anim = None
    
        self.plot_widget.hide()
        self.cbar_panel.hide()
        self.chf_graph.hide()
        self.setup_canvas()
        Power_initial = self.P_i.text()
        Power_end = self.P_e.text()
        Per = self.period.text()
        Flow = self.flow.text()
        Shap = self.shape.currentText()
        
        self.New_data.emit()
        self.Initial_Power.emit(Power_initial)
        self.End_Power.emit(Power_end)
        self.Period.emit(Per)
        self.Flow_Rate.emit(Flow)
        self.Shape_Entered.emit(Shap)
        self.Submit.emit()
        
    def correct_vals_chosen(self, p_i, p_e, p, flow, shape):
        self.P_i.clear()
        self.P_e.clear()
        self.period.clear()
        self.flow.clear()
        self.shape.setCurrentIndex(-1)
        self.current_values = QLabel(f'Current values chosen are: Initial Power = {p_i} MW | Final Power = {p_e} MW |' \
                                     f' Period = {p} s | Average Flow Velocity = {flow} m/s | Flux profile = {shape}', self)
        self.current_values.setFont(QFont("Arial", 14))
        self.current_values.resize(1600, 50)
        self.current_values.move(100, 50)
        self.current_values.show()
        
    def incorrect_vals_chosen(self):
        self.check_vals = QLabel('Check your values, at least one is out of range', self)
        self.check_vals.setFont(QFont("Arial", 14))
        self.check_vals.resize(500, 50)
        self.check_vals.move(1475, 480)
        self.check_vals.show()
        
    def updates(self, update_num):
        self.info.clear()
        if update_num == 1:
            self.update1 = QLabel('Solver solving final steady state temperature field', self)
            self.update1.setFont(QFont("Arial", 14))
            self.update1.resize(500, 30)
            self.update1.move(550, 300)
            self.update1.show()
        elif update_num == 2:
            self.update2 = QLabel('Solver solved final steady state temperature field', self)
            self.update2.setFont(QFont("Arial", 14))
            self.update2.resize(500, 30)
            self.update2.move(550, 320)
            self.update2.show()
        elif update_num == 3:
            self.update3 = QLabel('Solver solving initial steady state temperature field', self)
            self.update3.setFont(QFont("Arial", 14))
            self.update3.resize(500, 30)
            self.update3.move(550, 340)
            self.update3.show()
        elif update_num == 4:
            self.update4 = QLabel('Solver solved initial steady state temperature field', self)
            self.update4.setFont(QFont("Arial", 14))
            self.update4.resize(500, 30)
            self.update4.move(550, 360)
            self.update4.show()
        elif update_num == 5:
            self.update5 = QLabel('Solver solving time dependent temperature field', self)
            self.update5.setFont(QFont("Arial", 14))
            self.update5.resize(500, 30)
            self.update5.move(550, 380)
            self.update5.show()
        elif update_num == 6:
            self.update6 = QLabel('Solver solved time dependent temperature field', self)
            self.update6.setFont(QFont("Arial", 14))
            self.update6.resize(500, 30)
            self.update6.move(550, 400)
            self.update6.show()
        elif update_num == 7:
            self.update1.clear()
            self.update2.clear()
            self.update3.clear()
            self.update4.clear()
            self.update5.clear()
            self.update6.clear()
            
    def setup_canvas(self):
        # Main plot
        self.plot_widget = pg.PlotWidget(self)
        self.plot_widget.setBackground('k')
        self.plot_widget.move(200, 200)
        self.plot_widget.resize(700, 800)   # leave room on right for colorbars
        self.plot_widget.hide()

        # Colorbar panel (safe: no layouts, no parent hacking)
        self.cbar_panel = pg.GraphicsLayoutWidget(self)
        self.cbar_panel.move(900, 200)      # right of plot
        self.cbar_panel.resize(140, 800)
        self.cbar_panel.hide()
        
        self.chf_graph = pg.GraphicsLayoutWidget(self)
        self.chf_graph.move(1450, 150)
        self.chf_graph.resize(400, 400)
        self.chf_graph.hide()
        
    def interupt(self):
        self.info.setText('New values submitted, old simulation stopped')
        self.info.show()
            
    