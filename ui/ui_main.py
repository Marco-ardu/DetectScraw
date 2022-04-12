# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui/main.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1044, 652)
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 170, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 170, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(120, 120, 120))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 170, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        MainWindow.setPalette(palette)
        font = QtGui.QFont()
        font.setFamily("微軟正黑體")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        MainWindow.setFont(font)
        MainWindow.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.LabelFront = QtWidgets.QLabel(self.centralwidget)
        self.LabelFront.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.LabelFront.sizePolicy().hasHeightForWidth())
        self.LabelFront.setSizePolicy(sizePolicy)
        self.LabelFront.setMinimumSize(QtCore.QSize(350, 300))
        self.LabelFront.setMaximumSize(QtCore.QSize(1000, 900))
        font = QtGui.QFont()
        font.setFamily("微軟正黑體")
        font.setPointSize(40)
        font.setBold(True)
        font.setWeight(75)
        self.LabelFront.setFont(font)
        self.LabelFront.setAlignment(QtCore.Qt.AlignCenter)
        self.LabelFront.setObjectName("LabelFront")
        self.horizontalLayout.addWidget(self.LabelFront)
        self.LabelRear = QtWidgets.QLabel(self.centralwidget)
        self.LabelRear.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.LabelRear.sizePolicy().hasHeightForWidth())
        self.LabelRear.setSizePolicy(sizePolicy)
        self.LabelRear.setMinimumSize(QtCore.QSize(350, 300))
        self.LabelRear.setMaximumSize(QtCore.QSize(1000, 900))
        font = QtGui.QFont()
        font.setFamily("微軟正黑體")
        font.setPointSize(40)
        font.setBold(True)
        font.setWeight(75)
        self.LabelRear.setFont(font)
        self.LabelRear.setAlignment(QtCore.Qt.AlignCenter)
        self.LabelRear.setObjectName("LabelRear")
        self.horizontalLayout.addWidget(self.LabelRear)
        self.verticalLayout_8.addLayout(self.horizontalLayout)
        self.line_up = QtWidgets.QFrame(self.centralwidget)
        self.line_up.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_up.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_up.setObjectName("line_up")
        self.verticalLayout_8.addWidget(self.line_up)
        self.remind = QtWidgets.QLabel(self.centralwidget)
        self.remind.setMinimumSize(QtCore.QSize(0, 60))
        font = QtGui.QFont()
        font.setPointSize(40)
        self.remind.setFont(font)
        self.remind.setAlignment(QtCore.Qt.AlignCenter)
        self.remind.setObjectName("remind")
        self.verticalLayout_8.addWidget(self.remind)
        self.line_down = QtWidgets.QFrame(self.centralwidget)
        self.line_down.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_down.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_down.setObjectName("line_down")
        self.verticalLayout_8.addWidget(self.line_down)
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.SaveButton = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.SaveButton.sizePolicy().hasHeightForWidth())
        self.SaveButton.setSizePolicy(sizePolicy)
        self.SaveButton.setMinimumSize(QtCore.QSize(100, 0))
        self.SaveButton.setMaximumSize(QtCore.QSize(400, 16777215))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.SaveButton.setFont(font)
        self.SaveButton.setObjectName("SaveButton")
        self.verticalLayout_2.addWidget(self.SaveButton)
        self.left_label = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.left_label.sizePolicy().hasHeightForWidth())
        self.left_label.setSizePolicy(sizePolicy)
        self.left_label.setMinimumSize(QtCore.QSize(100, 0))
        self.left_label.setMaximumSize(QtCore.QSize(400, 16777215))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.left_label.setFont(font)
        self.left_label.setAlignment(QtCore.Qt.AlignCenter)
        self.left_label.setObjectName("left_label")
        self.verticalLayout_2.addWidget(self.left_label)
        self.right_label = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.right_label.sizePolicy().hasHeightForWidth())
        self.right_label.setSizePolicy(sizePolicy)
        self.right_label.setMinimumSize(QtCore.QSize(100, 0))
        self.right_label.setMaximumSize(QtCore.QSize(400, 16777215))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.right_label.setFont(font)
        self.right_label.setAlignment(QtCore.Qt.AlignCenter)
        self.right_label.setObjectName("right_label")
        self.verticalLayout_2.addWidget(self.right_label)
        self.verticalLayout_2.setStretch(0, 1)
        self.verticalLayout_2.setStretch(1, 1)
        self.verticalLayout_2.setStretch(2, 1)
        self.horizontalLayout_11.addLayout(self.verticalLayout_2)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.exp_time_label = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.exp_time_label.sizePolicy().hasHeightForWidth())
        self.exp_time_label.setSizePolicy(sizePolicy)
        self.exp_time_label.setMinimumSize(QtCore.QSize(90, 0))
        self.exp_time_label.setMaximumSize(QtCore.QSize(400, 16777215))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.exp_time_label.setFont(font)
        self.exp_time_label.setAlignment(QtCore.Qt.AlignCenter)
        self.exp_time_label.setObjectName("exp_time_label")
        self.verticalLayout.addWidget(self.exp_time_label)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.left_exp_time_value = QtWidgets.QSlider(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.left_exp_time_value.sizePolicy().hasHeightForWidth())
        self.left_exp_time_value.setSizePolicy(sizePolicy)
        self.left_exp_time_value.setMinimumSize(QtCore.QSize(100, 0))
        self.left_exp_time_value.setMaximumSize(QtCore.QSize(400, 16777215))
        self.left_exp_time_value.setMinimum(1)
        self.left_exp_time_value.setMaximum(33000)
        self.left_exp_time_value.setProperty("value", 20000)
        self.left_exp_time_value.setOrientation(QtCore.Qt.Horizontal)
        self.left_exp_time_value.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.left_exp_time_value.setTickInterval(1000)
        self.left_exp_time_value.setObjectName("left_exp_time_value")
        self.horizontalLayout_3.addWidget(self.left_exp_time_value)
        self.left_exp_time_edit = QtWidgets.QLineEdit(self.centralwidget)
        self.left_exp_time_edit.setObjectName("left_exp_time_edit")
        self.horizontalLayout_3.addWidget(self.left_exp_time_edit)
        self.horizontalLayout_3.setStretch(0, 5)
        self.horizontalLayout_3.setStretch(1, 1)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.right_exp_time_value = QtWidgets.QSlider(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.right_exp_time_value.sizePolicy().hasHeightForWidth())
        self.right_exp_time_value.setSizePolicy(sizePolicy)
        self.right_exp_time_value.setMinimumSize(QtCore.QSize(100, 0))
        self.right_exp_time_value.setMaximumSize(QtCore.QSize(400, 16777215))
        self.right_exp_time_value.setMinimum(1)
        self.right_exp_time_value.setMaximum(33000)
        self.right_exp_time_value.setProperty("value", 20000)
        self.right_exp_time_value.setOrientation(QtCore.Qt.Horizontal)
        self.right_exp_time_value.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.right_exp_time_value.setTickInterval(1000)
        self.right_exp_time_value.setObjectName("right_exp_time_value")
        self.horizontalLayout_2.addWidget(self.right_exp_time_value)
        self.right_exp_time_edit = QtWidgets.QLineEdit(self.centralwidget)
        self.right_exp_time_edit.setObjectName("right_exp_time_edit")
        self.horizontalLayout_2.addWidget(self.right_exp_time_edit)
        self.horizontalLayout_2.setStretch(0, 5)
        self.horizontalLayout_2.setStretch(1, 1)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.verticalLayout.setStretch(0, 1)
        self.verticalLayout.setStretch(1, 1)
        self.verticalLayout.setStretch(2, 1)
        self.horizontalLayout_11.addLayout(self.verticalLayout)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.sens_ios_label = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sens_ios_label.sizePolicy().hasHeightForWidth())
        self.sens_ios_label.setSizePolicy(sizePolicy)
        self.sens_ios_label.setMinimumSize(QtCore.QSize(70, 0))
        self.sens_ios_label.setMaximumSize(QtCore.QSize(400, 16777215))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.sens_ios_label.setFont(font)
        self.sens_ios_label.setAlignment(QtCore.Qt.AlignCenter)
        self.sens_ios_label.setObjectName("sens_ios_label")
        self.verticalLayout_3.addWidget(self.sens_ios_label)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.left_sens_ios_value = QtWidgets.QSlider(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.left_sens_ios_value.sizePolicy().hasHeightForWidth())
        self.left_sens_ios_value.setSizePolicy(sizePolicy)
        self.left_sens_ios_value.setMinimumSize(QtCore.QSize(100, 0))
        self.left_sens_ios_value.setMaximumSize(QtCore.QSize(400, 16777215))
        self.left_sens_ios_value.setMinimum(1)
        self.left_sens_ios_value.setMaximum(1600)
        self.left_sens_ios_value.setProperty("value", 800)
        self.left_sens_ios_value.setOrientation(QtCore.Qt.Horizontal)
        self.left_sens_ios_value.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.left_sens_ios_value.setTickInterval(80)
        self.left_sens_ios_value.setObjectName("left_sens_ios_value")
        self.horizontalLayout_4.addWidget(self.left_sens_ios_value)
        self.left_sens_ios_edit = QtWidgets.QLineEdit(self.centralwidget)
        self.left_sens_ios_edit.setObjectName("left_sens_ios_edit")
        self.horizontalLayout_4.addWidget(self.left_sens_ios_edit)
        self.horizontalLayout_4.setStretch(0, 5)
        self.horizontalLayout_4.setStretch(1, 1)
        self.verticalLayout_3.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.right_sens_ios_value = QtWidgets.QSlider(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.right_sens_ios_value.sizePolicy().hasHeightForWidth())
        self.right_sens_ios_value.setSizePolicy(sizePolicy)
        self.right_sens_ios_value.setMinimumSize(QtCore.QSize(100, 0))
        self.right_sens_ios_value.setMaximumSize(QtCore.QSize(400, 16777215))
        self.right_sens_ios_value.setMinimum(1)
        self.right_sens_ios_value.setMaximum(1600)
        self.right_sens_ios_value.setProperty("value", 800)
        self.right_sens_ios_value.setOrientation(QtCore.Qt.Horizontal)
        self.right_sens_ios_value.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.right_sens_ios_value.setTickInterval(80)
        self.right_sens_ios_value.setObjectName("right_sens_ios_value")
        self.horizontalLayout_5.addWidget(self.right_sens_ios_value)
        self.right_sens_ios_edit = QtWidgets.QLineEdit(self.centralwidget)
        self.right_sens_ios_edit.setObjectName("right_sens_ios_edit")
        self.horizontalLayout_5.addWidget(self.right_sens_ios_edit)
        self.horizontalLayout_5.setStretch(0, 5)
        self.horizontalLayout_5.setStretch(1, 1)
        self.verticalLayout_3.addLayout(self.horizontalLayout_5)
        self.verticalLayout_3.setStretch(0, 1)
        self.verticalLayout_3.setStretch(1, 1)
        self.verticalLayout_3.setStretch(2, 1)
        self.horizontalLayout_11.addLayout(self.verticalLayout_3)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.lensPos_label = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lensPos_label.sizePolicy().hasHeightForWidth())
        self.lensPos_label.setSizePolicy(sizePolicy)
        self.lensPos_label.setMinimumSize(QtCore.QSize(80, 0))
        self.lensPos_label.setMaximumSize(QtCore.QSize(400, 16777215))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.lensPos_label.setFont(font)
        self.lensPos_label.setAlignment(QtCore.Qt.AlignCenter)
        self.lensPos_label.setObjectName("lensPos_label")
        self.verticalLayout_4.addWidget(self.lensPos_label)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.left_lensPos_value = QtWidgets.QSlider(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.left_lensPos_value.sizePolicy().hasHeightForWidth())
        self.left_lensPos_value.setSizePolicy(sizePolicy)
        self.left_lensPos_value.setMinimumSize(QtCore.QSize(100, 0))
        self.left_lensPos_value.setMaximumSize(QtCore.QSize(400, 16777215))
        self.left_lensPos_value.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.left_lensPos_value.setMinimum(1)
        self.left_lensPos_value.setMaximum(255)
        self.left_lensPos_value.setProperty("value", 156)
        self.left_lensPos_value.setOrientation(QtCore.Qt.Horizontal)
        self.left_lensPos_value.setInvertedAppearance(False)
        self.left_lensPos_value.setInvertedControls(False)
        self.left_lensPos_value.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.left_lensPos_value.setTickInterval(10)
        self.left_lensPos_value.setObjectName("left_lensPos_value")
        self.horizontalLayout_6.addWidget(self.left_lensPos_value)
        self.left_lensPos_edit = QtWidgets.QLineEdit(self.centralwidget)
        self.left_lensPos_edit.setObjectName("left_lensPos_edit")
        self.horizontalLayout_6.addWidget(self.left_lensPos_edit)
        self.horizontalLayout_6.setStretch(0, 6)
        self.horizontalLayout_6.setStretch(1, 1)
        self.verticalLayout_4.addLayout(self.horizontalLayout_6)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.right_lensPos_value = QtWidgets.QSlider(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.right_lensPos_value.sizePolicy().hasHeightForWidth())
        self.right_lensPos_value.setSizePolicy(sizePolicy)
        self.right_lensPos_value.setMinimumSize(QtCore.QSize(100, 0))
        self.right_lensPos_value.setMaximumSize(QtCore.QSize(400, 16777215))
        self.right_lensPos_value.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.right_lensPos_value.setMinimum(1)
        self.right_lensPos_value.setMaximum(255)
        self.right_lensPos_value.setProperty("value", 156)
        self.right_lensPos_value.setOrientation(QtCore.Qt.Horizontal)
        self.right_lensPos_value.setInvertedAppearance(False)
        self.right_lensPos_value.setInvertedControls(False)
        self.right_lensPos_value.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.right_lensPos_value.setTickInterval(10)
        self.right_lensPos_value.setObjectName("right_lensPos_value")
        self.horizontalLayout_7.addWidget(self.right_lensPos_value)
        self.right_lensPos_edit = QtWidgets.QLineEdit(self.centralwidget)
        self.right_lensPos_edit.setObjectName("right_lensPos_edit")
        self.horizontalLayout_7.addWidget(self.right_lensPos_edit)
        self.horizontalLayout_7.setStretch(0, 6)
        self.horizontalLayout_7.setStretch(1, 1)
        self.verticalLayout_4.addLayout(self.horizontalLayout_7)
        self.verticalLayout_4.setStretch(0, 1)
        self.verticalLayout_4.setStretch(1, 1)
        self.verticalLayout_4.setStretch(2, 1)
        self.horizontalLayout_11.addLayout(self.verticalLayout_4)
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.Barcodelabel = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Barcodelabel.sizePolicy().hasHeightForWidth())
        self.Barcodelabel.setSizePolicy(sizePolicy)
        self.Barcodelabel.setMinimumSize(QtCore.QSize(80, 0))
        self.Barcodelabel.setMaximumSize(QtCore.QSize(400, 16777215))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.Barcodelabel.setFont(font)
        self.Barcodelabel.setAlignment(QtCore.Qt.AlignCenter)
        self.Barcodelabel.setObjectName("Barcodelabel")
        self.horizontalLayout_10.addWidget(self.Barcodelabel)
        self.BarCodeValue = QtWidgets.QLineEdit(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.BarCodeValue.sizePolicy().hasHeightForWidth())
        self.BarCodeValue.setSizePolicy(sizePolicy)
        self.BarCodeValue.setMinimumSize(QtCore.QSize(100, 0))
        self.BarCodeValue.setMaximumSize(QtCore.QSize(150, 16777215))
        self.BarCodeValue.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.BarCodeValue.setObjectName("BarCodeValue")
        self.horizontalLayout_10.addWidget(self.BarCodeValue)
        self.verticalLayout_6.addLayout(self.horizontalLayout_10)
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.autoexpleft = QtWidgets.QCheckBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.autoexpleft.sizePolicy().hasHeightForWidth())
        self.autoexpleft.setSizePolicy(sizePolicy)
        self.autoexpleft.setMinimumSize(QtCore.QSize(105, 0))
        self.autoexpleft.setMaximumSize(QtCore.QSize(400, 16777215))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.autoexpleft.setFont(font)
        self.autoexpleft.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.autoexpleft.setAutoFillBackground(False)
        self.autoexpleft.setChecked(True)
        self.autoexpleft.setObjectName("autoexpleft")
        self.horizontalLayout_9.addWidget(self.autoexpleft)
        self.autofocusleft = QtWidgets.QCheckBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.autofocusleft.sizePolicy().hasHeightForWidth())
        self.autofocusleft.setSizePolicy(sizePolicy)
        self.autofocusleft.setMinimumSize(QtCore.QSize(105, 0))
        self.autofocusleft.setMaximumSize(QtCore.QSize(400, 16777215))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.autofocusleft.setFont(font)
        self.autofocusleft.setCheckable(True)
        self.autofocusleft.setChecked(True)
        self.autofocusleft.setTristate(False)
        self.autofocusleft.setObjectName("autofocusleft")
        self.horizontalLayout_9.addWidget(self.autofocusleft)
        self.horizontalLayout_9.setStretch(0, 1)
        self.horizontalLayout_9.setStretch(1, 1)
        self.verticalLayout_6.addLayout(self.horizontalLayout_9)
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.autoexpright = QtWidgets.QCheckBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.autoexpright.sizePolicy().hasHeightForWidth())
        self.autoexpright.setSizePolicy(sizePolicy)
        self.autoexpright.setMinimumSize(QtCore.QSize(105, 0))
        self.autoexpright.setMaximumSize(QtCore.QSize(400, 16777215))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.autoexpright.setFont(font)
        self.autoexpright.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.autoexpright.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.autoexpright.setChecked(True)
        self.autoexpright.setAutoRepeat(False)
        self.autoexpright.setAutoExclusive(False)
        self.autoexpright.setObjectName("autoexpright")
        self.horizontalLayout_8.addWidget(self.autoexpright)
        self.autofocusright = QtWidgets.QCheckBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.autofocusright.sizePolicy().hasHeightForWidth())
        self.autofocusright.setSizePolicy(sizePolicy)
        self.autofocusright.setMinimumSize(QtCore.QSize(105, 0))
        self.autofocusright.setMaximumSize(QtCore.QSize(400, 16777215))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.autofocusright.setFont(font)
        self.autofocusright.setChecked(True)
        self.autofocusright.setObjectName("autofocusright")
        self.horizontalLayout_8.addWidget(self.autofocusright)
        self.verticalLayout_6.addLayout(self.horizontalLayout_8)
        self.verticalLayout_6.setStretch(0, 1)
        self.verticalLayout_6.setStretch(1, 1)
        self.verticalLayout_6.setStretch(2, 1)
        self.horizontalLayout_11.addLayout(self.verticalLayout_6)
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.btnStart = QtWidgets.QPushButton(self.centralwidget)
        self.btnStart.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(3)
        sizePolicy.setHeightForWidth(self.btnStart.sizePolicy().hasHeightForWidth())
        self.btnStart.setSizePolicy(sizePolicy)
        self.btnStart.setMinimumSize(QtCore.QSize(100, 0))
        self.btnStart.setMaximumSize(QtCore.QSize(400, 16777215))
        font = QtGui.QFont()
        font.setFamily("微軟正黑體")
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.btnStart.setFont(font)
        self.btnStart.setAutoRepeatDelay(300)
        self.btnStart.setObjectName("btnStart")
        self.verticalLayout_5.addWidget(self.btnStart)
        self.btnStop = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btnStop.sizePolicy().hasHeightForWidth())
        self.btnStop.setSizePolicy(sizePolicy)
        self.btnStop.setMinimumSize(QtCore.QSize(100, 0))
        self.btnStop.setMaximumSize(QtCore.QSize(400, 16777215))
        font = QtGui.QFont()
        font.setFamily("微軟正黑體")
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.btnStop.setFont(font)
        self.btnStop.setObjectName("btnStop")
        self.verticalLayout_5.addWidget(self.btnStop)
        self.ShutDown = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ShutDown.sizePolicy().hasHeightForWidth())
        self.ShutDown.setSizePolicy(sizePolicy)
        self.ShutDown.setMinimumSize(QtCore.QSize(100, 0))
        self.ShutDown.setMaximumSize(QtCore.QSize(400, 16777215))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.ShutDown.setFont(font)
        self.ShutDown.setObjectName("ShutDown")
        self.verticalLayout_5.addWidget(self.ShutDown)
        self.verticalLayout_5.setStretch(0, 1)
        self.verticalLayout_5.setStretch(1, 1)
        self.verticalLayout_5.setStretch(2, 1)
        self.horizontalLayout_11.addLayout(self.verticalLayout_5)
        self.verticalLayout_7 = QtWidgets.QVBoxLayout()
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.btnAllScreen = QtWidgets.QPushButton(self.centralwidget)
        self.btnAllScreen.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(3)
        sizePolicy.setHeightForWidth(self.btnAllScreen.sizePolicy().hasHeightForWidth())
        self.btnAllScreen.setSizePolicy(sizePolicy)
        self.btnAllScreen.setMinimumSize(QtCore.QSize(100, 0))
        self.btnAllScreen.setMaximumSize(QtCore.QSize(400, 16777215))
        font = QtGui.QFont()
        font.setFamily("微軟正黑體")
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.btnAllScreen.setFont(font)
        self.btnAllScreen.setAutoRepeatDelay(300)
        self.btnAllScreen.setObjectName("btnAllScreen")
        self.verticalLayout_7.addWidget(self.btnAllScreen)
        self.btnNoAllScreen = QtWidgets.QPushButton(self.centralwidget)
        self.btnNoAllScreen.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(3)
        sizePolicy.setHeightForWidth(self.btnNoAllScreen.sizePolicy().hasHeightForWidth())
        self.btnNoAllScreen.setSizePolicy(sizePolicy)
        self.btnNoAllScreen.setMinimumSize(QtCore.QSize(100, 0))
        self.btnNoAllScreen.setMaximumSize(QtCore.QSize(400, 16777215))
        font = QtGui.QFont()
        font.setFamily("微軟正黑體")
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.btnNoAllScreen.setFont(font)
        self.btnNoAllScreen.setAutoRepeatDelay(300)
        self.btnNoAllScreen.setObjectName("btnNoAllScreen")
        self.verticalLayout_7.addWidget(self.btnNoAllScreen)
        self.btnOpenPath = QtWidgets.QPushButton(self.centralwidget)
        self.btnOpenPath.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(3)
        sizePolicy.setHeightForWidth(self.btnOpenPath.sizePolicy().hasHeightForWidth())
        self.btnOpenPath.setSizePolicy(sizePolicy)
        self.btnOpenPath.setMinimumSize(QtCore.QSize(100, 0))
        self.btnOpenPath.setMaximumSize(QtCore.QSize(400, 16777215))
        font = QtGui.QFont()
        font.setFamily("微軟正黑體")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.btnOpenPath.setFont(font)
        self.btnOpenPath.setAutoRepeatDelay(300)
        self.btnOpenPath.setObjectName("btnOpenPath")
        self.verticalLayout_7.addWidget(self.btnOpenPath)
        self.horizontalLayout_11.addLayout(self.verticalLayout_7)
        self.horizontalLayout_11.setStretch(0, 1)
        self.horizontalLayout_11.setStretch(1, 3)
        self.horizontalLayout_11.setStretch(2, 3)
        self.horizontalLayout_11.setStretch(3, 3)
        self.horizontalLayout_11.setStretch(4, 2)
        self.horizontalLayout_11.setStretch(5, 1)
        self.horizontalLayout_11.setStretch(6, 1)
        self.verticalLayout_8.addLayout(self.horizontalLayout_11)
        self.verticalLayout_8.setStretch(0, 8)
        self.verticalLayout_8.setStretch(1, 1)
        self.verticalLayout_8.setStretch(2, 1)
        self.verticalLayout_8.setStretch(3, 1)
        self.verticalLayout_8.setStretch(4, 2)
        MainWindow.setCentralWidget(self.centralwidget)
        self.actionStart = QtWidgets.QAction(MainWindow)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("ui\\../../../Users/Users/Users/xiao\'tang/.designer/test_img/icons8-play-64.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionStart.setIcon(icon)
        self.actionStart.setObjectName("actionStart")
        self.actionStop = QtWidgets.QAction(MainWindow)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("ui\\../../../Users/Users/Users/xiao\'tang/.designer/test_img/icons8-stop-48.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionStop.setIcon(icon1)
        self.actionStop.setObjectName("actionStop")

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "OAK Scraw Detect"))
        self.LabelFront.setText(_translate("MainWindow", "左相机"))
        self.LabelRear.setText(_translate("MainWindow", "右相机"))
        self.remind.setText(_translate("MainWindow", "消息提醒"))
        self.SaveButton.setText(_translate("MainWindow", "保存参数"))
        self.left_label.setText(_translate("MainWindow", "左相机"))
        self.right_label.setText(_translate("MainWindow", "右相机"))
        self.exp_time_label.setText(_translate("MainWindow", "曝光时间"))
        self.left_exp_time_edit.setText(_translate("MainWindow", "20000"))
        self.right_exp_time_edit.setText(_translate("MainWindow", "20000"))
        self.sens_ios_label.setText(_translate("MainWindow", "感光度"))
        self.left_sens_ios_edit.setText(_translate("MainWindow", "800"))
        self.right_sens_ios_edit.setText(_translate("MainWindow", "800"))
        self.lensPos_label.setText(_translate("MainWindow", "焦距"))
        self.left_lensPos_edit.setText(_translate("MainWindow", "156"))
        self.right_lensPos_edit.setText(_translate("MainWindow", "156"))
        self.Barcodelabel.setText(_translate("MainWindow", "条形码"))
        self.autoexpleft.setText(_translate("MainWindow", "自动曝光"))
        self.autofocusleft.setText(_translate("MainWindow", "自动对焦"))
        self.autoexpright.setText(_translate("MainWindow", "自动曝光"))
        self.autofocusright.setText(_translate("MainWindow", "自动对焦"))
        self.btnStart.setText(_translate("MainWindow", "开启相机"))
        self.btnStop.setText(_translate("MainWindow", "关闭相机"))
        self.ShutDown.setText(_translate("MainWindow", "退出程序"))
        self.btnAllScreen.setText(_translate("MainWindow", "开启全屏"))
        self.btnNoAllScreen.setText(_translate("MainWindow", "取消全屏"))
        self.btnOpenPath.setText(_translate("MainWindow", "打开文件夹"))
        self.actionStart.setText(_translate("MainWindow", "Start"))
        self.actionStart.setToolTip(_translate("MainWindow", "<html><head/><body><p>starrrrt</p></body></html>\n"
"                "))
        self.actionStop.setText(_translate("MainWindow", "Stop"))
        self.actionStop.setToolTip(_translate("MainWindow", "<html><head/><body><p>stooop</p></body></html>\n"
"                "))
