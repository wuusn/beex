import sys
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QFileDialog, 
                               QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, 
                               QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget, 
                               QPushButton, QSpinBox, QHeaderView, QMessageBox,QMainWindow)
from PySide6.QtGui import QPixmap
from analyzer.image_overview import image_overview
from analyzer.umap import umap_distribution_analysis
from analyzer.violin_plots import violin_plots_distribution_analysis
from analyzer.cluster import hierarchical_clustering
from feature_extractor import extract_feature
from analyzer.bes import BES

import os
from PIL import Image
from pathlib import Path
import itertools
import pandas as pd
from preprocessor.util import get_paths
import multiprocessing

class ImageDialog(QDialog):
    def __init__(self, title, image_path, parent):
        super().__init__(parent)
        self.setWindowTitle(title)

        # Create a central widget and layout
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)

        # Create a QLabel to display the image
        self.image_label = QLabel()
        self.image_label.setPixmap(QPixmap(image_path))

        # Add the QLabel to the layout
        layout.addWidget(self.image_label)

        self.setLayout(layout)

class AddCohortDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Cohort")
        self.layout = QVBoxLayout(self)

        # Input for cohort name
        self.cohortNameEdit = QLineEdit(self)
        self.cohortNameEdit.setPlaceholderText("Enter Cohort Name")
        self.layout.addWidget(self.cohortNameEdit)

        # Buttons for selecting paths
        self.imagePathButton = QPushButton("Select Image Directory", self)
        self.imagePathButton.clicked.connect(self.selectImagePath)
        self.layout.addWidget(self.imagePathButton)
        self.imagePathLabel = QLabel("No directory selected", self)
        self.layout.addWidget(self.imagePathLabel)

        self.clinicalPathButton = QPushButton("Select Clinical Data Path (Optional)", self)
        self.clinicalPathButton.clicked.connect(self.selectClinicalPath)
        self.layout.addWidget(self.clinicalPathButton)
        self.clinicalPathLabel = QLabel("No file selected", self)
        self.layout.addWidget(self.clinicalPathLabel)

        # Accept and cancel buttons
        self.buttonsLayout = QHBoxLayout()
        self.addButton = QPushButton("Add")
        self.addButton.clicked.connect(self.accept)
        self.addButton.setEnabled(False)
        self.cancelButton = QPushButton("Cancel")
        self.cancelButton.clicked.connect(self.reject)
        self.buttonsLayout.addWidget(self.addButton)
        self.buttonsLayout.addWidget(self.cancelButton)
        self.layout.addLayout(self.buttonsLayout)

    def selectImagePath(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Image Directory")
        if dir_path:
            self.imagePathLabel.setText(dir_path)
            # check if cohortNameEdit.text equals to default
            if self.cohortNameEdit.text() == "":
                self.cohortNameEdit.setText(os.path.basename(dir_path))
            self.addButton.setEnabled(True)

    def selectClinicalPath(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Clinical Data File", "", "Excel files (*.xlsx *.xls)")
        if file_path:
            self.clinicalPathLabel.setText(file_path)

    def getCohortDetails(self):
        return (self.cohortNameEdit.text().strip(), self.imagePathLabel.text(), self.clinicalPathLabel.text())

class BEEX_UI(QDialog):
    def __init__(self):
        super().__init__()

        # combobox for selecting the mode
        self.mode_combobox = QComboBox()
        self.mode_combobox.addItems(['Pathology', 'Radiology'])
        mode_label = QLabel("Feature Mode:")
        mode_label.setBuddy(self.mode_combobox)

        # spinbox for selecting the number of workers
        self.num_worker_spinbox = QSpinBox()
        self.num_worker_spinbox.setValue(8)
        self.num_worker_spinbox.setRange(1, 32)
        self.num_worker_spinbox.setSingleStep(1)
        self.num_worker_spinbox.setMinimumWidth(50)
        worker_label = QLabel("CPU Workers:")
        worker_label.setBuddy(self.num_worker_spinbox)

        # save path
        self.savePathLabel = QLabel("No directory selected")
        self.savePathButton = QPushButton("Select Save Directory")
        self.savePathButton.clicked.connect(self.selectSavePath)
        self.savePathLabel.setBuddy(self.savePathButton)
        
        # layout for the top part of the dialog
        top_layout = QHBoxLayout()
        top_layout.addWidget(mode_label)
        top_layout.addWidget(self.mode_combobox)
        top_layout.addStretch(1)
        top_layout.addWidget(worker_label)
        top_layout.addWidget(self.num_worker_spinbox)
        top_layout.addStretch(1)
        top_layout.addWidget(self.savePathLabel)
        top_layout.addWidget(self.savePathButton)
        top_layout.addStretch(1)


        # cohort table where users add cohort info
        cohort_group = QGroupBox("Cohorts")
        self.cohort_table = QTableWidget(0, 3)  # Start with zero rows and three columns
        self.cohort_table.setHorizontalHeaderLabels(["Cohort Name", "Image Dir", "Clinical Data"])
        # dynamically adjust the column width
        self.cohort_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.cohort_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.cohort_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)

        # add & delete buttons
        self.addButton = QPushButton("Add")
        self.addButton.clicked.connect(self.showAddDialog)
        self.deleteButton = QPushButton("Delete Selected")
        self.deleteButton.clicked.connect(self.deleteSelectedRow)

        buttonLayout = QHBoxLayout()
        buttonLayout.addWidget(self.addButton)
        buttonLayout.addWidget(self.deleteButton)

        middle_layout = QVBoxLayout()
        middle_layout.addWidget(self.cohort_table)
        middle_layout.addLayout(buttonLayout)
        cohort_group.setLayout(middle_layout)

        # analyzers
        analyzer_group = QGroupBox("Analyzers")
        self.image_checkbox = QCheckBox('Image Overviewer')
        self.image_checkbox.setChecked(True)
        self.umap_checkbox = QCheckBox('UMAP')
        self.umap_checkbox.setChecked(True)
        self.violin_checkbox = QCheckBox('Violin Plots')
        self.violin_checkbox.setChecked(True)
        self.cluster_checkbox = QCheckBox('Clustergram')
        self.cluster_checkbox.setChecked(True)
        self.bes_checkbox = QCheckBox('BE Score')
        self.bes_checkbox.setChecked(True)
        self.pvca_checkbox = QCheckBox('PVCA')
        self.pvca_checkbox.setEnabled(False)
        self.pvca_checkbox.stateChanged.connect(self.togglePVCAColumnInput)

        # Line edit for PVCA columns
        self.pvca_columns_edit = QLineEdit()
        self.pvca_columns_edit.setPlaceholderText("PVCA, column names separated by commas")
        self.pvca_columns_edit.setEnabled(False)
        self.pvca_columns_edit.hide()


        analyzer_layout = QVBoxLayout()
        analyzer_sublayout1 = QHBoxLayout()
        analyzer_sublayout2 = QHBoxLayout()
        analyzer_sublayout1.addWidget(self.image_checkbox)
        analyzer_sublayout1.addWidget(self.umap_checkbox)
        analyzer_sublayout1.addWidget(self.violin_checkbox)
        analyzer_sublayout2.addWidget(self.cluster_checkbox)
        analyzer_sublayout2.addWidget(self.bes_checkbox)
        analyzer_sublayout2.addWidget(self.pvca_checkbox)
        analyzer_layout.addLayout(analyzer_sublayout1)
        analyzer_layout.addLayout(analyzer_sublayout2)
        analyzer_layout.addWidget(self.pvca_columns_edit)  # Add the PVCA columns line edit
        analyzer_group.setLayout(analyzer_layout)

        # Run and Quit buttons
        self.runButton = QPushButton("Run")
        self.runButton.clicked.connect(self.run)
        self.stopButton = QPushButton("Stop")
        self.stopButton.setEnabled(False)
        self.stopButton.clicked.connect(self.stop)

        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(self.runButton)
        bottom_layout.addWidget(self.stopButton)

        main_layout = QGridLayout(self)
        main_layout.addLayout(top_layout, 0, 0, 1, 2)
        main_layout.addWidget(cohort_group, 1, 0, 1, 2)
        main_layout.addWidget(analyzer_group, 2, 0, 1, 2)
        # main_layout.addLayout(save_path_layout, 3, 0, 1, 2)
        main_layout.addLayout(bottom_layout, 4, 0, 1, 2)

        self.setWindowTitle("BEEx for Medical Images")
        # self.process = QProcess(self)
    
    def selectSavePath(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Save Directory")
        if dir_path:
            self.savePathLabel.setText(dir_path)

    def togglePVCAColumnInput(self, state):
        if state == 2:
            self.pvca_columns_edit.setEnabled(True)
            self.pvca_columns_edit.show()
        else:
            self.pvca_columns_edit.setEnabled(False)
            self.pvca_columns_edit.hide()


    def showAddDialog(self):
        dialog = AddCohortDialog(self)
        if dialog.exec():
            name, image_path, clinical_path = dialog.getCohortDetails()
            if name and image_path:
                self.addRow(name, image_path, clinical_path)

    def addRow(self, name, image_path, clinical_path):
        row_position = self.cohort_table.rowCount()
        self.cohort_table.insertRow(row_position)
        self.cohort_table.setItem(row_position, 0, QTableWidgetItem(name))
        self.cohort_table.setItem(row_position, 1, QTableWidgetItem(image_path))
        clinical_text = clinical_path if clinical_path != "No file selected" else "None"
        self.cohort_table.setItem(row_position, 2, QTableWidgetItem(clinical_text))

    def deleteSelectedRow(self):
        current_row = self.cohort_table.currentRow()
        if current_row != -1:
            reply = QMessageBox.question(self, 'Delete Row', 'Are you sure you want to delete the selected row?',
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.cohort_table.removeRow(current_row)
    def run(self):
        print('run')
        self.runButton.setEnabled(False)
        mode = self.mode_combobox.currentText()
        n_workers = self.num_worker_spinbox.value()
        cohorts = []
        cohort_names = []
        for row in range(self.cohort_table.rowCount()):
            cohort_name = self.cohort_table.item(row, 0).text()
            image_dir = self.cohort_table.item(row, 1).text()
            clinical_data = self.cohort_table.item(row, 2).text()
            cohorts.append((cohort_name, image_dir, clinical_data))
            cohort_names.append(cohort_name)
        
        analyzers = {
            "Image Overviewer": self.image_checkbox.isChecked(),
            "UMAP": self.umap_checkbox.isChecked(),
            "Violin Plots": self.violin_checkbox.isChecked(),
            "Clustergram": self.cluster_checkbox.isChecked(),
            "BE Score": self.bes_checkbox.isChecked(),
            "PVCA": self.pvca_checkbox.isChecked(),
        }
        pvca_columns = self.pvca_columns_edit.text() if self.pvca_checkbox.isChecked() else None

        # Disable all widgets
        self.setWidgetsEnabled(False)
        self.stopButton.setEnabled(True)


        # Run the analysis
        # default save dir in Desktop
        # save_dir = os.path.join(os.path.expanduser("~"), "Desktop", "BEEx_Results")
        save_dir = self.savePathLabel.text()
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        image_files = {}
        mask_files = {}
        cohort_dirs = []
        for cohort_name, cohort_dir, _ in cohorts:
            cohort_dirs.append(cohort_dir)
            image_files[cohort_name] = sorted(get_paths(cohort_dir))
            mask_files[cohort_name] = [None]*len(image_files[cohort_name])
        for cohort_name, files in image_files.items():
            print(cohort_name, len(files))
        ext = Path(image_files[cohort_name][0]).suffix
        if ext in ['.nii', '.nii.gz', '.dcm', '.gz']:
            mode = 'Radiology'
        print('mode:', mode)
        try:
            features = extract_feature(image_files, mask_files, mode, cohort_dirs, save_dir, n_workers)
            
            if analyzers["Image Overviewer"]:
                res = image_overview(image_files, mode, n_workers, save_dir)      
                im = Image.open(res)
                im.show()           

            if analyzers["UMAP"]:
                res = umap_distribution_analysis(features, save_dir)
                im = Image.open(res)
                im.show()

            if analyzers["Violin Plots"]:
                res = violin_plots_distribution_analysis(features, save_dir)
                im = Image.open(res)
                im.show()

            if analyzers["Clustergram"]:
                res = hierarchical_clustering(features, save_dir)
                im = Image.open(res)
                im.show()

            if analyzers["BE Score"]:
                bes = {'BES':[], 'Value':[]}
                overall_bes = BES(features, len(cohort_names))
                bes['BES'].append('overall')
                bes['Value'].append(round(overall_bes, 4))
                print(f'Overall BES: {round(overall_bes, 4)}')

                pairs = itertools.combinations(cohort_names, 2)
                for cohort1, cohort2 in pairs:
                    pair_features = features[features['Cohort'].isin([cohort1, cohort2])]
                    s = BES(pair_features, 2)
                    bes['BES'].append(f'{cohort1} to {cohort2}')
                    bes['Value'].append(round(s, 4))
                    print(f'{cohort1} to {cohort2} BES:', round(s,4))
                # save bes dict to excel
                df = pd.DataFrame(bes)
                df.to_excel(os.path.join(save_dir, 'BES.xlsx'), index=False)
                print(f'Saving BES to {os.path.join(save_dir, "BES.xlsx")}')
        except Exception as e:
            print(e)
            QMessageBox.critical(self, "Error", str(e))
            self.stop()
            return

        # pop up a window show run successfully
        self.stop()
        QMessageBox.information(self, "Success", f"Analysis completed successfully! Results are saved to the {save_dir}.")
            
            

        # enable all widgets
        self.setWidgetsEnabled(True)
        self.stopButton.setEnabled(False)
        self.runButton.setEnabled(True)

    def setWidgetsEnabled(self, enabled):
        self.mode_combobox.setEnabled(enabled)
        self.num_worker_spinbox.setEnabled(enabled)
        self.addButton.setEnabled(enabled)
        self.deleteButton.setEnabled(enabled)
        self.runButton.setEnabled(enabled)
        self.cohort_table.setEnabled(enabled)
        self.image_checkbox.setEnabled(enabled)
        self.umap_checkbox.setEnabled(enabled)
        self.violin_checkbox.setEnabled(enabled)
        self.cluster_checkbox.setEnabled(enabled)
        self.bes_checkbox.setEnabled(enabled)
        # self.pvca_checkbox.setEnabled(enabled)
        # self.pvca_columns_edit.setEnabled(enabled)

    def stop(self):
        self.setWidgetsEnabled(True)
        self.stopButton.setEnabled(False)

    def selectSavePath(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Save Directory")
        if dir_path:
            self.savePathLabel.setText(dir_path)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    app = QApplication(sys.argv)
    window = BEEX_UI()
    window.show()
    sys.exit(app.exec())
