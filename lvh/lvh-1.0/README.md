# EchoNet-LVH

EchoNet-LVH is a deep learning workflow that automatically quantifies ventricular hypertrophy with precision equal to human experts and predicts etiology of LVH.  

![Figure1](Figure1.png)

Trained on 28,201 echocardiogram videos, our model accurately measures intraventricular wall thickness (mean absolute error [MAE] 1.4mm, 95% CI 1.2-1.5mm), left ventricular diameter (MAE 2.4mm, 95% CI 2.2-2.6mm), and posterior wall thickness (MAE 1.2mm, 95% CI 1.1-1.3mm) and classifies cardiac amyloidosis (area under the curve of 0.83) and hypertrophic cardiomyopathy (AUC 0.98) from other etiologies of LVH.  

<!-- ## Preprocessing -->

## Inference

### PLAX Hypertrophy Evaluation

Running inference for the PLAX hypertrophy model is possible with ```run_plax_inference.py``` using a simple CLI. The script searches a directory for .avi videos, runs inference on them and outputs results to an output directory. For example, if we have the file structure:

```
PLAX
|-> Videos
| |-> echo1_plax.avi
| |-> echo2_plax.avi
| |-> echo3_plax.avi
```

to run PLAX measurement inference, run

```
python run_plax_inference.py ./PLAX ./Inference
```

For each .avi found in the input directory, a folder with the same name as the file will be created in the output directory. Three files will be created and outputted to this directory: an .avi animation showing the model predictions overlayed on top of the original video. A .csv file with the frame-by-frame predictions, and a .png graph showing the measurements over time. The resulting file structure will be:

```
./
|-> PLAX
| |-> echo1_plax.avi
| |-> echo2_plax.avi
| |-> echo3_plax.avi
|-> Inference
| |-> echo1_plax
| | |-> echo1_plax.avi
| | |-> echo1_plax.csv
| | |-> echo1_plax.png
|-> Inference
| |-> echo2_plax
| | |-> echo2_plax.avi
| | |-> echo2_plax.csv
| | |-> echo2_plax.png
|-> Inference
| |-> echo3_plax
| | |-> echo3_plax.avi
| | |-> echo3_plax.csv
| | |-> echo3_plax.png
```

### A4C Disease Classification

Running binary video classification on A4C view echos is done in a similar fashion. ```run_classification_inference.py``` also expects an input directory and an output directory. The script will search the input directory for .avi videos, run inference on them, and output results to the output directory. Unlike the PLAX script, all inferences for the entire input directory will be saved in a single .csv file. For example if the original file structure is

```
./
|->A4C
| |-> echo1_a4c.avi
| |-> echo2_a4c.avi
| |-> echo3_a4c.avi
```

after running

```bash
python run_classification_inference.py ./A4C ./Inference
```

the resulting file structure will look like

```
./
|->A4C
| |-> echo1_a4c.avi
| |-> echo2_a4c.avi
| |-> echo3_a4c.avi
|->Inference
| |-> A4C.csv
```

For both PLAX and Classification, there are additional parameters that can be configured in the CLI. These can be found via the ```--help``` option.
